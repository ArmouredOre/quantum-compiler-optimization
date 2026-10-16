#!/usr/bin/env python
"""Issue #35: retrain Module A against the real Stage 6 reward and compare with the proxy.

Stages (run `train-proxy` and `train-pipeline` in parallel, then `eval`):

    python scripts/eval_closed_loop.py --stage train-proxy    --seeds 0 1 2 --timesteps 100000 --out docs/rl_training/closed_loop
    python scripts/eval_closed_loop.py --stage train-pipeline --seeds 0 1 2 --timesteps 100000 --out docs/rl_training/closed_loop
    python scripts/eval_closed_loop.py --stage eval           --seeds 0 1 2 --out docs/rl_training/closed_loop

`all` runs the three in sequence. Both variants log the end-to-end pipeline reward at every
held-out evaluation (`eval_pipeline_reward` column), so the reward curves are comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

from qco.modules.rl_scheduler.circuits import sample_circuits
from qco.modules.rl_scheduler.closed_loop import (
    evaluate_end_to_end,
    summarize_e2e,
    write_reward_curves,
)
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import bucket_of, load_benchmarks
from qco.modules.rl_scheduler.ppo import load_policy
from qco.modules.rl_scheduler.training import train

warnings.simplefilter("ignore")


def cfg_for(base: dict, variant: str, seed: int, timesteps: int | None) -> RLConfig:
    d = dict(base, seed=seed, eval_pipeline_reward=True, reward_source="pipeline" if variant == "pipeline" else "proxy")
    if timesteps:
        d["total_timesteps"] = timesteps
    return RLConfig.from_dict(d)


def load_history(run: Path) -> list[dict]:
    rows = list(csv.DictReader(open(run / "metrics.csv", encoding="utf-8")))
    num = lambda v: float(v) if v not in ("", None) else ""  # noqa: E731
    return [{k: (v if k == "iteration" else num(v)) for k, v in r.items()} for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train-proxy", "train-pipeline", "eval", "all"], default="all")
    ap.add_argument("--config", default="configs/rl_ppo_5q.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--out", default="docs/rl_training/closed_loop")
    ap.add_argument("--circuits", default="benchmarks/circuits")
    ap.add_argument("--n-heldout", type=int, default=60)
    ap.add_argument("--calibration", default="default")
    ap.add_argument("--save-policy", help="copy the first pipeline-trained seed's final.json here")
    a = ap.parse_args(argv)

    out = Path(a.out)
    base = RLConfig.from_yaml(a.config).to_dict()
    base["calibration"] = a.calibration

    for variant in ("proxy", "pipeline"):
        if a.stage in (f"train-{variant}", "all"):
            for s in a.seeds:
                run = out / f"{variant}_seed{s}"
                if not (run / "final.json").exists():
                    train(cfg_for(base, variant, s, a.timesteps), run)

    if a.stage in ("eval", "all"):
        cfg0 = RLConfig.from_dict({**base, "seed": 0})
        heldout = sample_circuits(cfg0.train_qubits[0], a.n_heldout, 4242, cfg0.circuit_families, cfg0.circuit_layers)
        for i, c in enumerate(heldout):
            c.name = f"heldout_{i}"
        bench = load_benchmarks(a.circuits)
        sets = {"heldout_5q": heldout}
        for b in ("5-6", "8-12", "16-20"):
            sets[f"bench_{b}"] = [c for c in bench if bench_of(c) == b]
        methods: dict[str, dict] = {"stub": {}}
        for variant in ("proxy", "pipeline"):
            methods[variant] = {}
        res: dict = {m: {} for m in methods}
        for name, circuits in sets.items():
            stub = evaluate_end_to_end(circuits, rl_policy_path=None, calibration=a.calibration)
            res["stub"][name] = {"seeds": [summarize_e2e(stub)]}
            for variant in ("proxy", "pipeline"):
                per_seed = []
                for s in a.seeds:
                    pol, payload = load_policy(out / f"{variant}_seed{s}" / "final.json")
                    rc = RLConfig.from_dict(payload["config"])
                    per_seed.append(summarize_e2e(evaluate_end_to_end(circuits, policy=pol, rl_config=rc, calibration=a.calibration)))
                res[variant][name] = {"seeds": per_seed}
            print(name, {m: round(mean(x["reward"] for x in res[m][name]["seeds"]), 4) for m in res})
        for m in res:
            for name, v in res[m].items():
                for k in ("reward", "depth_red_pct", "gate_red_pct"):
                    xs = [x[k] for x in v["seeds"]]
                    v[k] = mean(xs)
                    v[k + "_std"] = pstdev(xs) if len(xs) > 1 else 0.0
        (out / "closed_loop.json").write_text(json.dumps({"seeds": a.seeds, "sets": {k: len(v) for k, v in sets.items()}, "results": res}, indent=1))
        proxy_h = [load_history(out / f"proxy_seed{s}") for s in a.seeds]
        pipe_h = [load_history(out / f"pipeline_seed{s}") for s in a.seeds]
        write_reward_curves(proxy_h, pipe_h, out / "reward_curves.svg")
        (out / "closed_loop.md").write_text(to_markdown(res, sets, a.seeds, a.calibration), encoding="utf-8")
        if a.save_policy:
            import shutil

            Path(a.save_policy).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out / f"pipeline_seed{a.seeds[0]}" / "final.json", a.save_policy)
        print((out / "closed_loop.md").read_text(encoding="utf-8"))
    return 0


def bench_of(c) -> str:
    return bucket_of(c.num_qubits)


def to_markdown(res: dict, sets: dict, seeds, calibration: str) -> str:
    names = {"stub": "untrained greedy stand-in", "proxy": "PPO, proxy reward (as #26)", "pipeline": "**PPO, real Stage 6 reward**"}
    out = ["# Closed-loop reward: proxy vs real Stage 6 reward (issue #35)", "",
           f"End-to-end `HybridOptimizer` runs (Module A in the loop, rule-based Module B, no SMT re-synthesis), calibration `{calibration}`, seeds {list(seeds)}. "
           "Reward = Stage 6 `scalarized_reward(original, pipeline output)` (mean over circuits; mean ± std over seeds). "
           "Policies are the final checkpoints of 5-qubit training.", "",
           "| Method | " + " | ".join(f"{k} (n={len(v)})" for k, v in sets.items()) + " |", "|---|" + "---|" * len(sets)]
    for m, label in names.items():
        out.append(f"| {label} | " + " | ".join(f"{res[m][k]['reward']:.4f} ± {res[m][k]['reward_std']:.4f}" for k in sets) + " |")
    for metric, title in (("depth_red_pct", "Depth reduction %"), ("gate_red_pct", "Gate-count reduction %")):
        out += ["", f"## {title}", "", "| Method | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
        for m, label in names.items():
            out.append(f"| {label} | " + " | ".join(f"{res[m][k][metric]:.2f} ± {res[m][k][metric + '_std']:.2f}" for k in sets) + " |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
