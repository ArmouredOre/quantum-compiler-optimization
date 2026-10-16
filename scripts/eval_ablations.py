#!/usr/bin/env python
"""Issue #37: Module A ablations (CR on/off, reward-weight sweep) on 5-40 qubit benchmarks.

    python scripts/gen_circuits_40q.py
    python scripts/eval_ablations.py --stage train  --part 0/2 --seeds 0 1 --timesteps 40000 --out docs/rl_training/ablations &
    python scripts/eval_ablations.py --stage train  --part 1/2 --seeds 0 1 --timesteps 40000 --out docs/rl_training/ablations
    python scripts/eval_ablations.py --stage eval   --part 0/2 ...   (and 1/2)
    python scripts/eval_ablations.py --stage report --seeds 0 1 --out docs/rl_training/ablations

Stages are resumable (finished runs / per-run eval files are skipped), so a killed job can be restarted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qco.ir.parser import parse
from qco.modules.rl_scheduler.ablation import (
    BASELINE, VARIANTS, bucket_name, evaluate_run, slug, summarize, table, train_variant,
)
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import load_benchmarks
from qco.modules.rl_scheduler.plots import grouped_bars_svg

CR_OFF_AT_TEST = "CR-trained, CR off at test"


def circuits(a):
    small = load_benchmarks(a.circuits, 20)
    large = load_benchmarks(a.circuits_40q, 64)
    ghz = parse(f"{a.circuits}/ghz_n40.qasm")
    ghz.name = "ghz_n40"
    return small + large + [ghz]


def jobs(a):
    allj = [(v, s) for v in VARIANTS for s in a.seeds]
    i, n = (int(x) for x in a.part.split("/"))
    return [j for k, j in enumerate(allj) if k % n == i]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "eval", "report"], required=True)
    ap.add_argument("--config", default="configs/rl_ppo_5q.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--part", default="0/1", help="i/n: handle every n-th (variant, seed) job")
    ap.add_argument("--out", default="docs/rl_training/ablations")
    ap.add_argument("--circuits", default="benchmarks/circuits")
    ap.add_argument("--circuits-40q", default="benchmarks/circuits_40q")
    a = ap.parse_args(argv)
    out = Path(a.out)
    base = RLConfig.from_yaml(a.config)
    base = RLConfig.from_dict({**base.to_dict(), "total_timesteps": a.timesteps})
    runs, evals = out / "runs", out / "eval"

    if a.stage == "train":
        for v, s in jobs(a):
            print(f"train {v} seed {s}", flush=True)
            train_variant(base, v, s, runs)
        return 0

    if a.stage == "eval":
        evals.mkdir(parents=True, exist_ok=True)
        pool = circuits(a)
        for v, s in jobs(a):
            for tag, kw in ([(v, {})] + ([(CR_OFF_AT_TEST, {"cr_at_test": False})] if v == BASELINE else [])):
                f = evals / f"{slug(tag)}_seed{s}.json"
                if f.exists():
                    continue
                print(f"eval {tag} seed {s}", flush=True)
                res = evaluate_run(runs / f"{slug(v)}_seed{s}" / "final.json", pool, **kw)
                f.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    pool = circuits(a)
    qubits = {c.name: c.num_qubits for c in pool}
    names = list(VARIANTS) + [CR_OFF_AT_TEST]
    results = {v: {s: json.loads((evals / f"{slug(v)}_seed{s}.json").read_text()) for s in a.seeds} for v in names}
    summ = summarize(results, qubits)
    per_circuit = {v: {n: {m: sum(results[v][s][n][m] for s in a.seeds) / len(a.seeds) for m in ("depth_red_pct", "gate_red_pct", "reward")}
                       for n in qubits} for v in names}
    cr_vars = [BASELINE, "CR off (train + run)", CR_OFF_AT_TEST]
    w_vars = [v for v, (g, _) in VARIANTS.items() if g == "weights"] + [BASELINE]
    w_vars = [BASELINE] + [v for v in w_vars if v != BASELINE]
    md = [
        "# Module A ablations (issue #37)", "",
        f"Policies trained on generated 5-qubit circuits ({a.timesteps} timesteps, seeds {a.seeds}, `{a.config}`), run greedily on the "
        "benchmark circuits (`benchmarks/circuits` up to 20 qubits, `benchmarks/circuits_40q` + `ghz_n40` at 40). "
        "Values are mean ± std over seeds of the per-bucket mean; metrics are variant-independent (Stage 6 reward uses its default weights).", "",
        "Bucket sizes: " + ", ".join(f"{b} = {next(iter(summ[BASELINE][b]['n_circuits'] for _ in [0]))} circuits" for b in summ[BASELINE]), "",
    ]
    figs = {}
    for title, vs in (("CR pass on/off", cr_vars), ("Reward-weight sweep", w_vars)):
        md.append(f"## {title}\n")
        for metric, label in (("depth_red_pct", "Depth reduction %"), ("gate_red_pct", "Gate-count reduction %"), ("reward", "Stage 6 scalarized reward")):
            md += [f"### {label}\n", table(summ, metric, vs, 3 if metric == "reward" else 2), ""]
        bks = [b for b in summ[BASELINE]]
        # gate-count reduction is identical across the weight variants (CR does all of it), so plot reward instead
        plotted = (("depth_red_pct", "depth reduction %"), ("gate_red_pct", "gate-count reduction %")) if vs is cr_vars \
            else (("depth_red_pct", "depth reduction %"), ("reward", "Stage 6 scalarized reward"))
        for metric, label in plotted:
            fn = f"{'cr' if vs is cr_vars else 'weights'}_{metric}.svg"
            if metric == "reward":
                label = "Stage 6 reward"
            grouped_bars_svg(f"{title}: {label}", bks, {v: [summ[v][b][metric][0] for b in bks] for v in vs}, label,
                             {v: [summ[v][b][metric][1] for b in bks] for v in vs}, out / fn)
            figs[fn] = None
            md.append(f"![{title} {label}]({fn})\n")
    md += ["## Per circuit (mean over seeds): depth / gate reduction %\n",
           "| Circuit | qubits | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for n in qubits:
        md.append(f"| {n} | {qubits[n]} | " + " | ".join(f"{per_circuit[v][n]['depth_red_pct']:.1f} / {per_circuit[v][n]['gate_red_pct']:.1f}" for v in names) + " |")
    (out / "ablations.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "ablations.json").write_text(json.dumps({"summary": summ, "per_circuit": per_circuit, "seeds": a.seeds}, indent=1), encoding="utf-8")
    print("\n".join(md[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
