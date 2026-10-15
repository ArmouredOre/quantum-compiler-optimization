#!/usr/bin/env python
"""Issue #26: train on 5-qubit circuits, evaluate zero-shot on the 5/10/20-qubit benchmarks.

    python scripts/eval_generalization.py --seeds 0 1 2 --timesteps 100000 \
        --out docs/rl_training --save-checkpoint models/rl_ppo_5q.json

Trains (or reuses ``<out>/gen_runs/seed<k>_{cr,nocr}/final.json``) one CR and one
no-CR policy per seed, then writes ``generalization.{json,md}`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import Policies, load_benchmarks, study, summarize, to_markdown
from qco.modules.rl_scheduler.ppo import load_policy
from qco.modules.rl_scheduler.training import train
from qco.modules.smt_verifier.equivalence import EquivalenceChecker


def _policy(cfg: RLConfig, run_dir: Path):
    if not (run_dir / "final.json").exists():
        train(cfg, run_dir)
    pol, _ = load_policy(run_dir / "final.json")
    return pol


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rl_ppo_5q.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--out", default="docs/rl_training")
    ap.add_argument("--circuits", default="benchmarks/circuits")
    ap.add_argument("--no-ablation", action="store_true", help="skip the no-CR policies")
    ap.add_argument("--save-checkpoint", help="copy the first seed's CR policy here (loadable by RLScheduler.load)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    base = RLConfig.from_yaml(a.config).to_dict()
    if a.timesteps:
        base["total_timesteps"] = a.timesteps
    pols = {}
    for seed in a.seeds:
        cfg = RLConfig.from_dict({**base, "seed": seed})
        cfg_no = RLConfig.from_dict({**base, "seed": seed, "deterministic_reduction": False})
        p = Policies(_policy(cfg, out / "gen_runs" / f"seed{seed}_cr"), cfg)
        if not a.no_ablation:
            p.nocr, p.nocr_config = _policy(cfg_no, out / "gen_runs" / f"seed{seed}_nocr"), cfg_no
        pols[seed] = p

    circuits = load_benchmarks(a.circuits)
    result = study(circuits, pols, verify=EquivalenceChecker(backend="numeric"), log=print)
    summary = summarize(result)
    info = (f"Trained on generated 5-qubit circuits (families clifford_t / cnot_pauli / qaoa / qft, disjoint from `benchmarks/`), "
            f"{base['total_timesteps']} timesteps, `commutation={base['commutation']}`, reward = environment default.")
    (out / "generalization.json").write_text(json.dumps({"summary": summary, "result": result}, indent=1), encoding="utf-8")
    (out / "generalization.md").write_text(to_markdown(result, summary, info), encoding="utf-8")
    if a.save_checkpoint:
        dst = Path(a.save_checkpoint)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "gen_runs" / f"seed{a.seeds[0]}_cr" / "final.json", dst)
    print((out / "generalization.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
