#!/usr/bin/env python
"""Issue #29: train one policy per device profile and write the cross-backend matrix.

    python scripts/eval_backend_matrix.py --seeds 0 1 2 --timesteps 60000 --out docs/rl_training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qco.evaluation.calibration import list_calibrations
from qco.evaluation.hardware import list_hardware
from qco.modules.rl_scheduler.backend_matrix import DEFAULT_PROFILES, cross_matrix, to_markdown, train_backend_policies
from qco.modules.rl_scheduler.circuits import sample_circuits
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.ppo import save_policy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rl_ppo_5q.yaml")
    ap.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES), help=f"calibrations: {', '.join(list_calibrations())}; hardware profiles: {', '.join(list_hardware())}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--n-eval", type=int, default=60, help="held-out circuits (own seed, disjoint from training)")
    ap.add_argument("--out", default="docs/rl_training")
    ap.add_argument("--no-lookahead", action="store_true",
                    help="hide the calibrated one-step lookahead from the policy so it must learn the backend from reward alone")
    ap.add_argument("--models", help="directory to save each policy as <profile>_seed<k>.json")
    a = ap.parse_args(argv)

    base = RLConfig.from_yaml(a.config)
    if a.timesteps:
        base = RLConfig.from_dict({**base.to_dict(), "total_timesteps": a.timesteps})
    if a.no_lookahead:
        base = RLConfig.from_dict({**base.to_dict(), "lookahead": False})
    pols = train_backend_policies(base, tuple(a.profiles), tuple(a.seeds), log=None)
    pool = sample_circuits(base.train_qubits[0], a.n_eval, 4242, base.circuit_families, base.circuit_layers)
    res = cross_matrix(pols, pool, tuple(a.profiles))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "backend_matrix.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "backend_matrix.md").write_text(to_markdown(res), encoding="utf-8")
    if a.models:
        for p in pols:
            save_policy(p.policy, Path(a.models) / f"{p.calibration}_seed{p.seed}.json", {"config": p.config.to_dict()})
    print((out / "backend_matrix.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
