#!/usr/bin/env python
"""Train the Module A RL scheduler.

    python scripts/train_rl.py --config configs/rl_smoke.yaml --out runs/rl_smoke
    python scripts/train_rl.py --config configs/rl_ppo_5q.yaml --out runs/rl_5q --compare
"""

from __future__ import annotations

import argparse
import json

from qco.modules.rl_scheduler.config import RLConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML RLConfig (default: library defaults)")
    ap.add_argument("--out", default="runs/rl", help="output directory (metrics.csv, checkpoints, curves.svg)")
    ap.add_argument("--timesteps", type=int, help="override total_timesteps")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--backend", choices=["builtin", "sb3"], help="override backend")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/latest.json (builtin backend)")
    ap.add_argument("--compare", action="store_true", help="after training, compare agent vs CR-only/random/greedy and verify equivalence")
    a = ap.parse_args(argv)

    cfg = RLConfig.from_yaml(a.config) if a.config else RLConfig()
    d = cfg.to_dict()
    for k, v in (("total_timesteps", a.timesteps), ("seed", a.seed), ("backend", a.backend)):
        if v is not None:
            d[k] = v
    cfg = RLConfig.from_dict(d)

    if cfg.backend == "sb3":
        from qco.modules.rl_scheduler.sb3_backend import train_sb3

        train_sb3(cfg, a.out)
        return 0

    from qco.modules.rl_scheduler.training import compare, train

    policy, _ = train(cfg, a.out, resume=a.resume)
    if a.compare:
        print(json.dumps(compare(cfg, policy, verify=True), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
