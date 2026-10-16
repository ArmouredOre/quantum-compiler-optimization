"""Module A ablations (#37): CR pass on/off and the reward-weight sweep.

Every variant is a small override of the 5-qubit training config (``configs/rl_ppo_5q.yaml``); the
trained policies are then run, greedily, on benchmark circuits of every size (5-40 qubits) and scored
with fixed, variant-independent metrics (depth / gate-count reduction and the Stage 6 scalarized
reward with its default weights), so variants trained for different objectives stay comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from qco.evaluation.metrics import scalarized_reward
from qco.ir import IntermediateRepresentation
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import _pct
from qco.modules.rl_scheduler.ppo import PPOPolicy, load_policy, run_policy
from qco.modules.rl_scheduler.training import train

# name -> (group, RLConfig overrides). Weights are (depth, gate_count, fidelity, terminal).
_W = lambda d, g, f, t: {"weights": {"depth": d, "gate_count": g, "fidelity": f, "terminal": t, "time": 0.0}}  # noqa: E731
VARIANTS: dict[str, tuple[str, dict]] = {
    "baseline (CR on)": ("cr", {}),
    "CR off (train + run)": ("cr", {"deterministic_reduction": False}),
    "depth only": ("weights", _W(1.0, 0.0, 0.0, 1.0)),
    "gates only": ("weights", _W(0.0, 1.0, 0.0, 1.0)),
    "no fidelity term": ("weights", _W(1.0, 1.0, 0.0, 1.0)),
    "fidelity x4": ("weights", _W(1.0, 1.0, 2.0, 1.0)),
    "no terminal bonus": ("weights", _W(1.0, 1.0, 0.5, 0.0)),
    "terminal x3": ("weights", _W(1.0, 1.0, 0.5, 3.0)),
}
BASELINE = "baseline (CR on)"


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def variant_config(base: RLConfig, name: str, seed: int) -> RLConfig:
    d = base.to_dict()
    over = VARIANTS[name][1]
    for k, v in over.items():
        d[k] = {**d.get(k, {}), **v} if isinstance(v, dict) else v
    d["seed"] = seed
    return RLConfig.from_dict(d).validate()


def train_variant(base: RLConfig, name: str, seed: int, out_root: str | Path, log=None) -> Path:
    """Train (or reuse) ``out_root/<slug>_seed<k>/final.json`` and return its path."""
    run = Path(out_root) / f"{slug(name)}_seed{seed}"
    if not (run / "final.json").exists():
        train(variant_config(base, name, seed), run, log=log)
    return run / "final.json"


def evaluate_policy(
    policy: PPOPolicy, config: RLConfig, circuits: list[IntermediateRepresentation], *, cr_at_test: bool | None = None
) -> dict[str, dict]:
    """Run ``policy`` greedily on each circuit -> ``{name: {depth_red_pct, gate_red_pct, reward, seconds}}``.

    ``cr_at_test`` overrides the config's CR flag at run time only (the "CR-trained, CR off at test" cell).
    """
    cfg = config
    if cr_at_test is not None:
        cfg = RLConfig.from_dict({**config.to_dict(), "deterministic_reduction": cr_at_test})
    out = {}
    for c in circuits:
        t = time.time()
        after = run_policy(policy, c, cfg)[0]
        out[c.name] = {
            "depth_red_pct": _pct(c.depth(), after.depth()),
            "gate_red_pct": _pct(c.gate_count(), after.gate_count()),
            "reward": scalarized_reward(c, after, calibration="default"),
            "seconds": time.time() - t,
        }
    return out


def evaluate_run(final_json: str | Path, circuits, cr_at_test: bool | None = None) -> dict[str, dict]:
    policy, payload = load_policy(final_json)
    return evaluate_policy(policy, RLConfig.from_dict(payload["config"]), circuits, cr_at_test=cr_at_test)


@dataclass(slots=True)
class Bucket:
    name: str
    lo: int
    hi: int


BUCKETS = (Bucket("5-6q", 0, 6), Bucket("8-12q", 7, 12), Bucket("16-20q", 13, 20), Bucket("40q", 21, 64))


def bucket_name(n: int) -> str:
    return next(b.name for b in BUCKETS if b.lo <= n <= b.hi)


def summarize(results: dict[str, dict[int, dict[str, dict]]], qubits: dict[str, int]) -> dict:
    """``results[variant][seed][circuit]`` -> ``{variant: {bucket: {metric: (mean, std_over_seeds)}}}``.

    A seed's bucket value is the mean over that bucket's circuits; the std is over seeds.
    """
    out: dict = {}
    for var, by_seed in results.items():
        out[var] = {}
        for b in BUCKETS:
            names = [n for n, q in qubits.items() if b.lo <= q <= b.hi]
            if not names or not all(n in next(iter(by_seed.values())) for n in names):
                continue
            cell = {}
            for metric in ("depth_red_pct", "gate_red_pct", "reward"):
                per_seed = [mean(r[n][metric] for n in names) for r in by_seed.values()]
                cell[metric] = (mean(per_seed), pstdev(per_seed) if len(per_seed) > 1 else 0.0)
            cell["n_circuits"] = len(names)
            out[var][b.name] = cell
    return out


def table(summary: dict, metric: str, variants: list[str], digits: int = 2) -> str:
    buckets = [b.name for b in BUCKETS if any(b.name in summary.get(v, {}) for v in variants)]
    lines = ["| Variant | " + " | ".join(buckets) + " |", "|---|" + "---|" * len(buckets)]
    for v in variants:
        cells = []
        for b in buckets:
            c = summary.get(v, {}).get(b)
            cells.append("n/a" if c is None else f"{c[metric][0]:.{digits}f} ± {c[metric][1]:.{digits}f}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
