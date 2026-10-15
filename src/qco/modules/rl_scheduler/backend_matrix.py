"""Module A — per-backend training and the cross-backend matrix (novelty N4, issue #29).

One policy is trained per device profile (``qco.evaluation.calibration``): the
environment reward adds a calibrated execution-time term (``RewardWeights.time``)
and the policy's lookahead feature is the calibrated time saved
(``lookahead_metric="time"``), so the learner can see what its backend values.
``cross_matrix`` then runs every policy under every backend's calibration and
reports mean execution-time reduction (%), plus how often two backends' policies
produce *different* schedules.

The study is honest by construction: it reports mean ± std over training seeds
and counts per-seed wins, because backend differences can be smaller than seed
noise (see ``docs/rl_training/backend_matrix.md`` for the measured result).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, pstdev

from qco.evaluation.hardware import HARDWARE_PROFILES, get_hardware, hardware_execution_time
from qco.evaluation.metrics import execution_time
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler.baselines import greedy_depth_chooser, rollout
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.environment import RewardWeights
from qco.modules.rl_scheduler.ppo import PPOPolicy, run_policy
from qco.modules.rl_scheduler.training import train

DEFAULT_PROFILES = ("superconducting_ibm_like", "trapped_ion_like")


def retarget(config: RLConfig, name: str) -> RLConfig:
    """``config`` pointed at backend ``name``: a ``HardwareProfile`` if it is one, else a ``Calibration``."""
    d = config.to_dict()
    if name in HARDWARE_PROFILES:
        d.update(hardware=name, calibration=get_hardware(name).calibration)
    else:
        d.update(hardware=None, calibration=name)
    return RLConfig.from_dict(d)


def backend_config(base: RLConfig, calibration: str, seed: int, time_weight: float = 1.0) -> RLConfig:
    """``base`` specialised to one backend (a ``Calibration`` or ``HardwareProfile`` name): time-aware reward + lookahead."""
    d = retarget(base, calibration).to_dict()
    w = dict(d["weights"])
    w.update(depth=0.0, time=time_weight)
    d.update(lookahead_metric="time", weights=w, seed=seed)
    return RLConfig.from_dict(d)


@dataclass(slots=True)
class BackendPolicy:
    calibration: str
    seed: int
    policy: PPOPolicy
    config: RLConfig


def train_backend_policies(base: RLConfig, profiles=DEFAULT_PROFILES, seeds=(0, 1, 2), out_dir=None, log=None) -> list[BackendPolicy]:
    out = []
    for prof in profiles:
        for seed in seeds:
            cfg = backend_config(base, prof, seed)
            pol, _ = train(cfg, None if out_dir is None else f"{out_dir}/{prof}_seed{seed}", log=log)
            out.append(BackendPolicy(prof, seed, pol, cfg))
    return out


def time_of(circuit: IntermediateRepresentation, backend: str) -> float:
    """Execution time (ns) under a hardware profile or a calibration, whichever ``backend`` names."""
    if backend in HARDWARE_PROFILES:
        return hardware_execution_time(circuit, backend)
    return execution_time(circuit, backend).value


def time_reduction_pct(before: IntermediateRepresentation, after: IntermediateRepresentation, calibration: str) -> float:
    t0 = time_of(before, calibration)
    return 100.0 * (t0 - time_of(after, calibration)) / t0 if t0 else 0.0


def cross_matrix(policies: list[BackendPolicy], circuits: list[IntermediateRepresentation], profiles=DEFAULT_PROFILES) -> dict:
    """Train-on-A / test-on-B matrix of mean execution-time reduction (%).

    ``matrix[test][train] = {"mean", "std_over_seeds", "per_seed"}``; ``greedy_time``
    is the one-step calibrated-time greedy heuristic (no learning) on the same circuits.
    ``wins_own_backend[test]`` is True when the policy trained on ``test`` has the best
    mean in that column; ``own_wins_per_seed`` counts seeds where it beats every other
    profile's policy of the same seed.
    """
    seeds = sorted({p.seed for p in policies})
    by = {(p.calibration, p.seed): p for p in policies}
    matrix: dict = {}
    outputs: dict = {}
    for test in profiles:
        matrix[test] = {}
        for train_prof in profiles:
            per_seed = []
            for s in seeds:
                p = by[(train_prof, s)]
                cfg = retarget(p.config, test)
                reds = []
                for c in circuits:
                    out, _ = run_policy(p.policy, c, cfg, greedy=True)
                    outputs[(test, train_prof, s, c.name)] = out.gates
                    reds.append(time_reduction_pct(c, out, test))
                per_seed.append(mean(reds))
            matrix[test][train_prof] = {"mean": mean(per_seed), "std_over_seeds": pstdev(per_seed), "per_seed": per_seed}
        cfg = backend_config(policies[0].config, test, 0)
        matrix[test]["greedy_time"] = {
            "mean": mean(time_reduction_pct(c, rollout(c, cfg, greedy_depth_chooser(random.Random(0))), test) for c in circuits)
        }
    wins = {t: max(profiles, key=lambda a: matrix[t][a]["mean"]) == t for t in profiles}
    own_wins = {}
    for t in profiles:
        own_wins[t] = sum(
            all(matrix[t][t]["per_seed"][i] >= matrix[t][o]["per_seed"][i] for o in profiles if o != t) for i in range(len(seeds))
        )
    pairs = [(a, b) for i, a in enumerate(profiles) for b in profiles[i + 1:]]
    differ = [
        mean(1.0 if outputs[(t, a, s, c.name)] != outputs[(t, b, s, c.name)] else 0.0 for c in circuits)
        for t in profiles
        for s in seeds
        for a, b in pairs
    ]
    return {
        "profiles": list(profiles), "seeds": seeds, "n_circuits": len(circuits), "matrix": matrix,
        "wins_own_backend": wins, "own_wins_per_seed": own_wins,
        "fraction_schedules_differ": mean(differ),
    }


def to_markdown(res: dict, note: str = "") -> str:
    P = res["profiles"]
    out = ["# Cross-backend matrix: train on A, test on B (issue #29)", "",
           f"Metric: mean execution-time reduction (%) under the *test* backend's calibration, {res['n_circuits']} held-out 5-qubit "
           f"circuits, mean ± std over seeds {res['seeds']}. Rows = test backend, columns = backend the policy was trained on.", "",
           "| test \\ train | " + " | ".join(P) + " | greedy one-step time heuristic |", "|---|" + "---|" * (len(P) + 1)]
    for t in P:
        cells = []
        for a in P:
            m = res["matrix"][t][a]
            best = a == t
            txt = f"{m['mean']:.2f} ± {m['std_over_seeds']:.2f}"
            cells.append(f"**{txt}**" if best else txt)
        out.append(f"| {t} | " + " | ".join(cells) + f" | {res['matrix'][t]['greedy_time']['mean']:.2f} |")
    out += ["", f"* Policy wins on its own backend (mean): {res['wins_own_backend']}",
            f"* Seeds (out of {len(res['seeds'])}) where the own-backend policy beats the other profile's policy: {res['own_wins_per_seed']}",
            f"* Fraction of (circuit, test-backend, seed) cases where the two backends' policies emit different schedules: {res['fraction_schedules_differ']:.2f}", ""]
    if note:
        out += [note, ""]
    return "\n".join(out)
