"""Module A — zero-shot generalization study (issue #26).

Policies are trained on 5-qubit circuits only (``training.py``) and then run,
unchanged, on the benchmark circuits in ``benchmarks/circuits`` at 5 -> 10 -> 20
qubits. Because the policy scores per-action feature vectors, the same weights
apply to any circuit size.

Methods compared (all report reduction vs the *raw* benchmark circuit):

``stub``          the Phase 2 greedy ``RLScheduler().schedule`` (adjacent disjoint swaps)
``cr_only``       the deterministic CR pass alone
``greedy``        one-step-lookahead greedy-depth heuristic on top of CR
``agent_cr``      trained policy + CR (same step budget as training)
``agent_cr_x``    same, with the step budget scaled to the circuit size
``agent_nocr``    policy trained *and* run without CR (ablation)
``agent_cr_off``  the CR-trained policy run with CR disabled at test time

``max_episode_steps`` is 40 in training. Larger circuits need more moves, so
``agent_cr_x`` uses ``40 * n_qubits / 5`` steps.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.rl_scheduler.agent import RLScheduler
from qco.modules.rl_scheduler.baselines import EQUIV_QUBIT_CAP, greedy_depth_chooser, rollout
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.cr_pass import reduce_circuit
from qco.modules.rl_scheduler.ppo import PPOPolicy, run_policy

METHODS = ("stub", "cr_only", "greedy", "agent_cr", "agent_cr_x", "agent_nocr", "agent_cr_off")
BUCKETS = (("5-6", 0, 6), ("8-12", 7, 12), ("16-20", 13, 20), ("40", 21, 64))
SUITE_PREFIXES = ("rand_cliffordt", "rand_cnotpauli", "qft", "qaoa", "ghz")
MAX_QUBITS = 20


@dataclass(slots=True)
class Policies:
    """Trained policies for one seed: with-CR (main) and without-CR (ablation)."""

    cr: PPOPolicy
    cr_config: RLConfig
    nocr: PPOPolicy | None = None
    nocr_config: RLConfig | None = None


def load_benchmarks(circuits_dir: str | Path, max_qubits: int = MAX_QUBITS) -> list[IntermediateRepresentation]:
    """Benchmark circuits of the supported families with at most ``max_qubits`` qubits."""
    out = []
    for f in sorted(Path(circuits_dir).glob("*.qasm")):
        if not f.stem.startswith(SUITE_PREFIXES):
            continue
        ir = from_qasm2(f.read_text(encoding="utf-8"))
        ir.name = f.stem
        if ir.num_qubits <= max_qubits:
            out.append(ir)
    return out


def bucket_of(num_qubits: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= num_qubits <= hi:
            return name
    return "other"


def _pct(a: int, b: int) -> float:
    return 100.0 * (a - b) / a if a else 0.0


def _with(config: RLConfig, **kw) -> RLConfig:
    d = config.to_dict()
    d.update(kw)
    return RLConfig.from_dict(d)


def run_methods(circuit: IntermediateRepresentation, pol: Policies, seed: int = 0, verify=None) -> dict[str, dict]:
    """All methods on one circuit -> ``{method: {depth_red_pct, gate_red_pct, seconds[, equivalent]}}``."""
    n = circuit.num_qubits
    scaled = max(pol.cr_config.max_episode_steps, pol.cr_config.max_episode_steps * n // 5)
    runs = {
        "stub": lambda: RLScheduler().schedule(circuit),
        "cr_only": lambda: reduce_circuit(circuit),
        "greedy": lambda: rollout(circuit, pol.cr_config, greedy_depth_chooser(random.Random(seed))),
        "agent_cr": lambda: run_policy(pol.cr, circuit, pol.cr_config)[0],
        "agent_cr_x": lambda: run_policy(pol.cr, circuit, _with(pol.cr_config, max_episode_steps=scaled))[0],
        "agent_cr_off": lambda: run_policy(pol.cr, circuit, _with(pol.cr_config, deterministic_reduction=False))[0],
    }
    if pol.nocr is not None:
        runs["agent_nocr"] = lambda: run_policy(pol.nocr, circuit, pol.nocr_config)[0]
    res = {}
    for name, fn in runs.items():
        t = time.time()
        out = fn()
        r = {
            "depth_red_pct": _pct(circuit.depth(), out.depth()),
            "gate_red_pct": _pct(circuit.gate_count(), out.gate_count()),
            "seconds": time.time() - t,
        }
        if verify is not None and n <= EQUIV_QUBIT_CAP:
            r["equivalent"] = bool(verify.check(circuit, out).equivalent)
        res[name] = r
    return res


def study(
    circuits: list[IntermediateRepresentation],
    policies_by_seed: dict[int, Policies],
    verify=None,
    log=None,
) -> dict:
    """Run every method for every seed on every circuit.

    Returns ``{"circuits": [{name, qubits, gates, depth, bucket}], "runs": {seed: {name: {method: stats}}}}``.
    """
    meta = [{"name": c.name, "qubits": c.num_qubits, "gates": c.gate_count(), "depth": c.depth(), "bucket": bucket_of(c.num_qubits)} for c in circuits]
    runs: dict[int, dict] = {}
    for seed, pol in policies_by_seed.items():
        runs[seed] = {}
        for c in circuits:
            runs[seed][c.name] = run_methods(c, pol, seed, verify)
            if log:
                a = runs[seed][c.name]
                log(f"[seed {seed}] {c.name:22s} agent_cr depth-{a['agent_cr']['depth_red_pct']:.1f}% stub-{a['stub']['depth_red_pct']:.1f}%")
    return {"circuits": meta, "runs": runs}


def summarize(result: dict) -> dict:
    """Mean/std over seeds and circuits per (bucket, method) and per (circuit, method)."""
    seeds = sorted(result["runs"])
    methods = sorted({m for s in seeds for c in result["runs"][s].values() for m in c}, key=METHODS.index)
    per_circuit, per_bucket = {}, {}
    for c in result["circuits"]:
        per_circuit[c["name"]] = {
            m: {
                k: mean(result["runs"][s][c["name"]][m][k] for s in seeds)
                for k in ("depth_red_pct", "gate_red_pct")
            }
            for m in methods
            if all(m in result["runs"][s][c["name"]] for s in seeds)
        }
    for b, _, _ in BUCKETS:
        names = [c["name"] for c in result["circuits"] if c["bucket"] == b]
        if not names:
            continue
        per_bucket[b] = {"n_circuits": len(names)}
        for m in methods:
            if not all(m in result["runs"][s][n] for s in seeds for n in names):
                continue
            per_seed = [mean(result["runs"][s][n][m]["depth_red_pct"] for n in names) for s in seeds]
            gates = [mean(result["runs"][s][n][m]["gate_red_pct"] for n in names) for s in seeds]
            per_bucket[b][m] = {
                "depth_red_pct": mean(per_seed),
                "depth_red_std_over_seeds": pstdev(per_seed) if len(per_seed) > 1 else 0.0,
                "gate_red_pct": mean(gates),
            }
    return {"seeds": seeds, "per_bucket": per_bucket, "per_circuit": per_circuit}


def to_markdown(result: dict, summary: dict, train_info: str = "", title: str = "Zero-shot generalization: 5 -> 10 -> 20 qubits (issue #26)") -> str:
    labels = {"stub": "greedy `schedule()` stub", "cr_only": "CR only", "greedy": "greedy-depth + CR",
              "agent_cr": "**PPO + CR**", "agent_cr_x": "PPO + CR, scaled budget",
              "agent_nocr": "PPO, no CR (trained & run)", "agent_cr_off": "CR-trained PPO, CR off at test"}
    methods = [m for m in METHODS if any(m in v for v in summary["per_bucket"].values())]
    out = [f"# {title}", ""]
    if train_info:
        out += [train_info, ""]
    out += [f"Seeds: {summary['seeds']}. Values are mean % reduction vs the raw benchmark circuit "
            "(depth, then gate count); depth is mean ± std over seeds.", ""]
    out += ["## Depth reduction % by qubit bucket", "",
            "| Method | " + " | ".join(f"{b} qubits (n={summary['per_bucket'][b]['n_circuits']})" for b in summary["per_bucket"]) + " |",
            "|---|" + "---|" * len(summary["per_bucket"])]
    for m in methods:
        cells = []
        for b, v in summary["per_bucket"].items():
            x = v.get(m)
            cells.append(f"{x['depth_red_pct']:.1f} ± {x['depth_red_std_over_seeds']:.1f}" if x else "–")
        out.append(f"| {labels[m]} | " + " | ".join(cells) + " |")
    out += ["", "## Gate-count reduction %", "",
            "| Method | " + " | ".join(f"{b}" for b in summary["per_bucket"]) + " |",
            "|---|" + "---|" * len(summary["per_bucket"])]
    for m in methods:
        out.append(f"| {labels[m]} | " + " | ".join(f"{v[m]['gate_red_pct']:.1f}" if m in v else "–" for v in summary["per_bucket"].values()) + " |")
    out += ["", "## Per circuit (depth reduction %, mean over seeds)", "",
            "| Circuit | qubits | gates | depth | " + " | ".join(labels[m] for m in methods) + " |",
            "|---|---|---|---|" + "---|" * len(methods)]
    for c in result["circuits"]:
        pc = summary["per_circuit"][c["name"]]
        out.append(f"| {c['name']} | {c['qubits']} | {c['gates']} | {c['depth']} | " +
                   " | ".join(f"{pc[m]['depth_red_pct']:.1f}" if m in pc else "–" for m in methods) + " |")
    bad = [(s, n, m) for s, cs in result["runs"].items() for n, ms in cs.items() for m, r in ms.items() if r.get("equivalent") is False]
    checked = sum(1 for cs in result["runs"].values() for ms in cs.values() for r in ms.values() if "equivalent" in r)
    out += ["", f"Equivalence (numeric, <= {EQUIV_QUBIT_CAP} qubits): {checked} checks, {len(bad)} failures. "
            "Circuits above 8 qubits are not equivalence-checked here (numeric backend cap; the SMT path is the follow-up).", ""]
    return "\n".join(out)
