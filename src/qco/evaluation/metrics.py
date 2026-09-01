"""Stage 6 — the five evaluation parameters (docs/evaluation_parameters.md).

All fully implemented in Phase 2: they operate on the IR and a small gate-time /
error-rate table, so the benchmark harness works before any module is trained.
Device-calibrated tables replace the defaults in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass

from qco.ir.intermediate_representation import IntermediateRepresentation

# Default gate durations (ns) and error rates — coarse superconducting-style
# numbers; overridden with backend calibration data in Phase 5.
DEFAULT_GATE_TIME_NS = {"1q": 35.0, "2q": 300.0, "3q": 600.0}
DEFAULT_ERROR_RATE = {"1q": 1e-3, "2q": 1e-2, "3q": 3e-2}


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float
    unit: str


def _pct(before: float, after: float) -> float:
    return 0.0 if before == 0 else (before - after) / before * 100.0


def gate_count_reduction(before: IntermediateRepresentation, after: IntermediateRepresentation) -> MetricResult:
    return MetricResult("gate_count_reduction", _pct(before.gate_count(), after.gate_count()), "%")


def depth_reduction(before: IntermediateRepresentation, after: IntermediateRepresentation) -> MetricResult:
    return MetricResult("depth_reduction", _pct(before.depth(), after.depth()), "%")


def execution_time(circuit: IntermediateRepresentation, gate_time_ns: dict[str, float] | None = None) -> MetricResult:
    """Sum of per-gate durations along the critical path (ASAP layering)."""
    t = gate_time_ns or DEFAULT_GATE_TIME_NS
    frontier = [0.0] * circuit.num_qubits
    for g in circuit.gates:
        key = f"{len(g.qubits)}q"
        dur = t.get(key, t["2q"])
        start = max(frontier[q] for q in g.qubits)
        for q in g.qubits:
            frontier[q] = start + dur
    return MetricResult("execution_time", max(frontier, default=0.0), "ns")


def estimated_fidelity(circuit: IntermediateRepresentation, error_rate: dict[str, float] | None = None) -> MetricResult:
    """Product of per-gate success probabilities (simple depolarizing proxy)."""
    e = error_rate or DEFAULT_ERROR_RATE
    f = 1.0
    for g in circuit.gates:
        f *= 1.0 - e.get(f"{len(g.qubits)}q", e["2q"])
    return MetricResult("estimated_fidelity", f, "")


def runtime_complexity(seconds: float, circuit_size: int) -> MetricResult:
    """Empirical optimizer cost, normalized per gate (Big-O is characterized
    separately per module in the report)."""
    per_gate = seconds / circuit_size if circuit_size else 0.0
    return MetricResult("runtime_complexity", per_gate, "s/gate")


def scalarized_reward(
    before: IntermediateRepresentation,
    after: IntermediateRepresentation,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> float:
    """Single scalar fed back to Module A as the RL reward (closed loop).

    Positive when ``after`` is better than ``before``.
    """
    wg, wd, wf = weights
    g = gate_count_reduction(before, after).value / 100.0
    d = depth_reduction(before, after).value / 100.0
    f = estimated_fidelity(after).value - estimated_fidelity(before).value
    return wg * g + wd * d + wf * f
