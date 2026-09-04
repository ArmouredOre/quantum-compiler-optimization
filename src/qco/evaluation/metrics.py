"""Stage 6 — the five evaluation parameters (docs/evaluation_parameters.md).

All fully implemented in Phase 2: they operate on the IR and a named device
:mod:`~qco.evaluation.calibration` profile, so the benchmark harness works before
any module is trained. Each function's docstring names the exact
`docs/evaluation_parameters.md` row it implements; real per-backend calibration
data (via Qiskit's ``BackendProperties``) replaces the illustrative profiles in
Phase 5, without changing these signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

from qco.evaluation.calibration import Calibration, get_calibration
from qco.ir.intermediate_representation import IntermediateRepresentation

# Re-exported for backward compatibility with Phase 2 call sites.
DEFAULT_GATE_TIME_NS = get_calibration("default").gate_time_ns
DEFAULT_ERROR_RATE = get_calibration("default").error_rate


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float
    unit: str


def _pct(before: float, after: float) -> float:
    return 0.0 if before == 0 else (before - after) / before * 100.0


def gate_count_reduction(before: IntermediateRepresentation, after: IntermediateRepresentation) -> MetricResult:
    """docs/evaluation_parameters.md — "Gate Count Reduction (%)".

    ``(Gates_before - Gates_after) / Gates_before * 100``.
    """
    return MetricResult("gate_count_reduction", _pct(before.gate_count(), after.gate_count()), "%")


def depth_reduction(before: IntermediateRepresentation, after: IntermediateRepresentation) -> MetricResult:
    """docs/evaluation_parameters.md — "Circuit Depth Reduction (%)".

    ``(Depth_before - Depth_after) / Depth_before * 100``.
    """
    return MetricResult("depth_reduction", _pct(before.depth(), after.depth()), "%")


def execution_time(
    circuit: IntermediateRepresentation,
    calibration: "str | Calibration" = "default",
) -> MetricResult:
    """docs/evaluation_parameters.md — "Execution Time (ns or µs)".

    Sum of per-gate durations along the critical path (ASAP layering), using the
    named device ``calibration`` profile (see
    ``qco.evaluation.calibration.CALIBRATIONS``).
    """
    cal = get_calibration(calibration)
    frontier = [0.0] * circuit.num_qubits
    for g in circuit.gates:
        dur = cal.duration(len(g.qubits))
        start = max(frontier[q] for q in g.qubits)
        for q in g.qubits:
            frontier[q] = start + dur
    return MetricResult("execution_time", max(frontier, default=0.0), "ns")


def estimated_fidelity(
    circuit: IntermediateRepresentation,
    calibration: "str | Calibration" = "default",
) -> MetricResult:
    """docs/evaluation_parameters.md — "Estimated Fidelity / Error Reduction".

    Product of per-gate success probabilities (simple depolarizing proxy) under
    the named device ``calibration`` profile. "Error Reduction" itself is
    ``estimated_fidelity(after) - estimated_fidelity(before)``, computed by
    ``scalarized_reward`` below; a real noise-model simulator estimate (Qiskit
    Aer) replaces this proxy in Phase 5.
    """
    cal = get_calibration(calibration)
    f = 1.0
    for g in circuit.gates:
        f *= 1.0 - cal.error(len(g.qubits))
    return MetricResult("estimated_fidelity", f, "")


def runtime_complexity(seconds: float, circuit_size: int) -> MetricResult:
    """docs/evaluation_parameters.md — "Runtime Complexity".

    Empirical wall-clock cost of the optimizer itself, normalized per gate
    (theoretical Big-O is characterized separately per module in the report).
    """
    per_gate = seconds / circuit_size if circuit_size else 0.0
    return MetricResult("runtime_complexity", per_gate, "s/gate")


def scalarized_reward(
    before: IntermediateRepresentation,
    after: IntermediateRepresentation,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    calibration: "str | Calibration" = "default",
) -> float:
    """Single scalar fed back to Module A as the RL reward (closed loop).

    ``weights`` = (gate-count-reduction weight, depth-reduction weight,
    fidelity-delta weight); combines three of the five
    `docs/evaluation_parameters.md` metrics. Positive when ``after`` is better
    than ``before``.
    """
    wg, wd, wf = weights
    g = gate_count_reduction(before, after).value / 100.0
    d = depth_reduction(before, after).value / 100.0
    f = estimated_fidelity(after, calibration).value - estimated_fidelity(before, calibration).value
    return wg * g + wd * d + wf * f
