"""Stage 6 — run the optimizer over the benchmark suite and tabulate results.

Phase 2: works against the local IR and any callable optimizer
``fn(ir) -> ir``, under a named device :mod:`~qco.evaluation.calibration`
profile. The Qiskit O0-O3 baseline comparison is added in Phase 5 once the
``compiler`` extra is installed (``_qiskit_baseline`` stub below) — the row
schema already carries ``baseline_*`` columns so downstream consumers
(``to_markdown`` / ``to_csv``) don't change shape when it lands.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from qco.evaluation.calibration import Calibration, get_calibration
from qco.evaluation.metrics import (
    depth_reduction,
    estimated_fidelity,
    execution_time,
    gate_count_reduction,
    runtime_complexity,
    scalarized_reward,
)
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "circuits")

CSV_FIELDS = [
    "circuit", "gate_count_before", "gate_count_after", "depth_before", "depth_after",
    "gate_reduction_pct", "depth_reduction_pct", "exec_time_ns", "fidelity_after",
    "optimizer_seconds", "runtime_complexity_s_per_gate", "reward",
    "baseline_name", "baseline_gate_count", "baseline_depth",
]


@dataclass(slots=True)
class BenchmarkRow:
    circuit: str
    gate_count_before: int
    gate_count_after: int
    depth_before: int
    depth_after: int
    gate_reduction_pct: float
    depth_reduction_pct: float
    exec_time_ns: float
    fidelity_after: float
    optimizer_seconds: float
    runtime_complexity_s_per_gate: float
    reward: float
    # Baseline columns — populated once the Qiskit O0-O3 harness lands in
    # Phase 5 (`_qiskit_baseline`). Left as None until then so this schema is
    # stable across Phase 2 -> Phase 5, per issue #8's "baseline column stubbed".
    baseline_name: str | None = None
    baseline_gate_count: int | None = None
    baseline_depth: int | None = None


@dataclass(slots=True)
class BenchmarkReport:
    rows: list[BenchmarkRow] = field(default_factory=list)
    calibration_name: str = "default"

    def mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.rows]
        return sum(vals) / len(vals) if vals else 0.0

    def to_markdown(self) -> str:
        head = (
            f"Calibration profile: `{self.calibration_name}`\n\n"
            "| circuit | gates → | depth → | gate red % | depth red % | exec (ns) | fidelity | "
            "opt (s) | opt (s/gate) | reward | baseline gates/depth |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )
        lines = [head]
        for r in self.rows:
            baseline = (
                f"{r.baseline_gate_count}/{r.baseline_depth} ({r.baseline_name})"
                if r.baseline_gate_count is not None
                else "—"
            )
            lines.append(
                f"| {r.circuit} | {r.gate_count_before}→{r.gate_count_after} | "
                f"{r.depth_before}→{r.depth_after} | {r.gate_reduction_pct:.1f} | "
                f"{r.depth_reduction_pct:.1f} | {r.exec_time_ns:.0f} | {r.fidelity_after:.4f} | "
                f"{r.optimizer_seconds:.4f} | {r.runtime_complexity_s_per_gate:.2e} | "
                f"{r.reward:+.4f} | {baseline} |"
            )
        lines.append(
            f"| **mean** | | | {self.mean('gate_reduction_pct'):.1f} | "
            f"{self.mean('depth_reduction_pct'):.1f} | {self.mean('exec_time_ns'):.0f} | "
            f"{self.mean('fidelity_after'):.4f} | {self.mean('optimizer_seconds'):.4f} | "
            f"{self.mean('runtime_complexity_s_per_gate'):.2e} | {self.mean('reward'):+.4f} | |"
        )
        return "\n".join(lines)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in self.rows:
            writer.writerow({k: getattr(r, k) for k in CSV_FIELDS})
        return buf.getvalue()


def _load_suite(directory: str | None) -> list[tuple[str, IntermediateRepresentation]]:
    directory = directory or BENCH_DIR
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.qasm"))):
        with open(path, "r", encoding="utf-8") as fh:
            out.append((os.path.basename(path), from_qasm2(fh.read())))
    return out


def run_benchmarks(
    optimizer: Callable[[IntermediateRepresentation], IntermediateRepresentation],
    directory: str | None = None,
    calibration: "str | Calibration" = "default",
    baseline: Callable[[IntermediateRepresentation], IntermediateRepresentation] | None = None,
    baseline_name: str = "qiskit-O2",
) -> BenchmarkReport:
    """Run ``optimizer`` over every circuit in ``directory`` (default: the
    committed suite) and tabulate the five evaluation-parameter metrics.

    ``optimizer``: callable ``IntermediateRepresentation -> IntermediateRepresentation``.
    ``calibration``: a name from ``qco.evaluation.calibration.CALIBRATIONS`` or a
    ``Calibration`` instance — controls ``execution_time`` / ``estimated_fidelity``.
    ``baseline``: optional second optimizer (e.g. the Phase 5 Qiskit O0-O3 harness)
    run on the same circuits to fill in the ``baseline_*`` columns.
    """
    cal = get_calibration(calibration)
    report = BenchmarkReport(calibration_name=cal.name)
    for name, ir in _load_suite(directory):
        t0 = time.perf_counter()
        opt = optimizer(ir.copy())
        dt = time.perf_counter() - t0

        baseline_gate_count = baseline_depth = None
        if baseline is not None:
            base_ir = baseline(ir.copy())
            baseline_gate_count = base_ir.gate_count()
            baseline_depth = base_ir.depth()

        report.rows.append(
            BenchmarkRow(
                circuit=name,
                gate_count_before=ir.gate_count(),
                gate_count_after=opt.gate_count(),
                depth_before=ir.depth(),
                depth_after=opt.depth(),
                gate_reduction_pct=gate_count_reduction(ir, opt).value,
                depth_reduction_pct=depth_reduction(ir, opt).value,
                exec_time_ns=execution_time(opt, cal).value,
                fidelity_after=estimated_fidelity(opt, cal).value,
                optimizer_seconds=dt,
                runtime_complexity_s_per_gate=runtime_complexity(dt, opt.gate_count()).value,
                reward=scalarized_reward(ir, opt, calibration=cal),
                baseline_name=baseline_name if baseline is not None else None,
                baseline_gate_count=baseline_gate_count,
                baseline_depth=baseline_depth,
            )
        )
    return report


def _qiskit_baseline(ir, level):  # pragma: no cover - Phase 5
    raise NotImplementedError("Qiskit O0-O3 baseline comparison lands in Phase 5")
