"""Stage 6 — run the optimizer over the benchmark suite and tabulate results.

Phase 2: works against the local IR and any callable optimizer
``fn(ir) -> ir``. The Qiskit O0-O3 baseline comparison is added in Phase 5 once
the ``compiler`` extra is installed (``_qiskit_baseline`` stub below).
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field

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
    reward: float


@dataclass(slots=True)
class BenchmarkReport:
    rows: list[BenchmarkRow] = field(default_factory=list)

    def mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.rows]
        return sum(vals) / len(vals) if vals else 0.0

    def to_markdown(self) -> str:
        head = (
            "| circuit | gates → | depth → | gate red % | depth red % | exec (ns) | fidelity | opt (s) | reward |\n"
            "|---|---|---|---|---|---|---|---|---|"
        )
        lines = [head]
        for r in self.rows:
            lines.append(
                f"| {r.circuit} | {r.gate_count_before}→{r.gate_count_after} | "
                f"{r.depth_before}→{r.depth_after} | {r.gate_reduction_pct:.1f} | "
                f"{r.depth_reduction_pct:.1f} | {r.exec_time_ns:.0f} | {r.fidelity_after:.4f} | "
                f"{r.optimizer_seconds:.4f} | {r.reward:+.4f} |"
            )
        lines.append(
            f"| **mean** | | | {self.mean('gate_reduction_pct'):.1f} | "
            f"{self.mean('depth_reduction_pct'):.1f} | {self.mean('exec_time_ns'):.0f} | "
            f"{self.mean('fidelity_after'):.4f} | {self.mean('optimizer_seconds'):.4f} | "
            f"{self.mean('reward'):+.4f} |"
        )
        return "\n".join(lines)


def _load_suite(directory: str | None) -> list[tuple[str, IntermediateRepresentation]]:
    directory = directory or BENCH_DIR
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.qasm"))):
        with open(path, "r", encoding="utf-8") as fh:
            out.append((os.path.basename(path), from_qasm2(fh.read())))
    return out


def run_benchmarks(optimizer, directory: str | None = None) -> BenchmarkReport:
    """``optimizer``: callable ``IntermediateRepresentation -> IntermediateRepresentation``."""
    report = BenchmarkReport()
    for name, ir in _load_suite(directory):
        t0 = time.perf_counter()
        opt = optimizer(ir.copy())
        dt = time.perf_counter() - t0
        report.rows.append(
            BenchmarkRow(
                circuit=name,
                gate_count_before=ir.gate_count(),
                gate_count_after=opt.gate_count(),
                depth_before=ir.depth(),
                depth_after=opt.depth(),
                gate_reduction_pct=gate_count_reduction(ir, opt).value,
                depth_reduction_pct=depth_reduction(ir, opt).value,
                exec_time_ns=execution_time(opt).value,
                fidelity_after=estimated_fidelity(opt).value,
                optimizer_seconds=dt,
                reward=scalarized_reward(ir, opt),
            )
        )
    return report


def _qiskit_baseline(ir, level):  # pragma: no cover - Phase 5
    raise NotImplementedError("Qiskit O0-O3 baseline comparison lands in Phase 5")
