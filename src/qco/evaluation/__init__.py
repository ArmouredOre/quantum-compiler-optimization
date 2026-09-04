"""Stage 6 — evaluation & benchmarking engine."""

from qco.evaluation.calibration import Calibration, CALIBRATIONS, get_calibration, list_calibrations
from qco.evaluation.metrics import (
    MetricResult,
    gate_count_reduction,
    depth_reduction,
    execution_time,
    estimated_fidelity,
    runtime_complexity,
    scalarized_reward,
)
from qco.evaluation.benchmark_runner import BenchmarkReport, BenchmarkRow, run_benchmarks

__all__ = [
    "Calibration",
    "CALIBRATIONS",
    "get_calibration",
    "list_calibrations",
    "MetricResult",
    "gate_count_reduction",
    "depth_reduction",
    "execution_time",
    "estimated_fidelity",
    "runtime_complexity",
    "scalarized_reward",
    "BenchmarkReport",
    "BenchmarkRow",
    "run_benchmarks",
]
