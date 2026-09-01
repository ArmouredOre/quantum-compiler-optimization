"""Stage 6 — evaluation & benchmarking engine."""

from qco.evaluation.metrics import (
    MetricResult,
    gate_count_reduction,
    depth_reduction,
    execution_time,
    estimated_fidelity,
    scalarized_reward,
)
from qco.evaluation.benchmark_runner import BenchmarkReport, run_benchmarks

__all__ = [
    "MetricResult",
    "gate_count_reduction",
    "depth_reduction",
    "execution_time",
    "estimated_fidelity",
    "scalarized_reward",
    "BenchmarkReport",
    "run_benchmarks",
]
