"""Stage 6 — run optimizers over the benchmark suite and tabulate results.

``run_benchmarks`` takes any optimizer ``fn(ir) -> ir`` (or ``-> (ir,
solver_seconds)``) under a named device :mod:`~qco.evaluation.calibration`
profile. Baselines live in :mod:`qco.evaluation.baselines`.

One command builds every table (#23, #32)::

    python -m qco.evaluation.benchmark_runner --out results

prints the five-metric tables for Qiskit O0-O3 and writes
``results/summary.md`` (qco vs Qiskit / TKET / PyZX / BQSKit, ablations
minus A/B/C/D, and N1 window-selection: gnn / rule / random / exhaustive) plus
per-optimizer markdown + CSV.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import sys
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
from qco.evaluation.baselines import Unsupported, available_baselines, cx_cost
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "circuits")

CSV_FIELDS = [
    "circuit", "gate_count_before", "gate_count_after", "depth_before", "depth_after",
    "gate_reduction_pct", "depth_reduction_pct", "exec_time_ns", "fidelity_after",
    "optimizer_seconds", "runtime_complexity_s_per_gate", "reward",
    "baseline_name", "baseline_gate_count", "baseline_depth",
    "cx_cost_before", "cx_cost_after", "solver_seconds",
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
    cx_cost_before: int = 0          # multi-qubit cost in CX-equivalents (basis-independent)
    cx_cost_after: int = 0
    solver_seconds: float | None = None  # exact re-synthesis time, when the optimizer reports it

    @property
    def cx_cost_reduction_pct(self) -> float:
        b = self.cx_cost_before
        return 100.0 * (b - self.cx_cost_after) / b if b else 0.0


@dataclass(slots=True)
class BenchmarkReport:
    rows: list[BenchmarkRow] = field(default_factory=list)
    calibration_name: str = "default"
    skipped: list[str] = field(default_factory=list)  # circuits the optimizer can't handle

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


def _load_suite(directory: str | None, max_qubits: int | None = None) -> list[tuple[str, IntermediateRepresentation]]:
    directory = directory or BENCH_DIR
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.qasm"))):
        with open(path, "r", encoding="utf-8") as fh:
            ir = from_qasm2(fh.read())
        ir.name = os.path.basename(path)[: -len(".qasm")]
        if max_qubits is None or ir.num_qubits <= max_qubits:
            out.append((os.path.basename(path), ir))
    return out


def run_benchmarks(
    optimizer: Callable[[IntermediateRepresentation], IntermediateRepresentation],
    directory: str | None = None,
    calibration: "str | Calibration" = "default",
    baseline: Callable[[IntermediateRepresentation], IntermediateRepresentation] | None = None,
    baseline_name: str = "qiskit-O2",
    max_qubits: int | None = None,
) -> BenchmarkReport:
    """Run ``optimizer`` over every circuit in ``directory`` (default: the
    committed suite) and tabulate the five evaluation-parameter metrics.

    ``optimizer``: callable ``IntermediateRepresentation -> IntermediateRepresentation``.
    ``calibration``: a name from ``qco.evaluation.calibration.CALIBRATIONS`` or a
    ``Calibration`` instance — controls ``execution_time`` / ``estimated_fidelity``.
    ``optimizer`` may also return ``(ir, solver_seconds)``; raising
    :class:`~qco.evaluation.baselines.Unsupported` skips that circuit.
    ``baseline``: optional second optimizer (e.g. ``qiskit_baseline(2)``)
    run on the same circuits to fill in the ``baseline_*`` columns.
    """
    cal = get_calibration(calibration)
    report = BenchmarkReport(calibration_name=cal.name)
    for name, ir in _load_suite(directory, max_qubits):
        t0 = time.perf_counter()
        try:
            opt = optimizer(ir.copy())
        except Unsupported:
            report.skipped.append(name)
            continue
        dt = time.perf_counter() - t0
        solver_s = None
        if isinstance(opt, tuple):
            opt, solver_s = opt

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
                cx_cost_before=cx_cost(ir),
                cx_cost_after=cx_cost(opt),
                solver_seconds=solver_s,
            )
        )
    return report


def summary_markdown(reports: dict[str, BenchmarkReport], title: str) -> str:
    """One row per optimizer: suite means of the five metrics + CX-cost + solver time."""
    lines = [
        f"### {title}",
        "",
        "| optimizer | circuits | gate red % | depth red % | CX-cost red % | exec (ns) | "
        "fidelity | opt (s) | solver (s) | skipped |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        n = len(r.rows)
        cx = sum(row.cx_cost_reduction_pct for row in r.rows) / n if n else 0.0
        solver = [row.solver_seconds for row in r.rows if row.solver_seconds is not None]
        solver_cell = f"{sum(solver):.1f}" if solver else "—"
        lines.append(
            f"| {name} | {n} | {r.mean('gate_reduction_pct'):.1f} | {r.mean('depth_reduction_pct'):.1f} | "
            f"{cx:.1f} | {r.mean('exec_time_ns'):.0f} | {r.mean('fidelity_after'):.4f} | "
            f"{r.mean('optimizer_seconds'):.2f} | {solver_cell} | {len(r.skipped)} |"
        )
    return "\n".join(lines)


def per_circuit_markdown(reports: dict[str, BenchmarkReport]) -> str:
    """Circuit x optimizer matrix of ``gates / CX-cost`` after optimization."""
    names = list(reports)
    circuits = sorted({row.circuit for r in reports.values() for row in r.rows})
    cell = {(n, row.circuit): f"{row.gate_count_after} / {row.cx_cost_after}"
            for n, r in reports.items() for row in r.rows}
    before = {row.circuit: f"{row.gate_count_before} / {row.cx_cost_before}"
              for r in reports.values() for row in r.rows}
    lines = [
        "| circuit | input | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for c in circuits:
        lines.append(f"| {c[:-5]} | {before[c]} | " + " | ".join(cell.get((n, c), "n/a") for n in names) + " |")
    return "\n".join(lines)


def pipeline_optimizer(**config) -> Callable[[IntermediateRepresentation], tuple]:
    """qco's ``HybridOptimizer`` as a runner optimizer: best-gate-count front member."""
    from qco.pipeline import PipelineConfig, optimize

    def run(ir: IntermediateRepresentation):
        res = optimize(ir, PipelineConfig(**config))
        best = min(res.pareto_front, key=lambda p: (p.objective.gate_count, p.objective.depth))
        return best.circuit, res.solver_time_s

    return run


ABLATIONS = {
    "full": {},
    "minus A (RL)": {"use_rl": False},
    "minus B (GNN)": {"use_gnn": False},
    "minus C (SMT)": {"enable_resynthesis": False, "verify_rewrites": False},
    "minus D (NSGA-II)": {"use_ea": False},
}
WINDOW_MODES = ("gnn", "rule", "random", "exhaustive")


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description="qco Stage 6: baselines, ablations, window selection")
    ap.add_argument("--suite", default=None, help="directory of .qasm files (default: committed suite)")
    ap.add_argument("--out", default="results", help="directory for summary.md + per-optimizer md/csv")
    ap.add_argument("--calibration", default="default")
    ap.add_argument("--max-qubits", type=int, default=None, help="skip wider circuits")
    ap.add_argument("--budget", type=float, default=2.0, help="qco solver budget per circuit (s)")
    ap.add_argument("--window-max-qubits", type=int, default=6,
                    help="window-selection runs only on circuits this small ('exhaustive' has no budget)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["baselines", "ablations", "windows"])
    ap.add_argument("--gnn-checkpoint", default="auto",
                type=lambda s: None if s.lower() == "none" else s,
                help="'none' = rule-based fallback, 'auto' = checkpoints/module_b.pt if present, or a path")
    args = ap.parse_args(argv)

    def bench(opt, label, max_qubits=args.max_qubits):
        def logged(ir):  # progress to stderr: full runs take tens of minutes
            t0 = time.perf_counter()
            try:
                return opt(ir)
            finally:
                print(f"  [{label}] {ir.name} n={ir.num_qubits} {time.perf_counter() - t0:.1f}s",
                      file=sys.stderr, flush=True)

        return run_benchmarks(logged, args.suite, args.calibration, max_qubits=max_qubits)

    def qco(**cfg):
        return pipeline_optimizer(solver_budget_s=args.budget, calibration=args.calibration,
                                    gnn_checkpoint=args.gnn_checkpoint, **cfg)
    os.makedirs(args.out, exist_ok=True)
    sections = [f"# qco evaluation\n\nCalibration `{args.calibration}`, solver budget {args.budget}s/circuit. "
                "CX-cost = multi-qubit gates in CX-equivalents (basis-independent; tools emit different gate sets)."]
    all_reports: dict[str, BenchmarkReport] = {}

    full = bench(qco(), "qco")
    all_reports["qco"] = full
    if "baselines" not in args.skip:
        compared = {"qco": full}
        for name, opt in available_baselines().items():
            compared[name] = all_reports[name] = bench(opt, name)
            if name.startswith("qiskit-O"):
                print(f"## {name}\n\n{compared[name].to_markdown()}\n", flush=True)
        sections += [summary_markdown(compared, "qco vs baselines"), per_circuit_markdown(compared)]
    if "ablations" not in args.skip:
        ablated = {"full": full}
        for name, cfg in ABLATIONS.items():
            if cfg:
                ablated[name] = all_reports[name] = bench(qco(**cfg), name)
        sections.append(summary_markdown(ablated, "Ablations (full pipeline minus one module)"))
    if "windows" not in args.skip:
        windows = {mode: bench(qco(window_ranking=mode), f"windows-{mode}", args.window_max_qubits)
                   for mode in WINDOW_MODES}
        all_reports.update({f"windows-{m}": r for m, r in windows.items()})
        sections.append(summary_markdown(
            windows, f"N1 window selection (circuits <= {args.window_max_qubits} qubits)"))

    for name, r in all_reports.items():
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        with open(os.path.join(args.out, f"{slug}.md"), "w", encoding="utf-8") as fh:
            fh.write(r.to_markdown() + "\n")
        with open(os.path.join(args.out, f"{slug}.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write(r.to_csv())
    summary = "\n\n".join(sections) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
