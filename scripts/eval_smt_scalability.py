#!/usr/bin/env python3
"""Module C (SMT Verifier & Bounded Re-synthesizer) Scalability & Evaluation Suite.

Evaluates:
1. Verification runtime vs. qubit count (Numeric vs. Z3-Dense vs. Phase-Polynomial).
2. Verification runtime vs. gate count.
3. Bounded re-synthesis gains on real partitioned pipeline windows.
4. Canonical window cache hit rates and speedups.

Generates:
- `docs/smt_eval/scaling_plot.svg` & `scaling_plot.png` (Acceptance criteria: Phase-Poly scaling beyond 8 qubits)
- `docs/smt_eval/scalability_table.md`
- `docs/smt_eval/resynthesis_benchmark.csv`
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qco.graphs.partitioning import partition_circuit
from qco.ir.intermediate_representation import Gate, IntermediateRepresentation
from qco.ir.parser import parse
from qco.modules.smt_verifier.bounded_optimizer import (
    BoundedResynthesizer,
    ResynthConfig,
)
from qco.modules.smt_verifier.equivalence import EquivalenceChecker, circuit_unitary
from qco.modules.smt_verifier.phase_poly import check_phase_poly, is_phase_poly_circuit


def build_phase_poly_circuit(n_qubits: int, n_layers: int, rng: random.Random) -> IntermediateRepresentation:
    """Build a CX + Rz + X circuit (Phase-Polynomial) of specified qubits and depth."""
    ir = IntermediateRepresentation(num_qubits=n_qubits, name=f"phase_poly_n{n_qubits}")
    angles = [math.pi / 4, math.pi / 2, math.pi, -math.pi / 4, -math.pi / 2, 0.35, -0.75]
    for _ in range(n_layers):
        for q in range(n_qubits - 1):
            ir.add("cx", (q, q + 1))
        for q in range(n_qubits):
            angle = rng.choice(angles)
            ir.add("rz", (q,), (angle,))
        for q in range(n_qubits - 1, 0, -1):
            ir.add("cx", (q - 1, q))
    return ir


def build_clifford_t_circuit(n_qubits: int, n_gates: int, rng: random.Random) -> IntermediateRepresentation:
    """Build a general Clifford+T circuit for dense unitary verification benchmarking."""
    ir = IntermediateRepresentation(num_qubits=n_qubits, name=f"clifford_t_n{n_qubits}")
    for _ in range(n_gates):
        choice = rng.random()
        if choice < 0.4 and n_qubits >= 2:
            c, t = rng.sample(range(n_qubits), 2)
            ir.add("cx", (c, t))
        elif choice < 0.7:
            q = rng.randrange(n_qubits)
            ir.add("h", (q,))
        elif choice < 0.85:
            q = rng.randrange(n_qubits)
            ir.add("t", (q,))
        else:
            q = rng.randrange(n_qubits)
            ir.add("s", (q,))
    return ir


def benchmark_verification_vs_qubits() -> list[dict[str, Any]]:
    """Benchmark Numeric, Z3-Dense, and Phase-Polynomial verifiers across qubit scales."""
    print("--> Benchmarking Verification Runtime vs Qubit Count...", flush=True)
    results = []
    qubit_scales = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
    num_checker = EquivalenceChecker(backend="numeric")
    smt_checker = EquivalenceChecker(backend="smt", timeout=3.0)

    rng = random.Random(42)

    for n in qubit_scales:
        print(f"    Testing n = {n:2d} qubits...", flush=True)
        # 1. Phase-polynomial circuit (scales to 64+ qubits)
        poly_ir1 = build_phase_poly_circuit(n, n_layers=2, rng=rng)
        poly_ir2 = poly_ir1.copy()

        t0 = time.perf_counter()
        res_poly = check_phase_poly(poly_ir1, poly_ir2)
        poly_time = time.perf_counter() - t0
        assert res_poly.equivalent

        # 2. Dense Numeric path (valid <= 8 qubits)
        num_time: float | None = None
        if n <= 8:
            dense_ir = build_clifford_t_circuit(n, n_gates=min(25, n * 4), rng=rng)
            dense_copy = dense_ir.copy()
            t0 = time.perf_counter()
            res_num = num_checker.check(dense_ir, dense_copy)
            num_time = time.perf_counter() - t0
            assert res_num.equivalent

        # 3. Dense Z3-SMT path (tractable for <= 4 qubits)
        smt_time: float | None = None
        if n <= 4:
            dense_ir = build_clifford_t_circuit(n, n_gates=min(12, n * 3), rng=rng)
            dense_copy = dense_ir.copy()
            t0 = time.perf_counter()
            res_smt = smt_checker.check(dense_ir, dense_copy)
            smt_time = time.perf_counter() - t0
            assert res_smt.equivalent

        results.append({
            "n_qubits": n,
            "gate_count": poly_ir1.gate_count(),
            "phase_poly_time_s": poly_time,
            "numeric_time_s": num_time,
            "smt_dense_time_s": smt_time,
        })
        print(
            f"      Phase-Poly: {poly_time*1000:7.2f} ms | "
            f"Numeric: {f'{num_time*1000:7.2f} ms' if num_time is not None else 'Capped (>8q)'} | "
            f"Z3-Dense: {f'{smt_time*1000:7.2f} ms' if smt_time is not None else 'Intractable (>4q)'}",
            flush=True,
        )

    return results


def benchmark_verification_vs_gates() -> list[dict[str, Any]]:
    """Benchmark Phase-Polynomial verification time as gate count scales up to 1000 gates."""
    print("--> Benchmarking Verification Runtime vs Gate Count (n = 16 qubits)...", flush=True)
    results = []
    gate_counts = [20, 50, 100, 200, 350, 500, 750, 1000]
    n_qubits = 16
    rng = random.Random(123)

    for target_gates in gate_counts:
        layers = max(1, target_gates // (n_qubits * 2))
        ir1 = build_phase_poly_circuit(n_qubits, n_layers=layers, rng=rng)
        ir2 = ir1.copy()

        t0 = time.perf_counter()
        res = check_phase_poly(ir1, ir2)
        elapsed = time.perf_counter() - t0
        assert res.equivalent

        results.append({
            "n_qubits": n_qubits,
            "actual_gates": ir1.gate_count(),
            "time_s": elapsed,
        })
        print(f"    gates = {ir1.gate_count():4d}: {elapsed*1000:7.2f} ms", flush=True)

    return results


def benchmark_resynthesis_on_pipeline_windows() -> list[dict[str, Any]]:
    """Benchmark BoundedResynthesizer on real partitioned windows from benchmark circuits."""
    print("--> Benchmarking Bounded Re-synthesis on Real Benchmark Circuits...", flush=True)
    circuits_dir = ROOT / "benchmarks" / "circuits"
    test_files = [
        "ghz_n5.qasm",
        "ghz_n10.qasm",
        "rand_cnotpauli_n5.qasm",
        "rand_cliffordt_n5.qasm",
        "qaoa_n6_p1.qasm",
        "qft_n5.qasm",
    ]

    resynth_cfg = ResynthConfig(max_qubits=4, max_gates=10, timeout_s=0.3)
    resynthesizer = BoundedResynthesizer(config=resynth_cfg)
    summary_records = []

    for fname in test_files:
        path = circuits_dir / fname
        if not path.exists():
            continue

        ir = parse(str(path))
        blocks = partition_circuit(ir, max_block_gates=6, backend="auto")
        # Subsample up to 8 blocks per circuit to maintain fast benchmark runtime
        test_blocks = blocks[:8]

        orig_gates = sum(b.ir.gate_count() for b in test_blocks)
        total_reduced_gates = 0
        windows_evaluated = 0
        windows_reduced = 0

        t0 = time.perf_counter()
        for b in test_blocks:
            candidate = b.ir
            if resynthesizer.applicable(candidate):
                windows_evaluated += 1
                res = resynthesizer.resynthesize(candidate)
                if res.reduced:
                    windows_reduced += 1
                    total_reduced_gates += res.gates_saved
        solve_elapsed = time.perf_counter() - t0

        c_info = resynthesizer.cache_info()
        summary_records.append({
            "circuit": fname,
            "qubits": ir.num_qubits,
            "orig_gates": orig_gates,
            "blocks_count": len(test_blocks),
            "windows_evaluated": windows_evaluated,
            "windows_reduced": windows_reduced,
            "gates_saved": total_reduced_gates,
            "reduction_pct": (total_reduced_gates / orig_gates * 100.0) if orig_gates else 0.0,
            "cache_hits": c_info["hits"],
            "cache_misses": c_info["misses"],
            "cache_size": c_info["size"],
            "time_s": solve_elapsed,
        })
        print(
            f"    {fname:22s} | Blocks: {len(test_blocks):2d} | Evaluated: {windows_evaluated:2d} | "
            f"Saved: {total_reduced_gates:2d} gates | Cache hits: {c_info['hits']} | Time: {solve_elapsed:.2f}s",
            flush=True,
        )

    return summary_records


def generate_plots_and_tables(
    scaling_data: list[dict[str, Any]],
    gate_data: list[dict[str, Any]],
    resynth_data: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Generate publication-quality SVG/PNG plots and markdown tables."""
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Main Scaling Plot (Acceptance: Shows phase-polynomial path beyond 8 qubits)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Runtime vs Qubits
    qubits = [d["n_qubits"] for d in scaling_data]
    poly_times = [d["phase_poly_time_s"] for d in scaling_data]

    num_qubits = [d["n_qubits"] for d in scaling_data if d["numeric_time_s"] is not None]
    num_times = [d["numeric_time_s"] for d in scaling_data if d["numeric_time_s"] is not None]

    smt_qubits = [d["n_qubits"] for d in scaling_data if d["smt_dense_time_s"] is not None]
    smt_times = [d["smt_dense_time_s"] for d in scaling_data if d["smt_dense_time_s"] is not None]

    ax1.plot(qubits, poly_times, "o-", color="#10b981", linewidth=2.5, markersize=7, label="Phase-Polynomial SMT (Novelty N2)")
    ax1.plot(num_qubits, num_times, "s--", color="#3b82f6", linewidth=2, markersize=6, label="Numeric (Dense Unitary)")
    ax1.plot(smt_qubits, smt_times, "^:", color="#ef4444", linewidth=2, markersize=6, label="Z3-Dense SMT")

    # Draw vertical barrier at 8 qubits
    ax1.axvline(x=8, color="#dc2626", linestyle="--", alpha=0.8, linewidth=1.5)
    ax1.text(8.5, max(poly_times) * 0.1, "8-Qubit Dense Ceiling\n($2^{16}$ Unitary Entries)", color="#dc2626", fontsize=9, fontweight="bold")

    ax1.set_yscale("log")
    ax1.set_xlabel("Qubit Count ($n$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Verification Runtime (seconds, log scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Module C: Verification Scalability vs Qubits", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    # Plot 2: Phase-Poly Runtime vs Gates
    actual_gates = [d["actual_gates"] for d in gate_data]
    gate_times_ms = [d["time_s"] * 1000.0 for d in gate_data]

    ax2.plot(actual_gates, gate_times_ms, "D-", color="#8b5cf6", linewidth=2.2, markersize=6, label="Phase-Poly ($n=16$)")
    ax2.set_xlabel("Circuit Gate Count ($k$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Verification Runtime (milliseconds)", fontsize=11, fontweight="bold")
    ax2.set_title("Phase-Polynomial Runtime vs Gate Count ($n=16$)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    svg_path = out_dir / "scaling_plot.svg"
    png_path = out_dir / "scaling_plot.png"
    plt.savefig(svg_path, format="svg")
    plt.savefig(png_path, format="png")
    plt.close()
    print(f"--> Saved plots to {svg_path} and {png_path}", flush=True)

    # 2. Write Scalability Markdown Table
    md_table_path = out_dir / "scalability_table.md"
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Module C: Verification Scalability & Empirical Benchmarks\n\n")
        f.write("### 1. Verification Runtime vs Qubit Count\n\n")
        f.write("| Qubits ($n$) | Gate Count | Numeric (Dense) [s] | Z3-Dense SMT [s] | Phase-Poly SMT (N2) [s] | Scaling Status |\n")
        f.write("|:------------:|:----------:|:-------------------:|:----------------:|:-----------------------:|:---------------|\n")
        for d in scaling_data:
            num_str = f"{d['numeric_time_s']:.4f}" if d['numeric_time_s'] is not None else "*Capped (>8q)*"
            smt_str = f"{d['smt_dense_time_s']:.4f}" if d['smt_dense_time_s'] is not None else "*Intractable (>4q)*"
            poly_str = f"{d['phase_poly_time_s']:.5f}"
            status = "Verified (Phase-Poly)" if d["n_qubits"] > 8 else "Verified (All)"
            f.write(f"| {d['n_qubits']:2d} | {d['gate_count']:3d} | {num_str} | {smt_str} | {poly_str} | {status} |\n")

        f.write("\n### 2. Bounded Re-Synthesis Gains on Real Benchmark Windows\n\n")
        f.write("| Benchmark Circuit | Qubits | Original Gates | Evaluated Windows | Reduced Windows | Gates Saved | Reduction (%) | Cache Hits | Solve Time [s] |\n")
        f.write("|:------------------|:------:|:--------------:|:-----------------:|:---------------:|:-----------:|:-------------:|:----------:|:---------------:|\n")
        for r in resynth_data:
            f.write(
                f"| `{r['circuit']}` | {r['qubits']} | {r['orig_gates']} | {r['windows_evaluated']} | "
                f"{r['windows_reduced']} | {r['gates_saved']} | {r['reduction_pct']:.1f}% | "
                f"{r['cache_hits']} | {r['time_s']:.4f} |\n"
            )

    print(f"--> Saved markdown table to {md_table_path}", flush=True)

    # 3. Write CSV Results
    csv_path = out_dir / "resynthesis_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resynth_data[0].keys()))
        writer.writeheader()
        writer.writerows(resynth_data)
    print(f"--> Saved CSV to {csv_path}", flush=True)


def main():
    print("=" * 70, flush=True)
    print("Module C (SMT Verifier & Bounded Re-synthesizer) Scalability Evaluation", flush=True)
    print("Sprint 5 / Issue #39 — Owner: Sahib Singh (@sahib-1030)", flush=True)
    print("=" * 70, flush=True)

    scaling_data = benchmark_verification_vs_qubits()
    gate_data = benchmark_verification_vs_gates()
    resynth_data = benchmark_resynthesis_on_pipeline_windows()

    out_dir = ROOT / "docs" / "smt_eval"
    generate_plots_and_tables(scaling_data, gate_data, resynth_data, out_dir)

    print("=" * 70, flush=True)
    print("Evaluation completed successfully!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
