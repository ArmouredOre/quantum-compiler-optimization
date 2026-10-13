"""Generate publication-ready optimality and scaling plots for Module C (Review 3 Deliverable).

Sprint 6 / Issue #43 — Owner: Sahib Singh (@sahib-1030)
Generates:
- docs/smt_eval/optimality_and_scaling_summary.svg & .png (4-panel publication figure)
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_sprint6_report_figures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Professional color palette
    c_phase = "#059669"     # Emerald green
    c_numeric = "#2563eb"   # Royal blue
    c_z3 = "#dc2626"        # Crimson red
    c_accent = "#7c3aed"     # Deep purple
    c_bar = "#0284c7"        # Sky blue

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # -------------------------------------------------------------------------
    # Panel 1: Verification Runtime vs Qubit Count (Break 8-Qubit Ceiling)
    # -------------------------------------------------------------------------
    qubits_poly = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
    time_poly = [0.00014, 0.00019, 0.00036, 0.00013, 0.00026, 0.00022, 0.00049, 0.00034,
                 0.00031, 0.00077, 0.00041, 0.00060, 0.00063, 0.00110, 0.00164, 0.00191]

    qubits_num = [2, 3, 4, 5, 6, 7, 8]
    time_num = [0.01009, 0.00232, 0.00452, 0.00665, 0.00984, 0.03103, 1.16504]

    qubits_z3 = [2, 3, 4]
    time_z3 = [0.13000, 0.03136, 0.04060]

    ax1.plot(qubits_poly, time_poly, "o-", color=c_phase, linewidth=2.5, markersize=6, label="Phase-Poly SMT (Novelty N2)")
    ax1.plot(qubits_num, time_num, "s--", color=c_numeric, linewidth=2, markersize=6, label="Numeric (Dense Unitary)")
    ax1.plot(qubits_z3, time_z3, "^:", color=c_z3, linewidth=2, markersize=6, label="Z3-Dense SMT")

    # Ceiling line at 8 qubits
    ax1.axvline(x=8, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.text(8.5, 0.05, "Dense Unitary Ceiling\n(n = 8 qubits)", color="#b91c1c", fontweight="bold", fontsize=9)

    ax1.set_yscale("log")
    ax1.set_xlabel("Qubit Count ($n$)")
    ax1.set_ylabel("Verification Runtime (s, log-scale)")
    ax1.set_title("(a) Verification Runtime Scaling vs Qubits", fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.95)

    # -------------------------------------------------------------------------
    # Panel 2: Phase-Polynomial Runtime vs Gate Count
    # -------------------------------------------------------------------------
    gates = [46, 138, 276, 460, 690, 1058, 1426]
    gate_times_ms = [1.09, 0.58, 3.02, 2.08, 2.23, 3.57, 4.65]

    ax2.plot(gates, gate_times_ms, "D-", color=c_accent, linewidth=2.2, markersize=6, label="Empirical Runtime ($n=16$)")
    
    # Linear trend fit
    fit_coef = np.polyfit(gates, gate_times_ms, 1)
    fit_line = np.polyval(fit_coef, gates)
    ax2.plot(gates, fit_line, "--", color="#6b7280", linewidth=1.5, label=f"Linear Fit ($\mathcal{{O}}(k)$)")

    ax2.set_xlabel("Circuit Gate Count ($k$)")
    ax2.set_ylabel("Verification Runtime (ms)")
    ax2.set_title("(b) Phase-Poly Runtime vs Gate Count ($n=16$)", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.95)

    # -------------------------------------------------------------------------
    # Panel 3: Bounded Re-Synthesis Optimality on Benchmark Circuits
    # -------------------------------------------------------------------------
    circuits = ["rand_cnotpauli", "rand_cliffordt", "qaoa_n6", "qft_n5"]
    windows_eval = [8, 8, 8, 2]
    windows_opt = [1, 0, 0, 0]
    cache_hits = [0, 0, 1, 1]

    x = np.arange(len(circuits))
    width = 0.28

    rects1 = ax3.bar(x - width, windows_eval, width, label="Evaluated Windows", color="#94a3b8")
    rects2 = ax3.bar(x, windows_opt, width, label="Optimized Windows", color=c_phase)
    rects3 = ax3.bar(x + width, cache_hits, width, label="Cache Hits", color=c_bar)

    ax3.set_ylabel("Window Count")
    ax3.set_title("(c) Bounded Exact Re-Synthesis Windows & Cache", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(circuits)
    ax3.legend(loc="upper right", framealpha=0.95)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 4: Memory Footprint: Dense State-Space vs Phase-Polynomial
    # -------------------------------------------------------------------------
    n_space = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    dense_mem_mb = (16.0 * (4.0 ** n_space)) / (1024.0 * 1024.0)
    # Phase-poly bitvector space: n^2 + m*n bits for typical m = 50 gates -> KB
    poly_mem_kb = (n_space ** 2 + 50 * n_space) / 8.0 / 1024.0

    ax4.plot(n_space, dense_mem_mb, "s--", color=c_numeric, linewidth=2, label="Dense Unitary Matrix (MB)")
    ax4.axhline(y=1024, color="#ef4444", linestyle=":", linewidth=1.5, label="1 GB Memory Limit")
    ax4.axhline(y=16384, color="#b91c1c", linestyle="--", linewidth=1.5, label="16 GB RAM Ceiling")

    ax4.set_yscale("log")
    ax4.set_xlabel("Qubit Count ($n$)")
    ax4.set_ylabel("Memory Requirement (MB, log-scale)")
    ax4.set_title("(d) Asymptotic State-Space Memory Explosion", fontweight="bold")
    ax4.grid(True, which="both", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", framealpha=0.95)

    plt.suptitle("Module C: SMT Equivalence Verification & Bounded Re-Synthesis Evaluation", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_svg = out_dir / "optimality_and_scaling_summary.svg"
    out_png = out_dir / "optimality_and_scaling_summary.png"
    plt.savefig(out_svg, format="svg")
    plt.savefig(out_png, format="png")
    plt.close()
    print(f"--> Saved Sprint 6 report figures to {out_svg} and {out_png}")


if __name__ == "__main__":
    generate_sprint6_report_figures(ROOT / "docs" / "smt_eval")
