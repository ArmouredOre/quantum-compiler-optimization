"""Publication figure for the graph layer & partitioner section (Sprint 6 / Issue #48).

Reads its numbers from the committed #46 outputs, so nothing is typed in by hand:
  docs/partitioner_tuning.md   (written by benchmarks/tune_partition.py --downstream)
  docs/parallel_blocks.md      (the measured-speedup table)
Writes docs/graph_layer/graph_layer_summary.{svg,png}. Run from anywhere:
    python scripts/plot_graph_layer_summary.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TUNING = ROOT / "docs" / "partitioner_tuning.md"
PARALLEL = ROOT / "docs" / "parallel_blocks.md"
CIRCUITS = ("rand_cliffordt_n20", "rand_cnotpauli_n20")  # the two circuits where partitioning matters
QUBITS = 8  # the pipeline fixes max_block_qubits=8, so only those rows have a gate-reduction column

C_GREEDY, C_FALLBACK, C_PARALLEL, C_GREY = "#2563eb", "#dc2626", "#059669", "#94a3b8"


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def read_tuning(path: Path) -> dict:
    """{circuit: {"greedy": {(cut, red): [max_gates...]}, "fallback": [(max_gates, cut, red)...]}}"""
    if not path.exists():
        sys.exit(f"missing {path}; run benchmarks/tune_partition.py --downstream --out {path.relative_to(ROOT)}")
    out: dict = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"###\s+(\S+)", line)
        if m:
            current = m.group(1)
            continue
        if current not in CIRCUITS or not line.startswith("|"):
            continue
        c = cells(line)
        if len(c) < 13 or c[0] in ("backend",) or set(c[0]) <= {"-"}:
            continue
        backend, gates, qubits, _blocks, cut = c[0], int(c[1]), int(c[2]), c[3], int(c[4])
        red = c[11]
        if qubits != QUBITS or red == "-":
            continue
        entry = out.setdefault(current, {"greedy": {}, "fallback": []})
        if backend == "greedy":
            entry["greedy"].setdefault((cut, float(red)), []).append(gates)
        elif backend == "fallback":
            entry["fallback"].append((gates, cut, float(red)))
    for name in CIRCUITS:
        if name not in out or not out[name]["greedy"] or len(out[name]["fallback"]) < 2:
            sys.exit(f"{path.name}: no usable greedy/fallback rows for {name} (max_block_qubits={QUBITS}, downstream columns)")
        out[name]["fallback"].sort()
    return out


def read_parallel(path: Path) -> list[tuple[str, float, bool]]:
    """[(label, speedup, identical)] for the 2-worker rows of the measured-speedup table."""
    if not path.exists():
        sys.exit(f"missing {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        c = cells(line)
        if len(c) < 7 or c[2] != "2" or not re.fullmatch(r"[\d.]+x", c[5]):
            continue
        circuit = c[0].replace("rand_", "").replace("_n20", "")
        mode = "no resynth" if c[1].startswith("no") else "resynth, 20 s"
        rows.append((f"{circuit}\n{mode}", float(c[5][:-1]), c[6].lower() == "yes"))
    if not rows:
        sys.exit(f"{path.name}: no 2-worker rows found")
    return rows


def tradeoff_panel(ax, data: dict, name: str, title: str) -> None:
    fb = data[name]["fallback"]
    ax.plot([p[1] for p in fb], [p[2] for p in fb], "o-", color=C_FALLBACK, linewidth=2.2, markersize=6,
            label="Fallback (no width cap)")
    for i, (g, cut, red) in enumerate(fb):
        ax.annotate(str(g), (cut, red), textcoords="offset points",
                    xytext=(-14, 8) if i == 0 else (5, -11), fontsize=8, color=C_FALLBACK)
    for i, ((cut, red), gs) in enumerate(sorted(data[name]["greedy"].items(), key=lambda kv: -kv[0][1])):
        label = str(gs[0]) if len(gs) == 1 else f"{min(gs)}-{max(gs)}"
        ax.plot(cut, red, "s", color=C_GREEDY, markersize=9, label="Greedy (blocks <= 8 qubits)" if i == 0 else None)
        ax.annotate(f"max_gates {label}", (cut, red), textcoords="offset points",
                    xytext=(10, 14 if i == 0 else -16), fontsize=8, color=C_GREEDY)
    ax.set_xlabel("Cut (DAG edges crossing blocks)")
    ax.set_ylabel("Downstream gate reduction (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95)


def generate(out_dir: Path) -> None:
    tuning, parallel = read_tuning(TUNING), read_parallel(PARALLEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    tradeoff_panel(axes[0], tuning, CIRCUITS[0], f"(a) {CIRCUITS[0]}: cut vs gate reduction")
    tradeoff_panel(axes[1], tuning, CIRCUITS[1], f"(b) {CIRCUITS[1]}: cut vs gate reduction")
    for ax in axes[:2]:
        ax.set_ylim(bottom=-1)

    ax = axes[2]
    bars = ax.bar(range(len(parallel)), [p[1] for p in parallel], width=0.6,
                  color=[C_GREY if p[1] < 1 else C_PARALLEL for p in parallel])
    top = max(p[1] for p in parallel)
    for b, (_, s, ok) in zip(bars, parallel):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.03 * top, f"{s:.2f}x" + ("" if ok else "\n(output differs)"),
                ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(parallel)))
    ax.set_xticklabels([p[0] for p in parallel], fontsize=8)
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel("Block-stage speedup (2 workers / serial)")
    ax.set_title("(c) Parallel block processing", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Graph Layer & Partitioner: Block-Size Trade-off and Parallel Scaling",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    svg, png = out_dir / "graph_layer_summary.svg", out_dir / "graph_layer_summary.png"
    plt.savefig(svg, format="svg")
    plt.savefig(png, format="png")
    plt.close()
    print(f"--> Saved graph-layer figures to {svg} and {png}")


if __name__ == "__main__":
    generate(ROOT / "docs" / "graph_layer")
    