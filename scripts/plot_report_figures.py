"""Final-report figures drawn from the committed window_ranking.md (#42).

Nothing is re-evaluated and no number is typed in here: every bar is parsed from the report's tables,
so the figures cannot disagree with it. Error bars are the report's "+/-" (std over CV seeds only; they do
not capture the small number of positive circuits).

    python scripts/plot_report_figures.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RANKING = "### Ranking (all circuits"
FAMILY = "### Per-family GNN AUROC"


def read_table(text, heading):
    """Data rows of the first markdown table after `heading` (header and separator rows dropped)."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    except StopIteration:
        raise SystemExit(f"heading not found in report: {heading!r}")
    rows = []
    for l in lines[start + 1:]:
        if l.startswith("|"):
            rows.append([c.strip() for c in l.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]


def val(cell):
    """'0.294 ± 0.001' -> (0.294, 0.001); a bare number gets std 0."""
    mean, _, sd = cell.partition("±")
    return float(mean), (float(sd) if sd.strip() else 0.0)


def fig_pk(rows, out):
    ks = ["P@1", "P@3", "P@5"]
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for j, k in enumerate(ks):
        vals = [val(r[1 + j]) for r in rows]
        ax.bar([i + (j - 1) * w for i in range(len(rows))], [v[0] for v in vals], w,
               yerr=[v[1] for v in vals], capsize=2, label=k)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("precision@k (pooled over circuits)")
    ax.set_title("Window ranking, leave-family-out CV (error bars: seed std only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_family(rows, out):
    names = [f"{r[0]} ({r[1]} pos.)" for r in rows]
    vals = [val(r[2]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(names, [v[0] for v in vals], xerr=[v[1] for v in vals], capsize=3)
    ax.axvline(0.5, ls="--", lw=1, color="gray")
    ax.text(0.505, -0.45, "chance", color="gray", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("GNN window-head AUROC (out-of-fold)")
    ax.set_title("Per-family AUROC (error bars: seed std only)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "docs" / "gnn_eval" / "window_ranking.md")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "gnn_eval")
    a = ap.parse_args()
    text = a.src.read_text(encoding="utf-8")
    if "Â±" in text:
        raise SystemExit("report contains double-encoded '±'; repair it before plotting")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig_pk(read_table(text, RANKING), a.out_dir / "fig_window_pk.png")
    fig_family(read_table(text, FAMILY), a.out_dir / "fig_family_auroc.png")
    print("wrote", a.out_dir / "fig_window_pk.png", "and", a.out_dir / "fig_family_auroc.png")


if __name__ == "__main__":
    main()
    