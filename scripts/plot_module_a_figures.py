"""Final-report figures for Module A (#41), drawn from the committed result files.

Nothing is re-evaluated and no number is typed in: every bar comes from the JSON / CSV / markdown tables
written by the evaluation scripts (generalization, ablations, closed loop, backend benchmarks, scaling), so a
figure cannot disagree with its table. Error bars are the std over training seeds only (2-3 seeds).

    python scripts/plot_module_a_figures.py [--out docs/rl_training/figures]

Writes fig_*.png (200 dpi, for the report) and fig_*.svg.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "docs" / "rl_training"
# fixed categorical order (dataviz reference palette); a method keeps its colour in every figure
COL = {"stub": "#8d8c87", "cr_only": "#eb6834", "greedy": "#1baf7a", "agent_cr": "#2a78d6", "agent_nocr": "#6250d6",
       "proxy": "#2a78d6", "pipeline": "#e87ba4", "sc": "#2a78d6", "ion": "#eda100", "heur": "#8d8c87"}
LABEL = {"stub": "untrained stub", "cr_only": "CR only", "greedy": "greedy depth + CR", "agent_cr": "PPO + CR",
         "agent_nocr": "PPO, no CR"}
BUCKETS = ("5-6", "8-12", "16-20", "40")


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 100})


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def legend_below(ax, ncol, fontsize=8):
    ax.legend(frameon=False, fontsize=fontsize, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol)


def bars(ax, groups, series, errs=None, colors=None, labels=None):
    n = len(series)
    w = 0.8 / n
    for i, (k, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, vals, w * 0.92, color=(colors or {}).get(k), label=(labels or {}).get(k, k),
               yerr=None if errs is None else errs[k], error_kw={"lw": 0.8, "capsize": 1.5})
    ax.set_xticks(range(len(groups)), groups)


def fig_generalization(out: Path):
    small = load(RL / "generalization.json")["summary"]["per_bucket"]
    large = load(RL / "large" / "generalization_40q.json")["summary"]["per_bucket"]
    pb = {**small, "40": large["40"]}
    methods = ["stub", "cr_only", "greedy", "agent_cr", "agent_nocr"]
    ns = [pb[b]["n_circuits"] for b in BUCKETS]
    groups = [f"{b} qubits\n(n={n})" for b, n in zip(BUCKETS, ns)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in ((axes[0], "depth_red_pct", "Depth reduction (%)"), (axes[1], "gate_red_pct", "Gate-count reduction (%)")):
        bars(ax, groups, {m: [pb[b][m][key] for b in BUCKETS] for m in methods},
             None if key != "depth_red_pct" else {m: [pb[b][m]["depth_red_std_over_seeds"] for b in BUCKETS] for m in methods},
             COL, LABEL)
        ax.set_title(title, loc="left", fontsize=10)
    legend_below(axes[0], 3)
    fig.suptitle("Zero-shot generalization of the 5-qubit policy (benchmark circuits, 3 seeds)", x=0.01, y=1.04, ha="left", fontsize=11)
    save(fig, out, "fig_generalization")


def fig_per_circuit(out: Path):
    pc = {**load(RL / "generalization.json")["summary"]["per_circuit"], **load(RL / "large" / "generalization_40q.json")["summary"]["per_circuit"]}
    meta = {c["name"]: c["qubits"] for c in load(RL / "generalization.json")["result"]["circuits"]}
    meta.update({c["name"]: c["qubits"] for c in load(RL / "large" / "generalization_40q.json")["result"]["circuits"]})
    names = sorted(pc, key=lambda n: (meta[n], n))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (m, mk) in enumerate((("greedy", "o"), ("agent_cr", "D"))):
        ax.scatter(range(len(names)), [pc[n][m]["depth_red_pct"] for n in names], marker=mk, s=26, color=COL[m], label=LABEL[m], zorder=3)
    ax.set_xticks(range(len(names)), [f"{n} ({meta[n]}q)" for n in names], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("depth reduction (%)")
    ax.set_title("Per circuit: PPO + CR vs the greedy baseline, sorted by qubit count", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig_generalization_per_circuit")


def fig_ablations(out: Path):
    a = load(RL / "ablations" / "ablations.json")["summary"]
    bk = ["5-6q", "8-12q", "16-20q", "40q"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cr = {"baseline (CR on)": "CR on", "CR-trained, CR off at test": "CR on in training, off at test", "CR off (train + run)": "CR off"}
    palette = ["#2a78d6", "#eda100", "#eb6834"]
    series = {cr[k]: [a[k][b]["depth_red_pct"][0] for b in bk] for k in cr}
    errs = {cr[k]: [a[k][b]["depth_red_pct"][1] for b in bk] for k in cr}
    bars(axes[0], bk, series, errs, dict(zip(series, palette)))
    axes[0].set_title("CR pass on/off: depth reduction (%)", loc="left", fontsize=10)
    legend_below(axes[0], 3)
    wv = ["baseline (CR on)", "depth only", "gates only", "no fidelity term", "fidelity x4", "no terminal bonus", "terminal x3"]
    wp = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    series = {k.replace(" (CR on)", " (default)"): [a[k][b]["reward"][0] for b in bk] for k in wv}
    errs = {k.replace(" (CR on)", " (default)"): [a[k][b]["reward"][1] for b in bk] for k in wv}
    bars(axes[1], bk, series, errs, dict(zip(series, wp)))
    axes[1].set_title("Reward-weight sweep: Stage 6 reward", loc="left", fontsize=10)
    legend_below(axes[1], 4, 7)
    fig.suptitle("Module A ablations (5-qubit training, 2 seeds x 40k steps)", x=0.01, y=1.04, ha="left", fontsize=11)
    save(fig, out, "fig_ablations")


def fig_closed_loop(out: Path):
    c = load(RL / "closed_loop" / "closed_loop.json")
    sets = list(c["sets"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    series = {m: [c["results"][m][s]["reward"] for s in sets] for m in ("stub", "proxy", "pipeline")}
    errs = {m: [c["results"][m][s]["reward_std"] for s in sets] for m in series}
    lab = {"stub": "untrained stub", "proxy": "PPO, proxy reward", "pipeline": "PPO, real Stage 6 reward"}
    bars(axes[0], [f"{s.replace('bench_', '')}\n(n={c['sets'][s]})" for s in sets], series, errs, COL, lab)
    axes[0].set_title("End-to-end Stage 6 reward", loc="left", fontsize=10)
    legend_below(axes[0], 3)
    for kind, ls in (("proxy", "-"), ("pipeline", "--")):
        curves = []
        for s in c["seeds"]:
            rows = list(csv.DictReader((RL / "closed_loop" / f"{kind}_seed{s}" / "metrics.csv").open()))
            curves.append([(int(r["timesteps"]), float(r["eval_depth_red_pct"])) for r in rows if r["eval_depth_red_pct"]])
        for k, cv in enumerate(curves):
            axes[1].plot(*zip(*cv), ls, color=COL[kind], alpha=0.85, lw=1.2, label=lab[kind] if k == 0 else None)
    axes[1].set_xlabel("training timesteps")
    axes[1].set_title("Held-out depth reduction during training (%)", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Closed loop (#35): proxy reward vs the real Stage 6 reward, 2 seeds x 40k steps", x=0.01, y=1.04, ha="left", fontsize=11)
    save(fig, out, "fig_closed_loop")


def fig_backend(out: Path):
    b = load(RL / "backend_benchmarks" / "backend_benchmarks.json")
    names = list(b)
    tests = [("superconducting_ibm_like", "Test backend: superconducting"), ("trapped_ion_like", "Test backend: trapped ion")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (t, title) in zip(axes, tests):
        series = {"trained on superconducting": [b[n]["matrix"][t]["superconducting_ibm_like"]["mean"] for n in names],
                  "trained on trapped ion": [b[n]["matrix"][t]["trapped_ion_like"]["mean"] for n in names],
                  "greedy time heuristic": [b[n]["matrix"][t]["greedy_time"]["mean"] for n in names]}
        errs = {"trained on superconducting": [b[n]["matrix"][t]["superconducting_ibm_like"]["std_over_seeds"] for n in names],
                "trained on trapped ion": [b[n]["matrix"][t]["trapped_ion_like"]["std_over_seeds"] for n in names],
                "greedy time heuristic": [0] * len(names)}
        bars(ax, [f"{n} qubits" for n in names], series, errs,
             {"trained on superconducting": COL["sc"], "trained on trapped ion": COL["ion"], "greedy time heuristic": COL["heur"]})
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("execution-time reduction (%)")
    legend_below(axes[0], 3)
    fig.suptitle("Per-backend policies on the benchmarks (#29 / #37, 3 seeds)", x=0.01, y=1.04, ha="left", fontsize=11)
    save(fig, out, "fig_backend_policies")


def parse_scaling(md: str):
    """Rows of the gate-count sweep table of docs/module_a_complexity_and_scaling.md."""
    lines = md.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("### Gate-count sweep"))
    rows = []
    for l in lines[start + 1:]:
        if l.startswith("|"):
            rows.append([c.strip() for c in l.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]


def fig_scaling(out: Path):
    rows = parse_scaling((ROOT / "docs" / "module_a_complexity_and_scaling.md").read_text(encoding="utf-8"))
    G = [float(r[1]) for r in rows]
    cols = {"cr_pass (raw circuit)": 4, "legal_actions": 5, "encode_state (lookahead)": 6, "env_step": 7, "episode (whole circuit)": 8}
    pal = ["#eb6834", "#1baf7a", "#2a78d6", "#eda100", "#6250d6"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for (name, c), color in zip(cols.items(), pal):
        pts = [(g, float(r[c])) for g, r in zip(G, rows) if r[c] != "-"]
        ax.plot(*zip(*pts), "o-", ms=3.5, lw=1.3, color=color, label=name)
    g0 = G[0]
    for slope, ls in ((1, ":"), (2, "--")):
        ax.plot([g0, G[-1]], [1.5, 1.5 * (G[-1] / g0) ** slope], ls, color="#8d8c87", lw=0.9)
        ax.text(G[-1], 1.5 * (G[-1] / g0) ** slope, f" slope {slope}", fontsize=7, color="#52514e", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gates G (20 qubits)")
    ax.set_ylabel("time (ms)")
    ax.set_title("Module A cost drivers vs circuit size", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, out, "fig_scaling")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RL / "figures"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    for fn in (fig_generalization, fig_per_circuit, fig_ablations, fig_closed_loop, fig_backend, fig_scaling):
        fn(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
