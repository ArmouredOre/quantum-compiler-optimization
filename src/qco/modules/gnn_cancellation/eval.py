"""Precision / recall / F1 of Module B against the Sprint-1 rule-based teacher."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.gnn_cancellation.features import (
    CancellationCandidate,
    are_inverses,
    rule_based_candidates,
)


@dataclass(slots=True)
class ClassificationMetrics:
    """Micro-averaged binary classification counts and scores."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_gold: int
    n_pred: int


def pair_key(candidate: CancellationCandidate) -> tuple[int, int]:
    """Identity of a candidate for P/R/F1 (kind is ignored)."""
    return (candidate.a, candidate.b)


def metrics_from_counts(tp: int, fp: int, fn: int) -> ClassificationMetrics:
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return ClassificationMetrics(
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        n_gold=tp + fn,
        n_pred=tp + fp,
    )


def compare_candidates(
    gold: Sequence[CancellationCandidate],
    predicted: Sequence[CancellationCandidate],
) -> ClassificationMetrics:
    gold_set = {pair_key(c) for c in gold}
    pred_set = {pair_key(c) for c in predicted}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return metrics_from_counts(tp, fp, fn)


def adjacent_inverse_baseline(ir: IntermediateRepresentation) -> list[CancellationCandidate]:
    """Weaker teacher: only immediately consecutive inverse pairs.

    Used as the 'rule-based baseline' that the GNN must match or beat when both
    are scored against the full Sprint-1 generator on a held-out split.
    """
    out: list[CancellationCandidate] = []
    gates = ir.gates
    for i in range(len(gates) - 1):
        if are_inverses(gates[i], gates[i + 1]):
            out.append(CancellationCandidate(i, i + 1, "inverse_pair", 1.0))
    return out


def evaluate_predictor(
    predictor,
    circuits: Sequence[IntermediateRepresentation],
    *,
    threshold: float | None = None,
) -> ClassificationMetrics:
    """Micro-F1 of ``predictor.predict`` vs :func:`rule_based_candidates`."""
    tp = fp = fn = 0
    for ir in circuits:
        gold = rule_based_candidates(ir)
        kwargs = {} if threshold is None else {"threshold": threshold}
        predicted = predictor.predict(ir, **kwargs)
        m = compare_candidates(gold, predicted)
        tp += m.tp
        fp += m.fp
        fn += m.fn
    return metrics_from_counts(tp, fp, fn)


def evaluate_baseline(circuits: Sequence[IntermediateRepresentation]) -> ClassificationMetrics:
    """Micro-F1 of :func:`adjacent_inverse_baseline` vs the full rule generator."""
    tp = fp = fn = 0
    for ir in circuits:
        m = compare_candidates(rule_based_candidates(ir), adjacent_inverse_baseline(ir))
        tp += m.tp
        fp += m.fp
        fn += m.fn
    return metrics_from_counts(tp, fp, fn)


def held_out_split(
    items: Sequence,
    *,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list, list]:
    """Shuffle ``items`` and return ``(train, test)``."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    pooled = list(items)
    rng = random.Random(seed)
    rng.shuffle(pooled)
    if len(pooled) <= 1 or test_fraction == 0.0:
        return pooled, []
    n_test = max(1, int(round(len(pooled) * test_fraction)))
    n_test = min(n_test, len(pooled) - 1)
    return pooled[n_test:], pooled[:n_test]


# ---------------------------------------------------------------------------
# Sprint 3 (#25): verified ground truth, threshold calibration, P/R table
# ---------------------------------------------------------------------------

GoldCircuit = tuple[IntermediateRepresentation, list[CancellationCandidate]]


def train_val_test_split(
    items: Sequence,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list, list, list]:
    """Seeded, disjoint ``(train, val, test)`` split."""
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")
    train_val, test = held_out_split(items, test_fraction=test_fraction, seed=seed)
    rel_val = val_fraction / (1.0 - test_fraction)
    train, val = held_out_split(train_val, test_fraction=rel_val, seed=seed + 1)
    return train, val, test


def evaluate_on_gold(
    gold_circuits: Sequence[GoldCircuit],
    predict: Callable[[IntermediateRepresentation], Sequence[CancellationCandidate]],
) -> ClassificationMetrics:
    """Micro P/R/F1 of ``predict(ir)`` against precomputed gold candidates."""
    tp = fp = fn = 0
    for ir, gold in gold_circuits:
        m = compare_candidates(gold, predict(ir))
        tp += m.tp
        fp += m.fp
        fn += m.fn
    return metrics_from_counts(tp, fp, fn)


@dataclass(slots=True)
class ThresholdSweepPoint:
    threshold: float
    metrics: ClassificationMetrics


def sweep_thresholds(
    predictor,
    gold_circuits: Sequence[GoldCircuit],
    thresholds: Sequence[float] | None = None,
) -> list[ThresholdSweepPoint]:
    """Metrics at each decision threshold (``predictor.predict(ir, threshold=t)``)."""
    if thresholds is None:
        thresholds = [round(0.05 * k, 2) for k in range(1, 20)]
    return [
        ThresholdSweepPoint(
            t, evaluate_on_gold(gold_circuits, lambda ir, t=t: predictor.predict(ir, threshold=t))
        )
        for t in thresholds
    ]


def calibrate_threshold(
    predictor,
    val_gold: Sequence[GoldCircuit],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, list[ThresholdSweepPoint]]:
    """Pick the decision threshold maximising F1 on the validation split (ties -> higher)."""
    sweep = sweep_thresholds(predictor, val_gold, thresholds)
    if not sweep:
        raise ValueError("no thresholds to sweep")
    best = sweep[0]
    for p in sweep[1:]:
        if p.metrics.f1 >= best.metrics.f1:
            best = p
    return best.threshold, sweep


def build_pr_table(
    test_gold: Sequence[GoldCircuit],
    systems: dict[str, Callable[[IntermediateRepresentation], Sequence[CancellationCandidate]]],
) -> str:
    """Markdown P/R/F1 table; every system is scored against the same verified gold."""
    lines = [
        "| System | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, predict in systems.items():
        m = evaluate_on_gold(test_gold, predict)
        lines.append(
            f"| {name} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.tp} | {m.fp} | {m.fn} |"
        )
    return "\n".join(lines) + "\n"


def write_pr_table(path: str | Path, markdown: str, *, header: str = "") -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((header + "\n\n" if header else "") + markdown, encoding="utf-8")

import math

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

def _nll(logits, labels, T):
    tot = 0.0
    for z, y in zip(logits, labels):
        p = min(max(sigmoid(z / T), 1e-12), 1 - 1e-12)
        tot -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return tot / max(len(labels), 1)

def fit_temperature(logits, labels, lo=0.05, hi=20.0, n=200) -> float:
    if len(labels) < 2 or len(set(labels)) < 2:
        return 1.0
    grid = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)] + [1.0]
    return min(grid, key=lambda T: _nll(logits, labels, T))

def reliability_bins(probs, labels, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    return [(i / n_bins, (i + 1) / n_bins, len(b),
             sum(p for p, _ in b) / len(b), sum(y for _, y in b) / len(b))
            for i, b in enumerate(bins) if b]

def expected_calibration_error(probs, labels, n_bins=10) -> float:
    n = len(labels)
    return sum(c / n * abs(conf - acc)
               for _, _, c, conf, acc in reliability_bins(probs, labels, n_bins)) if n else 0.0

def plot_reliability(curves: dict, path) -> None:
    """curves: name -> (probs, labels). Points are annotated with bin counts. Needs matplotlib."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (p, y) in curves.items():
        rows = reliability_bins(p, y)
        ax.plot([r[3] for r in rows], [r[4] for r in rows], "o-",
                label=f"{name} (ECE={expected_calibration_error(p, y):.3f})")
        for r in rows:
            ax.annotate(str(r[2]), (r[3], r[4]), fontsize=6,
                        xytext=(2, 3), textcoords="offset points")
    ax.set_xlabel("confidence"); ax.set_ylabel("empirical reduction rate")
    ax.set_title("Reliability (labels = bin counts)", fontsize=9)
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

# --- window ranking ---------------------------------------------------------

def precision_at_k(per_circuit, k, rng=None) -> float:
    """per_circuit: list of [(score, label)], label in {0, 1, None}. Unknown labels are ignored.
    Ties are broken randomly so tie-heavy systems are not rewarded by window position."""
    rng = rng or random.Random(0)
    hits = tot = 0
    for items in per_circuit:
        known = [(s, y) for s, y in items if y is not None]
        rng.shuffle(known)
        known.sort(key=lambda t: -t[0])          # stable -> random order among ties
        top = known[:k]
        hits += sum(y for _, y in top); tot += len(top)
    return hits / tot if tot else 0.0

def rule_window_scorer(ir, windows):
    """Baseline: number of rule-based candidates fully inside the window."""
    cands = rule_based_candidates(ir)
    return [float(sum(1 for c in cands if s <= c.a and c.b < e)) for s, e in windows]

def window_ranking_table(test, systems, ks=(1, 3, 5), repeats=20, seed=0) -> str:
    """test: list[(ir, list[WindowLabel])]; systems: name -> fn(ir, windows) -> list[float]."""
    known = [r.label for _, rows in test for r in rows if r.label is not None]
    lines = [f"Base rate of reducible windows: {sum(known)}/{len(known)}\n",
             "| System | " + " | ".join(f"P@{k}" for k in ks) + " |",
             "|---|" + "---|" * len(ks)]
    for name, fn in systems.items():
        scored = [[(s, r.label) for s, r in zip(fn(ir, [(r.start, r.end) for r in rows]), rows)]
                  for ir, rows in test]
        vals = [sum(precision_at_k(scored, k, random.Random(seed + i)) for i in range(repeats)) / repeats
                for k in ks]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    return "\n".join(lines) + "\n"
