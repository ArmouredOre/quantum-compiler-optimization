"""Precision / recall / F1 of Module B against the Sprint-1 rule-based teacher."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

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
