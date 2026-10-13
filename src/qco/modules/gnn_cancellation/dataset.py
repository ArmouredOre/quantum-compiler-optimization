"""Dump / load cancellation labels (rule-based or verified) for Module B training."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any, Sequence

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.features import (
    CancellationCandidate,
    overlapping_gate_pairs,
    rule_based_candidates,
)

LabelledCircuit = tuple[IntermediateRepresentation, list[CancellationCandidate]]

LABEL_SOURCES = ("rule", "verified")


def verify_labels(
    ir: IntermediateRepresentation,
    candidates: Sequence[CancellationCandidate],
    *,
    checker: Any | None = None,
    backend: str = "auto",
) -> list[bool | None]:
    """Verified status per candidate: True / False, or None if unknown.

    ``None`` means the checker crashed on that candidate (it reports
    ``method == "error"``). That is not the same as "not equivalent", so
    callers must never train on it.
    """
    if not candidates:
        return []
    # Lazy import: keeps Module B importable without the verifier loaded.
    from qco.modules.smt_verifier.equivalence import verify_candidates_detailed

    try:
        results = verify_candidates_detailed(
            ir, list(candidates), checker=checker, backend=backend
        )
    except Exception as exc:
        warnings.warn(
            f"verification failed on {ir.name!r}: {exc!r}; labels unknown",
            stacklevel=2,
        )
        return [None] * len(candidates)
    if len(results) != len(candidates):
        warnings.warn(
            f"verifier returned {len(results)} results for {len(candidates)} "
            f"candidates on {ir.name!r}; labels unknown",
            stacklevel=2,
        )
        return [None] * len(candidates)
    return [None if res.method == "error" else bool(res.equivalent) for _, res in results]

def candidates_to_records(
    ir: IntermediateRepresentation,
    candidates: list[CancellationCandidate] | None = None,
    *,
    label_source: str = "rule",
    checker: Any | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Serialize a circuit and its labels (``label_source`` is "rule" or "verified")."""
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"label_source must be one of {LABEL_SOURCES}, got {label_source!r}")
    cands = rule_based_candidates(ir) if candidates is None else candidates
    if label_source == "verified":
        status = verify_labels(ir, cands, checker=checker, backend=backend)
    else:
        status = [None] * len(cands)  # not checked

    rows = []
    for c, ok in zip(cands, status):
        if label_source == "rule":
            label: int | None = 1
        else:
            label = None if ok is None else int(ok)
        rows.append(
            {
                "a": c.a,
                "b": c.b,
                "kind": c.kind,
                "score": c.score,
                "indices": list(c.gate_indices),
                "provisional_label": 1,  # kept for backward compatibility
                "verified": ok,
                "label": label,
            }
        )
    return {
        "name": ir.name,
        "num_qubits": ir.num_qubits,
        "gate_count": ir.gate_count(),
        "qasm": ir.to_qasm2(),
        "label_source": label_source,
        "candidates": rows,
    }


def dump_candidates(
    ir: IntermediateRepresentation,
    path: str | Path,
    candidates: list[CancellationCandidate] | None = None,
    *,
    label_source: str = "rule",
    checker: Any | None = None,
    backend: str = "auto",
) -> list[CancellationCandidate]:
    """Write ``circuit -> list[CancellationCandidate]`` JSON to ``path``."""
    cands = rule_based_candidates(ir) if candidates is None else candidates
    payload = candidates_to_records(
        ir, cands, label_source=label_source, checker=checker, backend=backend
    )
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cands


def load_candidates(path: str | Path) -> tuple[IntermediateRepresentation, list[CancellationCandidate]]:
    """Inverse of :func:`dump_candidates`; returns positives only."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ir = from_qasm2(payload["qasm"])
    ir.name = payload.get("name", ir.name)
    cands = [
        CancellationCandidate(
            a=row["a"],
            b=row["b"],
            kind=row["kind"],
            score=float(row.get("score", 1.0)),
            indices=tuple(row.get("indices") or ()),
        )
        for row in payload.get("candidates", [])
        if row.get("label", 1) == 1  # 0 = rejected, None = unknown, missing = old file
    ]
    return ir, cands


def dump_suite(
    circuits_dir: str | Path,
    out_dir: str | Path,
    *,
    label_source: str = "rule",
    backend: str = "auto",
) -> list[Path]:
    """Dump a JSON label file for every ``.qasm`` in ``circuits_dir``."""
    src = Path(circuits_dir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    checker = None
    if label_source == "verified":
        from qco.modules.smt_verifier.equivalence import EquivalenceChecker

        checker = EquivalenceChecker(backend=backend)  # one checker, reused

    written: list[Path] = []
    for qasm_path in sorted(src.glob("*.qasm")):
        ir = from_qasm2(qasm_path.read_text(encoding="utf-8"))
        ir.name = qasm_path.stem
        out_path = dest / f"{qasm_path.stem}.json"
        dump_candidates(ir, out_path, label_source=label_source, checker=checker, backend=backend)
        written.append(out_path)
    return written


def load_suite(label_dir: str | Path) -> list[LabelledCircuit]:
    """Load every dumped ``.json`` label file in ``label_dir``."""
    return [load_candidates(path) for path in sorted(Path(label_dir).glob("*.json"))]


def load_training_dataset(dataset: Any) -> list[LabelledCircuit]:
    """Coerce paths / IRs / ``(IR, candidates)`` pairs into a labelled list.

    * directory of ``.json`` dumps from :func:`dump_suite`
    * directory of ``.qasm`` files (labels generated by the rule-based teacher)
    * a single ``.json`` / ``.qasm`` path
    * iterable of :class:`IntermediateRepresentation`
    * iterable of ``(ir, candidates)`` pairs
    """
    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if path.is_dir():
            jsons = sorted(path.glob("*.json"))
            if jsons:
                return [load_candidates(p) for p in jsons]
            qasms = sorted(path.glob("*.qasm"))
            if qasms:
                return [_labelled_qasm(p) for p in qasms]
            raise FileNotFoundError(f"no .json or .qasm files in {path}")
        if path.suffix == ".json":
            return [load_candidates(path)]
        if path.suffix == ".qasm":
            return [_labelled_qasm(path)]
        raise ValueError(f"unsupported training path: {path}")

    items = list(dataset)
    if not items:
        return []
    first = items[0]
    if isinstance(first, IntermediateRepresentation):
        return [(ir, rule_based_candidates(ir)) for ir in items]
    if isinstance(first, (tuple, list)) and len(first) == 2:
        return [(ir, list(cands)) for ir, cands in items]
    raise TypeError(
        "dataset must be a path, a list of IntermediateRepresentation, "
        "or a list of (ir, candidates) pairs"
    )


def sample_training_pairs(
    ir: IntermediateRepresentation,
    positives: Sequence[CancellationCandidate],
    *,
    negative_ratio: int = 4,
    rng: random.Random | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Positive rule pairs plus sampled overlapping-support negatives."""
    rng = rng or random.Random(0)
    pos = {(c.a, c.b) for c in positives}
    pool = overlapping_gate_pairs(ir)
    neg = [p for p in pool if p not in pos]
    rng.shuffle(neg)
    if pos:
        neg = neg[: max(len(pos) * negative_ratio, 1)]
    else:
        neg = neg[: min(len(neg), 16)]
    pairs = list(pos) + neg
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    combined = list(zip(pairs, labels))
    rng.shuffle(combined)
    if not combined:
        return [], []
    out_pairs, out_labels = zip(*combined)
    return list(out_pairs), list(out_labels)


def _labelled_qasm(path: Path) -> LabelledCircuit:
    ir = from_qasm2(path.read_text(encoding="utf-8"))
    ir.name = path.stem
    return ir, rule_based_candidates(ir)
