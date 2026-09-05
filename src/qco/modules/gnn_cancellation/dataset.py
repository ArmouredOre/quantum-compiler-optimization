"""Dump / load provisional cancellation labels for Module B training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.features import (
    CancellationCandidate,
    rule_based_candidates,
)


def candidates_to_records(
    ir: IntermediateRepresentation,
    candidates: list[CancellationCandidate] | None = None,
) -> dict[str, Any]:
    """Serialize a circuit and its provisional (rule-based) labels."""
    cands = rule_based_candidates(ir) if candidates is None else candidates
    return {
        "name": ir.name,
        "num_qubits": ir.num_qubits,
        "gate_count": ir.gate_count(),
        "qasm": ir.to_qasm2(),
        "candidates": [
            {
                "a": c.a,
                "b": c.b,
                "kind": c.kind,
                "score": c.score,
                "indices": list(c.gate_indices),
                "provisional_label": 1,
            }
            for c in cands
        ],
    }


def dump_candidates(
    ir: IntermediateRepresentation,
    path: str | Path,
    candidates: list[CancellationCandidate] | None = None,
) -> list[CancellationCandidate]:
    """Write ``circuit -> list[CancellationCandidate]`` JSON to ``path``."""
    cands = rule_based_candidates(ir) if candidates is None else candidates
    payload = candidates_to_records(ir, cands)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cands


def load_candidates(path: str | Path) -> tuple[IntermediateRepresentation, list[CancellationCandidate]]:
    """Inverse of :func:`dump_candidates`."""
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
    ]
    return ir, cands


def dump_suite(circuits_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Dump a JSON label file for every ``.qasm`` in ``circuits_dir``."""
    src = Path(circuits_dir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for qasm_path in sorted(src.glob("*.qasm")):
        ir = from_qasm2(qasm_path.read_text(encoding="utf-8"))
        ir.name = qasm_path.stem
        out_path = dest / f"{qasm_path.stem}.json"
        dump_candidates(ir, out_path)
        written.append(out_path)
    return written
