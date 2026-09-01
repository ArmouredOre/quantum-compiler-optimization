"""Stage 2 — qubit-interaction hypergraph.

Vertices are qubits; each hyperedge is the qubit set of one multi-qubit gate
(single-qubit gates add no coupling). This is the structure handed to the
multilevel partitioner (KaHyPar / METIS) in Stage 3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from qco.ir.intermediate_representation import IntermediateRepresentation


@dataclass(slots=True)
class QubitHypergraph:
    num_qubits: int
    # hyperedges[i] = (frozenset_of_qubits, weight)
    hyperedges: list[tuple[frozenset[int], int]] = field(default_factory=list)

    def add(self, qubits: frozenset[int], weight: int = 1) -> None:
        self.hyperedges.append((qubits, weight))

    def qubit_degrees(self) -> dict[int, int]:
        deg: Counter[int] = Counter()
        for qs, w in self.hyperedges:
            for q in qs:
                deg[q] += w
        return {q: deg.get(q, 0) for q in range(self.num_qubits)}

    def to_kahypar_hmetis(self) -> str:
        """Serialize to the hMETIS/KaHyPar hypergraph file format."""
        lines = [f"{len(self.hyperedges)} {self.num_qubits} 1"]  # '1' = weighted edges
        for qs, w in self.hyperedges:
            lines.append(f"{w} " + " ".join(str(q + 1) for q in sorted(qs)))
        return "\n".join(lines) + "\n"


def build_hypergraph(ir: IntermediateRepresentation, window: int | None = None) -> QubitHypergraph:
    """Build the qubit-interaction hypergraph.

    If ``window`` is given, multi-qubit gates within a sliding window of that many
    gates are merged into a single hyperedge (coarser coupling, fewer edges).
    """
    hg = QubitHypergraph(num_qubits=ir.num_qubits)
    if window is None:
        for g in ir.gates:
            if g.is_multi_qubit:
                hg.add(frozenset(g.qubits))
        return hg

    bucket: set[int] = set()
    count = 0
    for g in ir.gates:
        if g.is_multi_qubit:
            bucket.update(g.qubits)
        count += 1
        if count >= window and bucket:
            hg.add(frozenset(bucket), weight=len(bucket))
            bucket, count = set(), 0
    if bucket:
        hg.add(frozenset(bucket), weight=len(bucket))
    return hg
