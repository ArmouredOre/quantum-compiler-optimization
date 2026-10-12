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


# Hyperedge weight = gate cost in CX-equivalents (standard decompositions), so
# the partitioner avoids cutting the expensive interactions first.
_CX_COST = {"swap": 3, "rxx": 2, "ryy": 2, "rzz": 2, "ccx": 6, "cswap": 8}


def gate_cost(name: str) -> int:
    return _CX_COST.get(name, 1)


def build_hypergraph(ir: IntermediateRepresentation, window: int | None = None) -> QubitHypergraph:
    """Build the qubit-interaction hypergraph.

    Each multi-qubit gate contributes its qubit set with weight ``gate_cost``;
    identical qubit sets are merged and their weights summed. If ``window`` is
    given, multi-qubit gates within each window of that many gates are first
    grouped by shared qubits (connected components), and each group becomes one
    hyperedge: coarser coupling, fewer edges.
    """
    weights: dict[frozenset[int], int] = {}
    if window is None:
        groups = [(frozenset(g.qubits), gate_cost(g.name)) for g in ir.gates if g.is_multi_qubit]
    else:
        groups = []
        for start in range(0, len(ir.gates), window):
            groups.extend(_interaction_groups(ir.gates[start : start + window]))
    for qs, w in groups:
        weights[qs] = weights.get(qs, 0) + w
    hg = QubitHypergraph(num_qubits=ir.num_qubits)
    for qs, w in weights.items():
        hg.add(qs, w)
    return hg


def _interaction_groups(gates) -> list[tuple[frozenset[int], int]]:
    """Connected components (by shared qubits) of the multi-qubit gates."""
    groups: list[tuple[set[int], int]] = []
    for g in gates:
        if not g.is_multi_qubit:
            continue
        qs, w = set(g.qubits), gate_cost(g.name)
        for other in [grp for grp in groups if grp[0] & qs]:
            groups.remove(other)
            qs |= other[0]
            w += other[1]
        groups.append((qs, w))
    return [(frozenset(qs), w) for qs, w in groups]
