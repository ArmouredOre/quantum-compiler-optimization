"""Stage 2 — gate-dependency DAG.

Nodes are gate indices into ``IntermediateRepresentation.gates``; an edge
``i -> j`` means gate ``j`` uses a qubit last written by gate ``i``. Implemented
in Phase 2 without a hard NetworkX dependency (a tiny adjacency structure is
enough for scheduling and critical-path queries); a ``to_networkx`` helper is
provided for modules that want the full library.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from qco.ir.intermediate_representation import Gate, IntermediateRepresentation

# Commutation table: for each gate, the Pauli it commutes with on each of its
# qubits (positionally), or None. E.g. cx commutes with Z on the control and X
# on the target. Two gates commute if, on every qubit they share, both commute
# with the same Pauli P: each is then block-diagonal in P's eigenbasis there,
# and the blocks act on disjoint remaining qubits. Sufficient, not necessary.
_PAULI_ACTION: dict[str, tuple[str | None, ...]] = {
    "id": ("I",),
    "z": ("Z",), "s": ("Z",), "sdg": ("Z",), "t": ("Z",), "tdg": ("Z",), "rz": ("Z",), "p": ("Z",),
    "x": ("X",), "rx": ("X",),
    "y": ("Y",), "ry": ("Y",),
    "h": (None,),
    "cz": ("Z", "Z"), "cp": ("Z", "Z"), "crz": ("Z", "Z"), "rzz": ("Z", "Z"),
    "cs": ("Z", "Z"), "csdg": ("Z", "Z"),
    "cx": ("Z", "X"), "crx": ("Z", "X"), "rxx": ("X", "X"),
    "cy": ("Z", "Y"), "cry": ("Z", "Y"), "ryy": ("Y", "Y"),
    "ch": ("Z", None), "swap": (None, None),
    "ccx": ("Z", "Z", "X"), "cswap": ("Z", None, None),
}


def commutes(a: Gate, b: Gate) -> bool:
    """True if ``a`` and ``b`` provably commute (False means 'not sure')."""
    if a.name == b.name and a.qubits == b.qubits:
        return True  # same gate type on same wires; params may differ (rotations commute)
    pa = dict(zip(a.qubits, _PAULI_ACTION.get(a.name, (None,) * len(a.qubits))))
    pb = dict(zip(b.qubits, _PAULI_ACTION.get(b.name, (None,) * len(b.qubits))))
    for q in pa.keys() & pb.keys():
        x, y = pa[q], pb[q]
        if not (x == "I" or y == "I" or (x is not None and x == y)):
            return False
    return True


@dataclass(slots=True)
class GateDAG:
    num_gates: int
    succ: dict[int, set[int]] = field(default_factory=dict)
    pred: dict[int, set[int]] = field(default_factory=dict)

    def add_edge(self, u: int, v: int) -> None:
        if u == v:
            return
        self.succ.setdefault(u, set()).add(v)
        self.pred.setdefault(v, set()).add(u)

    def roots(self) -> list[int]:
        return [i for i in range(self.num_gates) if not self.pred.get(i)]

    def topological_order(self) -> list[int]:
        indeg = {i: len(self.pred.get(i, ())) for i in range(self.num_gates)}
        queue = deque(i for i, d in indeg.items() if d == 0)
        order: list[int] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in sorted(self.succ.get(u, ())):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(order) != self.num_gates:
            raise ValueError("cycle detected in gate DAG")
        return order

    def critical_path_length(self) -> int:
        longest = {i: 1 for i in range(self.num_gates)}
        for u in self.topological_order():
            for v in self.succ.get(u, ()):
                longest[v] = max(longest[v], longest[u] + 1)
        return max(longest.values(), default=0)

    def to_networkx(self):  # pragma: no cover - optional dependency
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(range(self.num_gates))
        for u, vs in self.succ.items():
            g.add_edges_from((u, v) for v in vs)
        return g


def build_dag(ir: IntermediateRepresentation) -> GateDAG:
    """Build the gate-dependency DAG from an IR (last-writer wins per qubit)."""
    dag = GateDAG(num_gates=len(ir.gates))
    last_on_qubit: dict[int, int] = {}
    for idx, gate in enumerate(ir.gates):
        for q in gate.qubits:
            prev = last_on_qubit.get(q)
            if prev is not None:
                dag.add_edge(prev, idx)
        for q in gate.qubits:
            last_on_qubit[q] = idx
    return dag
