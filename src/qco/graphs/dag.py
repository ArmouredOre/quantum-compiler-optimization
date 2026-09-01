"""Stage 2 — gate-dependency DAG.

Nodes are gate indices into ``IntermediateRepresentation.gates``; an edge
``i -> j`` means gate ``j`` uses a qubit last written by gate ``i``. Implemented
in Phase 2 without a hard NetworkX dependency (a tiny adjacency structure is
enough for scheduling and critical-path queries); a ``to_networkx`` helper is
provided for modules that want the full library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qco.ir.intermediate_representation import IntermediateRepresentation


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
        queue = [i for i, d in indeg.items() if d == 0]
        order: list[int] = []
        while queue:
            u = queue.pop(0)
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
