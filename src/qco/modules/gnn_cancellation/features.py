"""Module B — node/edge feature encoding for the circuit graph.

The GNN consumes the gate-dependency DAG; each node is one gate. Phase 2 fixes
the feature layout (so training-data generation can start) and ships a working
``rule_based_candidates`` baseline the GNN must beat.
"""

from __future__ import annotations

from dataclasses import dataclass

from qco.graphs.dag import build_dag
from qco.ir.intermediate_representation import ARITY, SELF_INVERSE, Gate, IntermediateRepresentation

_GATE_VOCAB = sorted(ARITY)


@dataclass(frozen=True, slots=True)
class CancellationCandidate:
    """A proposed cancellation of gates ``a`` and ``b`` (indices into the IR)."""

    a: int
    b: int
    kind: str            # "inverse_pair" | "rotation_merge" | "identity_chain"
    score: float = 0.0    # GNN confidence in [0, 1]


def gate_node_features(gate: Gate, num_qubits: int) -> list[float]:
    """One-hot gate type + arity + normalized qubit positions + parametric flag."""
    onehot = [1.0 if gate.name == g else 0.0 for g in _GATE_VOCAB]
    q0 = gate.qubits[0] / max(num_qubits - 1, 1)
    q1 = (gate.qubits[1] / max(num_qubits - 1, 1)) if gate.is_two_qubit else -1.0
    return [*onehot, float(len(gate.qubits)), q0, q1, float(bool(gate.params))]


def build_pyg_data(ir: IntermediateRepresentation):  # pragma: no cover - Phase 3
    """Build a ``torch_geometric.data.Data`` object (Phase 3, Prisha)."""
    raise NotImplementedError("needs the `gnn` extra (torch-geometric)")


def rule_based_candidates(ir: IntermediateRepresentation) -> list[CancellationCandidate]:
    """Deterministic baseline: adjacent inverse pairs on identical qubits.

    Used to (a) bootstrap labeled training data and (b) act as the fallback
    inside the pipeline until the GNN is trained.
    """
    dag = build_dag(ir)
    out: list[CancellationCandidate] = []
    for u in range(len(ir.gates)):
        for v in sorted(dag.succ.get(u, ())):
            g, h = ir.gates[u], ir.gates[v]
            if g.qubits != h.qubits:
                continue
            if g.name == h.name and g.name in SELF_INVERSE and not g.params:
                out.append(CancellationCandidate(u, v, "inverse_pair", 1.0))
            elif g.name == h.name and g.params and h.params:
                out.append(CancellationCandidate(u, v, "rotation_merge", 1.0))
    return out
