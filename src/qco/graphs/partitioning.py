"""Stage 3 — multilevel partitioning and block scheduling.

Phase 2 provides:
* the ``Block`` data contract handed to Stage 4, and
* a dependency-free fallback partitioner (connected components over the
  gate DAG, then greedy size-capped splitting) so the pipeline runs end-to-end
  before KaHyPar/METIS is wired in.

Phase 3 (Ajay M) replaces ``_fallback_partition`` with a real multilevel
hypergraph cut via ``qco.graphs.hypergraph.QubitHypergraph.to_kahypar_hmetis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qco.graphs.dag import build_dag
from qco.ir.intermediate_representation import IntermediateRepresentation


@dataclass(slots=True)
class Block:
    """A near-independent sub-circuit handed to the optimization modules."""

    index: int
    ir: IntermediateRepresentation
    source_gate_ids: list[int] = field(default_factory=list)
    boundary_qubits: frozenset[int] = frozenset()


def partition_circuit(
    ir: IntermediateRepresentation,
    max_block_gates: int = 64,
    backend: str = "auto",
) -> list[Block]:
    """Split ``ir`` into blocks of at most ``max_block_gates`` gates.

    ``backend``: ``"kahypar"``, ``"metis"``, or ``"auto"`` (try those, else the
    dependency-free fallback).
    """
    if backend in ("kahypar", "metis", "auto"):
        try:
            return _multilevel_partition(ir, max_block_gates, backend)
        except (ImportError, NotImplementedError):
            if backend != "auto":
                raise
    return _fallback_partition(ir, max_block_gates)


def _multilevel_partition(ir, max_block_gates, backend):  # pragma: no cover - Phase 3
    raise NotImplementedError(
        "multilevel hypergraph partitioning (KaHyPar/METIS) lands in Phase 3"
    )


def _fallback_partition(ir: IntermediateRepresentation, max_block_gates: int) -> list[Block]:
    """Weakly-connected components of the DAG, then size-capped greedy chunks."""
    dag = build_dag(ir)
    parent = list(range(len(ir.gates)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for u, vs in dag.succ.items():
        for v in vs:
            union(u, v)

    comps: dict[int, list[int]] = {}
    for i in range(len(ir.gates)):
        comps.setdefault(find(i), []).append(i)

    blocks: list[Block] = []
    for gate_ids in comps.values():
        for start in range(0, len(gate_ids), max_block_gates):
            chunk = gate_ids[start : start + max_block_gates]
            sub = IntermediateRepresentation(
                num_qubits=ir.num_qubits,
                gates=[ir.gates[g] for g in chunk],
                name=f"{ir.name}#block{len(blocks)}",
            )
            touched = {q for g in sub.gates for q in g.qubits}
            blocks.append(
                Block(
                    index=len(blocks),
                    ir=sub,
                    source_gate_ids=chunk,
                    boundary_qubits=frozenset(touched),
                )
            )
    return blocks
