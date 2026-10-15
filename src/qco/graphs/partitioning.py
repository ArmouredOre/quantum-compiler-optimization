"""Stage 3 — partitioning into blocks for the optimization modules.

Blocks are capped at ``max_block_gates`` gates and ``max_block_qubits`` qubits,
so each one stays small enough for exact re-synthesis and dense verification
(novelty N1/N2). Backends:

* ``"auto"`` / ``"greedy"``: dependency-free sweep that merges the open blocks a
  gate touches while both caps hold (``_greedy_blocks``).
* ``"kahypar"`` / ``"metis"``: cut the qubit hypergraph into
  ``ceil(n / max_block_qubits)`` parts first (KaHyPar needs a preset ``.ini``
  via ``$KAHYPAR_INI``; METIS via ``pymetis``), then the same sweep, never
  merging across parts. On the benchmark suite this cuts *more* wires than
  greedy (random circuits have no qubit locality), so it is opt-in.
* ``"fallback"``: connected components of the DAG, then size-capped chunks.
  No width cap - lowest wire cut, but blocks span the whole circuit.

Contract (all backends): every gate is in exactly one block, and concatenating
the blocks in list order keeps every qubit's gate order - so the stitched
circuit equals the input and per-block rewrites compose (N3 block crossover).
schedule_blocks / stitch_blocks enforce that contract for any block list: they validate exactly-once coverage and reorder blocks by dependency.
"""

from __future__ import annotations

import heapq
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from collections import Counter
from dataclasses import dataclass, field

from qco.graphs.dag import build_dag
from qco.graphs.hypergraph import QubitHypergraph, build_hypergraph
from qco.ir.intermediate_representation import IntermediateRepresentation


@dataclass(slots=True)
class Block:
    """A near-independent sub-circuit handed to the optimization modules."""

    index: int
    ir: IntermediateRepresentation
    source_gate_ids: list[int] = field(default_factory=list)
    # Qubits this block shares with at least one other block (its cut wires).
    boundary_qubits: frozenset[int] = frozenset()


def partition_circuit(
    ir: IntermediateRepresentation,
    max_block_gates: int = 64,
    backend: str = "auto",
    max_block_qubits: int = 8,
) -> list[Block]:
    """Split ``ir`` into blocks of at most ``max_block_gates`` gates and
    ``max_block_qubits`` qubits (a single wider gate gets its own block).

    ``backend``: ``"auto"`` / ``"greedy"``, ``"kahypar"``, ``"metis"`` or
    ``"fallback"`` (see module docstring). A missing solver raises ImportError.
    """
    if backend == "fallback":
        return _fallback_partition(ir, max_block_gates)
    if backend not in ("auto", "greedy", "kahypar", "metis"):
        raise ValueError(f"unknown partition backend {backend!r}")
    parts = None
    k = -(-ir.num_qubits // max_block_qubits)
    if backend == "kahypar":
        ini = os.environ.get("KAHYPAR_INI")
        if not ini:
            raise ImportError("KaHyPar needs $KAHYPAR_INI pointing to a preset .ini")
        parts = _kahypar_parts(build_hypergraph(ir), k, ini) if k > 1 else None
    elif backend == "metis":
        parts = _metis_parts(build_hypergraph(ir), k) if k > 1 else None
    return _greedy_blocks(ir, max_block_gates, max_block_qubits, parts)


def cut_size(ir: IntermediateRepresentation, blocks: list[Block]) -> int:
    """Number of gate-DAG edges (qubit wires) that cross between blocks."""
    owner = {g: b.index for b in blocks for g in b.source_gate_ids}
    dag = build_dag(ir)
    return sum(owner[u] != owner[v] for u, vs in dag.succ.items() for v in vs)


def _kahypar_parts(hg: QubitHypergraph, k: int, ini: str) -> list[int]:
    import kahypar

    with tempfile.NamedTemporaryFile("w", suffix=".hgr", delete=False) as fh:
        fh.write(hg.to_kahypar_hmetis())
    try:
        h = kahypar.createHypergraphFromFile(fh.name, k)
    finally:
        os.unlink(fh.name)
    ctx = kahypar.Context()
    ctx.loadINIconfiguration(ini)
    ctx.setK(k)
    ctx.setEpsilon(0.03)
    ctx.setSeed(0)
    ctx.suppressOutput(True)
    kahypar.partition(h, ctx)
    return [h.blockID(q) for q in range(hg.num_qubits)]


def _metis_parts(hg: QubitHypergraph, k: int) -> list[int]:
    """METIS on the clique expansion of the hypergraph (edge weight = summed cost)."""
    import pymetis

    w: Counter[tuple[int, int]] = Counter()
    for qs, weight in hg.hyperedges:
        for a in qs:
            for b in qs:
                if a != b:
                    w[(a, b)] += weight
    adj = [[] for _ in range(hg.num_qubits)]
    for a, b in sorted(w):
        adj[a].append(b)
    xadj = [0]
    for nbrs in adj:
        xadj.append(xadj[-1] + len(nbrs))
    adjncy = [b for nbrs in adj for b in nbrs]
    eweights = [w[(a, b)] for a, nbrs in enumerate(adj) for b in nbrs]
    _, parts = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy, eweights=eweights)
    return list(parts)


def _greedy_blocks(
    ir: IntermediateRepresentation,
    max_block_gates: int,
    max_block_qubits: int,
    parts: list[int] | None = None,
) -> list[Block]:
    """Sweep gates in order; every qubit is in at most one open block.

    A gate merges the open blocks on its qubits (they are qubit-disjoint, so
    any interleaving is valid) if the result fits both caps and, with
    ``parts``, stays in one part. Otherwise those blocks are emitted and the
    gate opens a new one. A block is emitted before any later block touching
    its qubits opens, so emission order is a valid concatenation order.
    """
    owner: dict[int, tuple[list[int], set[int]]] = {}  # qubit -> open (gate ids, qubits)
    emitted: list[list[int]] = []
    for i, g in enumerate(ir.gates):
        touched = list({id(owner[q]): owner[q] for q in g.qubits if q in owner}.values())
        qubits = set(g.qubits).union(*(qs for _, qs in touched))
        fits = (
            len(qubits) <= max_block_qubits
            and sum(len(ids) for ids, _ in touched) < max_block_gates
            and (parts is None or len(touched) <= 1 or len({parts[q] for q in qubits}) == 1)
        )
        if fits:
            new = (sorted(x for ids, _ in touched for x in ids) + [i], qubits)
        else:
            emitted.extend(ids for ids, _ in touched)
            new = ([i], set(g.qubits))
        for _, qs in touched:
            for q in qs:
                del owner[q]
        for q in new[1]:
            owner[q] = new
    emitted.extend({id(b): b[0] for b in owner.values()}.values())
    return _make_blocks(ir, emitted)


def _boundaries(ir: IntermediateRepresentation, groups: list[list[int]]) -> list[frozenset[int]]:
    """Per group: qubits it shares with at least one other group."""
    touched = [{q for g in grp for q in ir.gates[g].qubits} for grp in groups]
    shared = Counter(q for t in touched for q in t)
    return [frozenset(q for q in t if shared[q] > 1) for t in touched]


def _make_blocks(ir: IntermediateRepresentation, groups: list[list[int]]) -> list[Block]:
    bounds = _boundaries(ir, groups)
    return [
        Block(
            index=i,
            ir=IntermediateRepresentation(
                num_qubits=ir.num_qubits,
                gates=[ir.gates[g] for g in grp],
                name=f"{ir.name}#block{i}",
            ),
            source_gate_ids=grp,
            boundary_qubits=bounds[i],
        )
        for i, grp in enumerate(groups)
    ]


def _schedule_order(ir: IntermediateRepresentation, blocks: Sequence[Block]) -> list[int]:
    """Positions of ``blocks`` in a valid dependency order.

    Validates exactly-once coverage, then runs Kahn's algorithm on the block
    graph (edge A -> B if a gate in B depends on a gate in A). Ties go to the
    smallest input position, so a list that is already in a valid order comes
    back unchanged. Raises ValueError on bad coverage or a cycle (a cycle means
    some block is non-convex in the gate DAG).
    """
    n = len(ir.gates)
    owner = [-1] * n
    for pos, b in enumerate(blocks):
        ids = b.source_gate_ids
        if not ids:
            raise ValueError(f"block {pos} is empty")
        if any(x >= y for x, y in zip(ids, ids[1:])):
            raise ValueError(f"block {pos}: source_gate_ids must be strictly increasing")
        for g in ids:
            if not 0 <= g < n:
                raise ValueError(f"block {pos}: gate id {g} out of range")
            if owner[g] != -1:
                raise ValueError(f"gate {g} is in blocks {owner[g]} and {pos}")
            owner[g] = pos
    missing = [g for g in range(n) if owner[g] == -1]
    if missing:
        raise ValueError(f"{len(missing)} gate(s) in no block, e.g. {missing[:5]}")

    succ: list[set[int]] = [set() for _ in blocks]
    indeg = [0] * len(blocks)
    for u, vs in build_dag(ir).succ.items():
        for v in vs:
            a, b = owner[u], owner[v]
            if a != b and b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1

    ready = [p for p, d in enumerate(indeg) if d == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        p = heapq.heappop(ready)
        order.append(p)
        for s in succ[p]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready, s)
    if len(order) != len(blocks):
        stuck = sorted(set(range(len(blocks))) - set(order))
        raise ValueError(f"cyclic block dependencies (non-convex blocks): positions {stuck}")
    return order


def schedule_blocks(ir: IntermediateRepresentation, blocks: Sequence[Block]) -> list[Block]:
    """Blocks in dependency order, with ``index`` renumbered to the new position
    and ``boundary_qubits`` recomputed from ``source_gate_ids``."""
    scheduled = [blocks[p] for p in _schedule_order(ir, blocks)]
    bounds = _boundaries(ir, [b.source_gate_ids for b in scheduled])
    return [replace(b, index=i, boundary_qubits=bounds[i]) for i, b in enumerate(scheduled)]


def stitch_blocks(
    ir: IntermediateRepresentation,
    blocks: Sequence[Block],
    rewritten: Sequence[IntermediateRepresentation] | None = None,
) -> IntermediateRepresentation:
    """Concatenate blocks into one circuit in dependency order.

    ``rewritten[i]`` replaces ``blocks[i].ir`` (indexed by the position in
    ``blocks``, not by schedule order). A rewrite must stay on its block's
    qubits; that is what makes dependency order sufficient for correctness.
    """
    if rewritten is not None and len(rewritten) != len(blocks):
        raise ValueError(f"{len(rewritten)} rewrites for {len(blocks)} blocks")
    out = IntermediateRepresentation(num_qubits=ir.num_qubits, name=ir.name)
    for pos in _schedule_order(ir, blocks):
        piece = blocks[pos].ir if rewritten is None else rewritten[pos]
        if rewritten is not None:
            allowed = {q for g in blocks[pos].source_gate_ids for q in ir.gates[g].qubits}
            used = {q for g in piece.gates for q in g.qubits}
            if not used <= allowed:
                raise ValueError(f"rewrite of block {pos} touches qubits {sorted(used - allowed)} outside the block")
        out.gates.extend(piece.gates)
    return out


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

    chunks = [
        gate_ids[start : start + max_block_gates]
        for gate_ids in comps.values()
        for start in range(0, len(gate_ids), max_block_gates)
    ]
    return _make_blocks(ir, chunks)
