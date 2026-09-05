"""Module B — node/edge feature encoding and rule-based cancellation candidates.

Sprint 1 freezes the node-feature layout, builds a PyTorch Geometric ``Data``
object from the gate DAG, and expands the deterministic candidate generator
used to bootstrap training labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qco.graphs.dag import build_dag
from qco.ir.intermediate_representation import (
    ARITY,
    PARAMETRIC,
    SELF_INVERSE,
    Gate,
    IntermediateRepresentation,
)

# ---------------------------------------------------------------------------
# Frozen node-feature layout (do not reorder without bumping FEATURE_DIM and
# regenerating any dumped datasets).
# ---------------------------------------------------------------------------

GATE_VOCAB: tuple[str, ...] = tuple(sorted(ARITY))
MAX_ARITY = 3  # widest supported gate is ``ccx``

# one-hot type | arity | normalized qubit positions | parametric flag | depth
IDX_ONEHOT = 0
IDX_ARITY = len(GATE_VOCAB)
IDX_QUBITS = IDX_ARITY + 1
IDX_PARAMETRIC = IDX_QUBITS + MAX_ARITY
IDX_DEPTH = IDX_PARAMETRIC + 1
FEATURE_DIM = IDX_DEPTH + 1

FEATURE_LAYOUT: tuple[str, ...] = (
    *(f"type_{name}" for name in GATE_VOCAB),
    "arity",
    *(f"qubit_{i}" for i in range(MAX_ARITY)),
    "parametric",
    "depth_position",
)

# Directed DAG edge: shared-qubit count, index span, same-name, same-support.
EDGE_FEATURE_DIM = 4

_TWO_PI = 2.0 * math.pi
_ANGLE_TOL = 1e-8

# Diagonal / Z-rotation family and X/Y families used by the commute checker.
_DIAGONAL = frozenset({"z", "s", "sdg", "t", "tdg", "rz", "p", "cz", "cp", "crz", "rzz"})
_X_FAMILY = frozenset({"x", "rx"})
_Y_FAMILY = frozenset({"y", "ry"})
_CX_LIKE = frozenset({"cx"})
_CZ_LIKE = frozenset({"cz"})
_CCX_LIKE = frozenset({"ccx"})


@dataclass(frozen=True, slots=True)
class CancellationCandidate:
    """A proposed cancellation of gates (indices into the IR).

    ``a`` / ``b`` are the first and last gate in the pattern.  Longer chains
    also fill ``indices``; when it is empty, the pair is ``(a, b)``.
    """

    a: int
    b: int
    kind: str  # "inverse_pair" | "rotation_merge" | "identity_chain"
    score: float = 0.0  # GNN confidence in [0, 1]
    indices: tuple[int, ...] = ()

    @property
    def gate_indices(self) -> tuple[int, ...]:
        return self.indices if self.indices else (self.a, self.b)


def gate_node_features(
    gate: Gate,
    num_qubits: int,
    *,
    index: int = 0,
    num_gates: int = 1,
    layer: int | None = None,
    circuit_depth: int | None = None,
) -> list[float]:
    """Encode one gate into the frozen Sprint-1 feature vector.

    Layout: one-hot type, arity, normalized qubit positions (padded to
    ``MAX_ARITY`` with -1), parametric flag, depth position in ``[0, 1]``.
    """
    onehot = [1.0 if gate.name == name else 0.0 for name in GATE_VOCAB]
    denom = max(num_qubits - 1, 1)
    qpos = [-1.0] * MAX_ARITY
    for i, q in enumerate(gate.qubits[:MAX_ARITY]):
        qpos[i] = q / denom
    if layer is not None and circuit_depth is not None:
        depth_pos = (layer - 1) / max(circuit_depth - 1, 1)
    else:
        depth_pos = index / max(num_gates - 1, 1)
    return [
        *onehot,
        float(len(gate.qubits)),
        *qpos,
        float(bool(gate.params) or gate.name in PARAMETRIC),
        float(depth_pos),
    ]


def _asap_layers(ir: IntermediateRepresentation) -> tuple[list[int], int]:
    """ASAP layer index (1-based) per gate, plus circuit depth."""
    frontier = [0] * ir.num_qubits
    layers: list[int] = []
    for gate in ir.gates:
        layer = max((frontier[q] for q in gate.qubits), default=0) + 1
        layers.append(layer)
        for q in gate.qubits:
            frontier[q] = layer
    return layers, max(frontier, default=0)


def _edge_features(ir: IntermediateRepresentation, u: int, v: int) -> list[float]:
    gu, gv = ir.gates[u], ir.gates[v]
    shared = float(len(set(gu.qubits) & set(gv.qubits)))
    span = (v - u) / max(len(ir.gates) - 1, 1)
    same_name = float(gu.name == gv.name)
    same_support = float(set(gu.qubits) == set(gv.qubits))
    return [shared, span, same_name, same_support]


def build_pyg_data(ir: IntermediateRepresentation):
    """Build a ``torch_geometric.data.Data`` object for ``ir``.

    Requires the ``gnn`` extra (``torch`` + ``torch-geometric``).
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "build_pyg_data requires the `gnn` extra: pip install -e '.[gnn]'"
        ) from exc

    layers, depth = _asap_layers(ir)
    n = len(ir.gates)
    rows = [
        gate_node_features(
            gate,
            ir.num_qubits,
            index=i,
            num_gates=n,
            layer=layers[i] if layers else 1,
            circuit_depth=depth,
        )
        for i, gate in enumerate(ir.gates)
    ]
    x = torch.tensor(rows, dtype=torch.float32).reshape(n, FEATURE_DIM)

    dag = build_dag(ir)
    src: list[int] = []
    dst: list[int] = []
    efeat: list[list[float]] = []
    for u, vs in dag.succ.items():
        for v in sorted(vs):
            src.append(u)
            dst.append(v)
            efeat.append(_edge_features(ir, u, v))

    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, len(src))
    edge_attr = torch.tensor(efeat, dtype=torch.float32).reshape(len(src), EDGE_FEATURE_DIM)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_qubits = ir.num_qubits
    data.circuit_name = ir.name
    return data


# ---------------------------------------------------------------------------
# Inverse / commute / chain helpers
# ---------------------------------------------------------------------------

def _angle_close(a: float, b: float, tol: float = _ANGLE_TOL) -> bool:
    delta = abs(a - b) % _TWO_PI
    return min(delta, _TWO_PI - delta) < tol


def _params_close(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if len(a) != len(b):
        return False
    return all(_angle_close(x, y) for x, y in zip(a, b))


def _angles_sum_identity(params_list: list[tuple[float, ...]]) -> bool:
    if not params_list:
        return False
    width = len(params_list[0])
    if any(len(p) != width for p in params_list):
        return False
    totals = [sum(p[i] for p in params_list) for i in range(width)]
    return all(_angle_close(t, 0.0) for t in totals)


def are_inverses(g: Gate, h: Gate) -> bool:
    """True iff ``g`` then ``h`` (same qubits) multiply to identity."""
    if g.qubits != h.qubits:
        return False
    try:
        inv = g.inverse()
    except NotImplementedError:
        return False
    return inv.name == h.name and _params_close(inv.params, h.params)


def _shares_qubits(g: Gate, h: Gate) -> bool:
    return not set(g.qubits).isdisjoint(h.qubits)


def _wire_clear(ir: IntermediateRepresentation, i: int, j: int) -> bool:
    """No gate between ``i`` and ``j`` touches the qubits of the pair."""
    involved = set(ir.gates[i].qubits) | set(ir.gates[j].qubits)
    for k in range(i + 1, j):
        if set(ir.gates[k].qubits) & involved:
            return False
    return True


def _cx_commutes_with(cx: Gate, other: Gate) -> bool:
    ctrl, tgt = cx.qubits[0], cx.qubits[1]
    if len(other.qubits) == 1:
        q = other.qubits[0]
        if q == tgt and other.name in _X_FAMILY:
            return True
        if q == ctrl and other.name in _DIAGONAL:
            return True
        return False
    if other.name in _CX_LIKE and other.qubits == cx.qubits:
        return True
    return False


def _cz_commutes_with(cz: Gate, other: Gate) -> bool:
    support = set(cz.qubits)
    if len(other.qubits) == 1:
        return other.qubits[0] in support and other.name in _DIAGONAL
    if other.name in _CZ_LIKE and set(other.qubits) == support:
        return True
    return False


def _ccx_commutes_with(ccx: Gate, other: Gate) -> bool:
    c0, c1, tgt = ccx.qubits
    if len(other.qubits) == 1:
        q = other.qubits[0]
        if q == tgt and other.name in _X_FAMILY:
            return True
        if q in (c0, c1) and other.name in _DIAGONAL:
            return True
    return False


def gates_commute(a: Gate, b: Gate) -> bool:
    """Conservative pairwise commutation (false means 'not sure / no')."""
    if not _shares_qubits(a, b):
        return True
    if a.qubits == b.qubits:
        if a.name in _DIAGONAL and b.name in _DIAGONAL:
            return True
        if a.name in _X_FAMILY and b.name in _X_FAMILY:
            return True
        if a.name in _Y_FAMILY and b.name in _Y_FAMILY:
            return True
        if a.name == b.name:
            return True
        return False
    for left, right in ((a, b), (b, a)):
        if left.name in _CX_LIKE and len(left.qubits) == 2 and _cx_commutes_with(left, right):
            return True
        if left.name in _CZ_LIKE and len(left.qubits) == 2 and _cz_commutes_with(left, right):
            return True
        if left.name in _CCX_LIKE and len(left.qubits) == 3 and _ccx_commutes_with(left, right):
            return True
    return False


def _can_commute_together(ir: IntermediateRepresentation, i: int, j: int) -> bool:
    """Slide the inverse pair together past intervening gates that commute."""
    gi = ir.gates[i]
    involved = set(gi.qubits) | set(ir.gates[j].qubits)
    for k in range(i + 1, j):
        gk = ir.gates[k]
        if not (set(gk.qubits) & involved):
            continue
        if not gates_commute(gi, gk):
            return False
    return True


def _add_candidate(
    out: list[CancellationCandidate],
    seen: set[tuple],
    a: int,
    b: int,
    kind: str,
    indices: tuple[int, ...] = (),
) -> None:
    key = (kind, indices if indices else (a, b))
    if key in seen:
        return
    seen.add(key)
    out.append(CancellationCandidate(a, b, kind, 1.0, indices))


def _identity_period(name: str) -> int | None:
    """Smallest n > 1 such that ``g**n == I`` for a non-self-inverse named gate."""
    if name in {"s", "sdg"}:
        return 4
    if name in {"t", "tdg"}:
        return 8
    return None


def _runs_on_support(ir: IntermediateRepresentation) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, gate in enumerate(ir.gates):
        groups.setdefault(gate.qubits, []).append(i)
    return groups


def _split_named_runs(
    ir: IntermediateRepresentation, idxs: list[int]
) -> list[list[int]]:
    """Maximal wire-adjacent runs of the same gate name on one support."""
    if not idxs:
        return []
    runs: list[list[int]] = [[idxs[0]]]
    for prev, cur in zip(idxs, idxs[1:]):
        same = ir.gates[prev].name == ir.gates[cur].name
        if same and _wire_clear(ir, prev, cur):
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return runs


def rule_based_candidates(ir: IntermediateRepresentation) -> list[CancellationCandidate]:
    """Deterministic candidate generator (provisional training labels).

    Finds:
    * adjacent inverse pairs (nothing on the shared wires between them)
    * inverse pairs that become adjacent after commuting through blockers
    * rotation chains on the same support that merge into one rotation
    * identity chains (self-inverse repeats, S^4, T^8, rotations summing to 0)
    """
    out: list[CancellationCandidate] = []
    seen: set[tuple] = set()
    n = len(ir.gates)

    for i in range(n):
        for j in range(i + 1, n):
            if not are_inverses(ir.gates[i], ir.gates[j]):
                continue
            if _wire_clear(ir, i, j):
                _add_candidate(out, seen, i, j, "inverse_pair")
            elif _can_commute_together(ir, i, j):
                _add_candidate(out, seen, i, j, "inverse_pair")

    for _qubits, idxs in _runs_on_support(ir).items():
        for run in _split_named_runs(ir, idxs):
            if len(run) < 2:
                continue
            name = ir.gates[run[0]].name
            params = [ir.gates[k].params for k in run]
            chain = tuple(run)

            if name in PARAMETRIC:
                if _angles_sum_identity(params):
                    _add_candidate(out, seen, run[0], run[-1], "identity_chain", chain)
                else:
                    _add_candidate(out, seen, run[0], run[-1], "rotation_merge", chain)
                continue

            if name in SELF_INVERSE and len(run) >= 4 and len(run) % 2 == 0:
                _add_candidate(out, seen, run[0], run[-1], "identity_chain", chain)

            period = _identity_period(name)
            if period is not None and len(run) >= period and len(run) % period == 0:
                _add_candidate(out, seen, run[0], run[-1], "identity_chain", chain)

    return out
