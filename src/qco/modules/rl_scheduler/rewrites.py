"""Module A — pure IR rewrites for the RL environment's action set.

Each function takes an :class:`~qco.ir.intermediate_representation.IntermediateRepresentation`
and an already-legal ``(i, j)`` gate-index pair and returns a **new** IR with the
rewrite applied. Kept free of any Gymnasium/environment state so they can be
unit tested directly and reused by other modules (e.g. a future search-based
scheduler) without instantiating ``SchedulerEnv``.

Legality is the caller's responsibility — ``SchedulerEnv.legal_actions()``
already only offers safe ``(i, j)`` pairs, and ``apply_action`` re-validates
defensively (raises ``IllegalActionError``) so a misused action never silently
corrupts the circuit.

Sprint 2 fix
------------
Sprint 1's ``commute`` and ``merge_rotation`` checked *qubit disjointness* /
*gate equality* but never that ``i`` and ``j`` were **adjacent** in the gate
list. Called directly with a non-adjacent pair they silently produced an
inequivalent circuit (``legal_actions()`` and ``spaces.decode_action`` only ever
emit adjacent pairs, which is why the Sprint 1 smoke test never hit it). Both
now raise ``IllegalActionError`` unless ``|i - j| == 1``.
"""

from __future__ import annotations

from qco.ir.intermediate_representation import PARAMETRIC, Gate, IntermediateRepresentation
from qco.modules.rl_scheduler.commutation import commutes, validate_mode
from qco.modules.rl_scheduler.gate_algebra import NOT_REDUCIBLE, reduce_pair, same_operands


class IllegalActionError(ValueError):
    """Raised when a rewrite is applied to a gate pair that doesn't support it."""


def _swap_adjacent(gates: list[Gate], i: int, j: int) -> list[Gate]:
    """Return a copy of ``gates`` with positions ``i`` and ``j`` swapped."""
    new_gates = list(gates)
    new_gates[i], new_gates[j] = new_gates[j], new_gates[i]
    return new_gates


def can_commute(a: Gate, b: Gate, mode: str = "disjoint") -> bool:
    """Whether ``a`` and ``b`` may be swapped under the given commutation ``mode``.

    The default stays ``"disjoint"`` (Sprint 1 semantics) so existing callers are
    unaffected; pass ``"full"`` to also allow provably-commuting gates that share
    qubits (see :mod:`qco.modules.rl_scheduler.commutation`).
    """
    return commutes(a, b, validate_mode(mode))


def can_merge_rotation(a: Gate, b: Gate) -> bool:
    """Same rotation gate, same qubit(s), back-to-back -> mergeable via angle sum."""
    return a.name == b.name and a.qubits == b.qubits and a.name in PARAMETRIC


def _check_adjacent(op: str, i: int, j: int, n: int) -> None:
    if not (0 <= i < n and 0 <= j < n):
        raise IllegalActionError(f"{op}({i}, {j}): index out of range for {n} gates")
    if abs(i - j) != 1:
        raise IllegalActionError(
            f"{op}({i}, {j}): gates must be adjacent in the gate list (|i - j| == 1); "
            "a non-adjacent rewrite can reorder past gates it does not commute with"
        )


def dag_edges(gates: list[Gate]) -> list[tuple[int, int]]:
    """``(i, j)`` with ``i < j`` such that gates ``i`` and ``j`` are *consecutive on a
    shared qubit wire* (no gate between them touches that qubit): the edges of the
    gate-dependency DAG, each listed once."""
    last: dict[int, int] = {}
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for j, g in enumerate(gates):
        for q in g.qubits:
            i = last.get(q)
            if i is not None and (i, j) not in seen:
                seen.add((i, j))
                edges.append((i, j))
            last[q] = j
    return edges


def _overlaps(a: Gate, b: Gate) -> bool:
    return not set(a.qubits).isdisjoint(b.qubits)


def flip_strategy(gates: list[Gate], i: int, j: int, mode: str = "full") -> str | None:
    """How to reorder so gate ``j`` runs before gate ``i`` (``i < j``), or ``None``.

    ``"hoist"``: move ``j`` to just before ``i``; needs ``j`` to commute with ``i``
    and with every gate in between that touches its qubits.
    ``"sink"``: move ``i`` to just after ``j``; needs ``i`` to commute with every
    gate in between that touches its qubits, and with ``j``.

    Either is a sequence of adjacent swaps of commuting (or disjoint) gates, so the
    circuit's unitary is unchanged by construction.
    """
    u, v = gates[i], gates[j]
    if not can_commute(u, v, mode):
        return None
    if all(can_commute(gates[k], v, mode) for k in range(i + 1, j) if _overlaps(gates[k], v)):
        return "hoist"
    if all(can_commute(u, gates[k], mode) for k in range(i + 1, j) if _overlaps(u, gates[k])):
        return "sink"
    return None


def flippable_edges(gates: list[Gate], mode: str = "full") -> list[tuple[int, int]]:
    """DAG edges whose two gates can be legally reordered (and are not identical)."""
    return [
        (i, j)
        for i, j in dag_edges(gates)
        if gates[i] != gates[j] and flip_strategy(gates, i, j, mode) is not None
    ]


def _flip(ir: IntermediateRepresentation, i: int, j: int, mode: str) -> IntermediateRepresentation:
    gates = ir.gates
    n = len(gates)
    lo, hi = (i, j) if i < j else (j, i)
    if not (0 <= lo < hi < n):
        raise IllegalActionError(f"commute({i}, {j}): index out of range for {n} gates")
    if (lo, hi) not in set(dag_edges(gates)):
        raise IllegalActionError(
            f"commute({i}, {j}): gates must be consecutive on a shared qubit (a dependency edge)"
        )
    strategy = flip_strategy(gates, lo, hi, mode)
    if strategy is None:
        a, b = gates[lo], gates[hi]
        raise IllegalActionError(
            f"commute({i}, {j}): {a.name}{a.qubits} and {b.name}{b.qubits} cannot be reordered under mode {mode!r}"
        )
    u, v = gates[lo], gates[hi]
    if strategy == "hoist":
        new_gates = gates[:lo] + [v, u] + gates[lo + 1 : hi] + gates[hi + 1 :]
    else:
        new_gates = gates[:lo] + gates[lo + 1 : hi] + [v, u] + gates[hi + 1 :]
    out = ir.copy()
    out.gates = new_gates
    return out


def commute(ir: IntermediateRepresentation, i: int, j: int, mode: str = "disjoint") -> IntermediateRepresentation:
    """Reorder two commuting gates at positions ``i`` and ``j``.

    ``mode="disjoint"`` (Sprint 1): swap two **adjacent** gates on disjoint qubits.
    ``mode="full"``: flip a dependency edge — ``i`` and ``j`` must be consecutive on
    a shared qubit and commute; the reorder is realized by hoisting/sinking one gate
    past gates it commutes with (see :func:`flip_strategy`). Unlike a disjoint swap
    this changes the dependency DAG, so it can change circuit depth.

    Pure reordering either way: gate count and the multiset of gates are unchanged,
    and the circuit's unitary is identical. Used for ``ActionKind.COMMUTE``.
    """
    mode = validate_mode(mode)
    gates = ir.gates
    if mode == "full":
        return _flip(ir, i, j, mode)
    _check_adjacent("commute", i, j, len(gates))
    a, b = gates[i], gates[j]
    if not can_commute(a, b, mode):
        raise IllegalActionError(
            f"commute({i}, {j}): {a.name}{a.qubits} and {b.name}{b.qubits} do not commute under mode {mode!r}"
        )
    out = ir.copy()
    out.gates = _swap_adjacent(gates, i, j)
    return out


def move(ir: IntermediateRepresentation, i: int, earlier: bool, mode: str = "disjoint") -> IntermediateRepresentation:
    """Move the gate at index ``i`` one slot earlier or later via a commuting swap.

    ``MOVE_EARLIER`` swaps ``(i-1, i)``; ``MOVE_LATER`` swaps ``(i, i+1)``. Both
    are no-ops (return an unchanged copy) at the schedule boundary or when the
    neighbor does not commute, rather than raising, since "move as far as
    possible" is a reasonable degenerate case for an RL agent to encounter
    often and shouldn't crash an episode.
    """
    gates = ir.gates
    if earlier:
        if i <= 0 or i >= len(gates):
            return ir.copy()
        lo, hi = i - 1, i
    else:
        if i < 0 or i >= len(gates) - 1:
            return ir.copy()
        lo, hi = i, i + 1
    if not can_commute(gates[lo], gates[hi], mode):
        return ir.copy()
    out = ir.copy()
    out.gates = _swap_adjacent(gates, lo, hi)
    return out


def merge_rotation(ir: IntermediateRepresentation, i: int, j: int) -> IntermediateRepresentation:
    """Merge two **adjacent** equal-axis rotations at ``i``, ``j`` by summing angles.

    ``Rz(a) . Rz(b) = Rz(a+b)`` (and likewise for ``rx``/``ry``/``p``): replaces
    the pair with a single gate carrying the summed parameter, reducing gate
    count by one. If the summed angle is the gate's identity period (see
    :mod:`qco.modules.rl_scheduler.gate_algebra`) the gate is dropped entirely,
    reducing gate count by two.
    """
    gates = ir.gates
    _check_adjacent("merge_rotation", i, j, len(gates))
    a, b = gates[i], gates[j]
    if not can_merge_rotation(a, b) or not same_operands(a, b):
        raise IllegalActionError(
            f"merge_rotation({i}, {j}): gates {a.name}/{b.name} on {a.qubits}/{b.qubits} not mergeable"
        )
    merged = reduce_pair(a, b)
    if merged is NOT_REDUCIBLE:  # pragma: no cover - guarded by can_merge_rotation above
        raise IllegalActionError(f"merge_rotation({i}, {j}): gates did not reduce")

    lo, hi = (i, j) if i < j else (j, i)
    out = ir.copy()
    new_gates = list(gates)
    del new_gates[hi]
    del new_gates[lo]
    if merged is not None:
        new_gates.insert(lo, merged)
    out.gates = new_gates
    return out


def apply_action(
    ir: IntermediateRepresentation,
    kind: str,
    i: int,
    j: int,
    commutation: str = "disjoint",
) -> IntermediateRepresentation:
    """Dispatch helper used by ``SchedulerEnv.step``; ``kind`` is an ``ActionKind`` name.

    Centralizing the dispatch here (rather than in ``environment.py``) keeps
    ``step`` itself a thin bookkeeping wrapper and makes every rewrite
    independently unit-testable.
    """
    if kind == "NOOP":
        return ir.copy()
    if kind == "COMMUTE":
        return commute(ir, i, j, commutation)
    if kind == "MOVE_EARLIER":
        return move(ir, i, earlier=True, mode=commutation)
    if kind == "MOVE_LATER":
        return move(ir, i, earlier=False, mode=commutation)
    if kind == "MERGE_ROTATION":
        return merge_rotation(ir, i, j)
    raise IllegalActionError(f"unknown action kind {kind!r}")
