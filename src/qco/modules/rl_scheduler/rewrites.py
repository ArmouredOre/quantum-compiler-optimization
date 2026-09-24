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
"""

from __future__ import annotations

from dataclasses import replace

from qco.ir.intermediate_representation import PARAMETRIC, Gate, IntermediateRepresentation


class IllegalActionError(ValueError):
    """Raised when a rewrite is applied to a gate pair that doesn't support it."""


def _swap_adjacent(gates: list[Gate], i: int, j: int) -> list[Gate]:
    """Return a copy of ``gates`` with positions ``i`` and ``j`` swapped."""
    new_gates = list(gates)
    new_gates[i], new_gates[j] = new_gates[j], new_gates[i]
    return new_gates


def can_commute(a: Gate, b: Gate) -> bool:
    """Two gates on disjoint qubits always commute (mirrors ``legal_actions``)."""
    return set(a.qubits).isdisjoint(b.qubits)


def can_merge_rotation(a: Gate, b: Gate) -> bool:
    """Same rotation gate, same qubit(s), back-to-back -> mergeable via angle sum."""
    return a.name == b.name and a.qubits == b.qubits and a.name in PARAMETRIC


def commute(ir: IntermediateRepresentation, i: int, j: int) -> IntermediateRepresentation:
    """Swap two adjacent, qubit-disjoint gates at positions ``i`` and ``j``.

    Pure reordering: gate count and per-gate contents are unchanged, only their
    position in the schedule moves. Used for ``ActionKind.COMMUTE`` and, by the
    same swap primitive, for ``MOVE_EARLIER`` / ``MOVE_LATER`` (see
    :func:`move`) since both are single adjacent-transposition steps on disjoint
    qubits.
    """
    gates = ir.gates
    if not (0 <= i < len(gates) and 0 <= j < len(gates)):
        raise IllegalActionError(f"commute({i}, {j}): index out of range for {len(gates)} gates")
    a, b = gates[i], gates[j]
    if not can_commute(a, b):
        raise IllegalActionError(f"commute({i}, {j}): gates share qubit(s) {set(a.qubits) & set(b.qubits)}")
    out = ir.copy()
    out.gates = _swap_adjacent(gates, i, j)
    return out


def move(ir: IntermediateRepresentation, i: int, earlier: bool) -> IntermediateRepresentation:
    """Move the gate at index ``i`` one slot earlier or later via a commuting swap.

    ``MOVE_EARLIER`` swaps ``(i-1, i)``; ``MOVE_LATER`` swaps ``(i, i+1)``. Both
    are no-ops (return an unchanged copy) at the schedule boundary or when the
    neighbor shares a qubit, rather than raising, since "move as far as
    possible" is a reasonable degenerate case for an RL agent to encounter
    often and shouldn't crash an episode.
    """
    gates = ir.gates
    if earlier:
        if i <= 0 or i >= len(gates):
            return ir.copy()
        neighbor = i - 1
        if not can_commute(gates[neighbor], gates[i]):
            return ir.copy()
        out = ir.copy()
        out.gates = _swap_adjacent(gates, neighbor, i)
        return out
    else:
        if i < 0 or i >= len(gates) - 1:
            return ir.copy()
        neighbor = i + 1
        if not can_commute(gates[i], gates[neighbor]):
            return ir.copy()
        out = ir.copy()
        out.gates = _swap_adjacent(gates, i, neighbor)
        return out


def merge_rotation(ir: IntermediateRepresentation, i: int, j: int) -> IntermediateRepresentation:
    """Merge two adjacent equal-axis rotations at ``i``, ``j`` by summing angles.

    ``Rz(a) . Rz(b) = Rz(a+b)`` (and likewise for ``rx``/``ry``/``p``): replaces
    the pair with a single gate carrying the summed parameter, reducing gate
    count by one. If the summed angle is (numerically) a multiple of 2*pi the
    gate becomes the identity and is dropped entirely, reducing gate count by
    two — this is the same cancellation the Sprint-1 GNN module's rule-based
    generator recognizes (see ``qco.modules.gnn_cancellation.features``), so
    Module A and Module B agree on what "cancels".
    """
    gates = ir.gates
    if not (0 <= i < len(gates) and 0 <= j < len(gates)):
        raise IllegalActionError(f"merge_rotation({i}, {j}): index out of range for {len(gates)} gates")
    a, b = gates[i], gates[j]
    if not can_merge_rotation(a, b):
        raise IllegalActionError(f"merge_rotation({i}, {j}): gates {a.name}/{b.name} on {a.qubits}/{b.qubits} not mergeable")

    lo, hi = (i, j) if i < j else (j, i)
    theta = a.params[0] + b.params[0]

    out = ir.copy()
    new_gates = list(gates)
    del new_gates[hi]
    del new_gates[lo]

    TWO_PI = 6.283185307179586
    remainder = theta % TWO_PI
    is_identity = min(remainder, TWO_PI - remainder) < 1e-9
    if not is_identity:
        merged = replace(a, params=(theta,))
        new_gates.insert(lo, merged)

    out.gates = new_gates
    return out


def apply_action(ir: IntermediateRepresentation, kind: str, i: int, j: int) -> IntermediateRepresentation:
    """Dispatch helper used by ``SchedulerEnv.step``; ``kind`` is an ``ActionKind`` name.

    Centralizing the dispatch here (rather than in ``environment.py``) keeps
    ``step`` itself a thin bookkeeping wrapper and makes every rewrite
    independently unit-testable.
    """
    if kind == "NOOP":
        return ir.copy()
    if kind == "COMMUTE":
        return commute(ir, i, j)
    if kind == "MOVE_EARLIER":
        return move(ir, i, earlier=True)
    if kind == "MOVE_LATER":
        return move(ir, i, earlier=False)
    if kind == "MERGE_ROTATION":
        return merge_rotation(ir, i, j)
    raise IllegalActionError(f"unknown action kind {kind!r}")