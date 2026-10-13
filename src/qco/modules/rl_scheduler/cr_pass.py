"""Module A — deterministic Commutation-and-Reduction (CR) pass.

Design (docs/phase1_literature_survey.md, Tao et al. 2026): pair the learned
policy with a deterministic pass so the agent spends its capacity on
*non-trivial* rewrites instead of rediscovering textbook cancellations.

What it does
------------
Repeatedly, until a fixed point:

1. **Trivial identity removal** — drop parametric gates whose angle is a
   multiple of their identity period (``rz(0)``, ``rz(2*pi)``, ...).
2. **Commuting inverse cancellation / rotation merge** — for each gate ``g``,
   scan forward along the gate list. Gates on disjoint qubits are skipped.
   A gate that touches ``g``'s qubits either

   * is a *partner* (same operands and :func:`~gate_algebra.reduce_pair`
     says the two combine) -> cancel / merge them, or
   * must *commute with ``g``* (so ``g`` can slide past it), or
   * blocks the scan (the first non-commuting gate ends the search).

   Because ``g`` slides past only gates it commutes with, the circuit's unitary
   is unchanged by every rewrite.

This subsumes the "adjacent" case (no blockers) and the "commuting inverse"
case (blockers that commute) from the issue, plus exact discrete-phase merges
(``t.t -> s``, ``s.s -> z``) and same-axis rotation merging.

Properties (all covered by ``tests/test_rl_cr_pass.py``)
--------------------------------------------------------
* **Sound** — output is unitary-equivalent to the input (up to global phase,
  the project's equivalence convention).
* **Monotone** — never increases gate count or depth.
* **Terminating** — every rewrite removes at least one gate.
* **Idempotent** — ``reduce(reduce(c)) == reduce(c)``.
* **Deterministic** — no randomness; same input, same output.

Gate-count reduction only: reordering commuting gates (the RL agent's job) can
change *depth* but cannot expose new cancellations, because a gate that does
not commute with ``g`` stays between ``g`` and its partner in every valid
order. See ``docs/gaps_and_notes_module_a.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from qco.ir.intermediate_representation import Gate, IntermediateRepresentation
from qco.modules.rl_scheduler.commutation import commutes
from qco.modules.rl_scheduler.gate_algebra import NOT_REDUCIBLE, is_trivial_identity, reduce_pair, same_operands

DEFAULT_MAX_SWEEPS = 10_000


@dataclass(slots=True)
class CRStats:
    """What a CR run did (counts of rewrite events, not of gates)."""

    identities_removed: int = 0
    pairs_cancelled: int = 0
    pairs_merged: int = 0

    @property
    def gates_removed(self) -> int:
        """Net reduction in gate count: identity=1, cancelled pair=2, merge=1."""
        return self.identities_removed + 2 * self.pairs_cancelled + self.pairs_merged

    @property
    def events(self) -> int:
        return self.identities_removed + self.pairs_cancelled + self.pairs_merged


def _find_partner(gates: list[Gate], i: int, window: int | None) -> int | None:
    """Index ``j > i`` of a gate ``gates[i]`` can be brought next to and reduced with."""
    g = gates[i]
    qs = set(g.qubits)
    stop = len(gates) if window is None else min(len(gates), i + 1 + window)
    for j in range(i + 1, stop):
        h = gates[j]
        if qs.isdisjoint(h.qubits):
            continue
        if same_operands(g, h) and reduce_pair(g, h) is not NOT_REDUCIBLE:
            return j
        if not commutes(g, h, "full"):
            return None
    return None


def reduce_with_stats(
    ir: IntermediateRepresentation,
    *,
    window: int | None = None,
) -> tuple[IntermediateRepresentation, CRStats]:
    """Run the CR pass to a fixed point; return the reduced circuit and event counts.

    ``window`` bounds how far ahead each gate scans for a partner (``None`` =
    unbounded). A bound trades a little reduction for a hard O(n * window) cost
    per sweep, useful when the pass runs inside an RL step on large circuits.
    """
    stats = CRStats()
    gates = list(ir.gates)

    # 1. trivial identities
    kept = [g for g in gates if not is_trivial_identity(g)]
    stats.identities_removed += len(gates) - len(kept)
    gates = kept

    # 2. commuting reduction. After any rewrite, restart from the beginning:
    # removing a gate can unblock an earlier gate's scan (nested cancellations
    # like A B B^-1 A^-1). Every rewrite strictly shrinks the list, so this ends.
    i = 0
    while i < len(gates):
        j = _find_partner(gates, i, window)
        if j is None:
            i += 1
            continue
        merged = reduce_pair(gates[i], gates[j])
        if merged is None:
            del gates[j]
            del gates[i]
            stats.pairs_cancelled += 1
        else:
            # ``gates[i]`` slid forward to ``gates[j]``; the merged gate sits there.
            gates[j] = merged  # type: ignore[assignment]
            del gates[i]
            stats.pairs_merged += 1
        i = 0

    out = IntermediateRepresentation(ir.num_qubits, gates, ir.name)
    return out, stats


def reduce_circuit(ir: IntermediateRepresentation, *, window: int | None = None) -> IntermediateRepresentation:
    """Convenience wrapper returning only the reduced circuit."""
    return reduce_with_stats(ir, window=window)[0]


def is_reduced(ir: IntermediateRepresentation, *, window: int | None = None) -> bool:
    """True iff the CR pass has nothing left to do on ``ir``."""
    return reduce_with_stats(ir, window=window)[1].events == 0
