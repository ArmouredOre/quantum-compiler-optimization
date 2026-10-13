"""Module A — gate-level algebra shared by the rewrites and the CR pass.

Pure functions over :class:`~qco.ir.intermediate_representation.Gate`, no
circuit or environment state. Everything the deterministic
Commutation-and-Reduction (CR) pass and the RL environment's
``MERGE_ROTATION`` action need to know about *two gates multiplying together*
lives here, so the two can never disagree about what cancels.

Equivalence convention
----------------------
The project's verifier (``EquivalenceChecker``) compares unitaries **up to a
global phase**, so a rotation by a multiple of ``2*pi`` (which is ``-I``) counts
as the identity for ``rx/ry/rz/rzz/rxx``. This is *not* true for the
controlled rotations ``crx/cry/crz``: ``crz(2*pi)`` is a ``CZ`` (a relative
phase), so their identity period is ``4*pi``. ``p`` / ``cp`` are exactly the
identity at ``2*pi``. The table below encodes this; getting it wrong would make
the CR pass silently delete a real gate.
"""

from __future__ import annotations

import math
from dataclasses import replace

from qco.ir.intermediate_representation import PARAMETRIC, SELF_INVERSE, Gate

TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi

# Angle after which each parametric gate is the identity (up to global phase).
IDENTITY_PERIOD: dict[str, float] = {
    "rx": TWO_PI,
    "ry": TWO_PI,
    "rz": TWO_PI,
    "rzz": TWO_PI,
    "rxx": TWO_PI,
    "p": TWO_PI,
    "cp": TWO_PI,
    "crx": FOUR_PI,
    "cry": FOUR_PI,
    "crz": FOUR_PI,
}

# Gates whose action does not depend on the order of their qubit operands.
SYMMETRIC: frozenset[str] = frozenset({"cz", "swap", "cp", "rzz", "rxx"})

# Exact discrete-phase identities: g . g == result (``None`` -> identity).
_PHASE_SQUARES: dict[str, str] = {"t": "s", "tdg": "sdg", "s": "z", "sdg": "z"}
_PHASE_INVERSES: frozenset[frozenset[str]] = frozenset(
    {frozenset({"s", "sdg"}), frozenset({"t", "tdg"})}
)

ANGLE_TOL = 1e-9


class _NotReducible:
    """Sentinel type: the two gates do not combine into something simpler."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NOT_REDUCIBLE"


NOT_REDUCIBLE = _NotReducible()


def operand_key(g: Gate) -> tuple[int, ...]:
    """Qubit operands, order-normalized for symmetric gates."""
    return tuple(sorted(g.qubits)) if g.name in SYMMETRIC else g.qubits


def same_operands(a: Gate, b: Gate) -> bool:
    return operand_key(a) == operand_key(b)


def angle_is_identity(name: str, theta: float, tol: float = ANGLE_TOL) -> bool:
    """True iff ``name(theta)`` is the identity (up to global phase)."""
    period = IDENTITY_PERIOD.get(name)
    if period is None:
        return False
    r = theta % period
    return min(r, period - r) < tol


def is_trivial_identity(g: Gate) -> bool:
    """A parametric gate whose single angle is a multiple of its period."""
    if g.name not in PARAMETRIC or len(g.params) != 1:
        return False
    return angle_is_identity(g.name, g.params[0])


def reduce_pair(a: Gate, b: Gate) -> "Gate | None | _NotReducible":
    """Combine ``a`` followed by ``b`` (adjacent, same operands).

    Returns
    -------
    ``None``
        the pair multiplies to the identity -> drop both gates.
    a :class:`Gate`
        the pair is equivalent to this single gate -> replace both with it.
    ``NOT_REDUCIBLE``
        nothing to do.

    Never inspects neighbouring gates: whether ``a`` and ``b`` may be brought
    together is the caller's job (see :mod:`qco.modules.rl_scheduler.cr_pass`).
    """
    if not same_operands(a, b):
        return NOT_REDUCIBLE

    # Self-inverse gates: g . g = I.  (cx/ccx need *identical* ordered operands,
    # which ``same_operands`` already guarantees for non-symmetric gates.)
    if a.name == b.name and a.name in SELF_INVERSE:
        return None

    # s . sdg = t . tdg = I.
    if frozenset({a.name, b.name}) in _PHASE_INVERSES:
        return None

    # Exact discrete phase squares: t.t = s, tdg.tdg = sdg, s.s = z, sdg.sdg = z.
    if a.name == b.name and a.name in _PHASE_SQUARES:
        return replace(a, name=_PHASE_SQUARES[a.name])

    # Same-axis rotations add their angles.
    if a.name == b.name and a.name in PARAMETRIC:
        if len(a.params) != 1 or len(b.params) != 1:
            return NOT_REDUCIBLE
        theta = a.params[0] + b.params[0]
        if angle_is_identity(a.name, theta):
            return None
        return replace(a, params=(theta,))

    return NOT_REDUCIBLE
