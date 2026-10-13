"""Module A — the commutation relation used by the CR pass and the environment.

Two sound sources are combined (a union of sound relations is sound):

1. the project's shared, conservative pairwise table
   (``qco.modules.gnn_cancellation.features.gates_commute``, which
   ``qco.modules.evolutionary.nsga2`` also builds on); and
2. a **wire-type rule** added here.

Wire-type rule
--------------
Every supported gate acts on each of its qubits in one of four ways:

====  ==========================================================  =============
type  meaning                                                     examples
====  ==========================================================  =============
Z     diagonal in the computational basis on this wire (control   ``rz``, ``cp``,
      wires of controlled gates are Z-type)                       CX *control*
X     built from X-axis Pauli exponentials on this wire           ``rx``, ``rxx``,
                                                                  CX *target*
Y     built from Y-axis Pauli exponentials on this wire           ``ry``
G     anything else                                               ``h``, ``swap``
====  ==========================================================  =============

Two gates commute if **on every qubit they share, both have the same non-G
type**. Why: each such gate has the form ``sum_z |z><z|_Z (x) V(z)_rest`` where
``V`` only involves commuting Pauli-axis operators, so shared Z wires carry
commuting projectors and shared X (or Y) wires carry commuting same-axis
operators. This recognizes, among others, any two diagonal gates
(``rzz(0,1)`` with ``rzz(1,2)`` — the heart of QAOA/QFT scheduling), and CX gates
sharing a control or a target, none of which the shared table knows about.

Soundness: ``gates_commute``/``commutes`` must only ever answer ``True`` when
the unitaries *exactly* commute (not merely up to a global phase — anticommuting
Paulis such as ``x`` and ``z`` are deliberately **not** treated as commuting). A
false positive would make the CR pass or a ``COMMUTE`` action silently change the
circuit. ``tests/test_rl_cr_pass.py`` checks both sources against real
unitaries for every gate pair on a 3-qubit register.

Modes
-----
``"disjoint"``  only gates on disjoint qubits commute (Sprint 1 behaviour).
                Swapping such gates never changes the dependency DAG, so it
                cannot change depth or gate count.
``"full"``      also gates that share qubits but provably commute. Swapping
                those *does* change the DAG, which is what lets a scheduler
                actually reduce depth.
"""

from __future__ import annotations

from qco.ir.intermediate_representation import Gate
from qco.modules.gnn_cancellation.features import gates_commute as _shared_gates_commute

COMMUTATION_MODES = ("disjoint", "full")

# Per-gate wire types, listed in operand order (controls first).
_WIRE_TYPES: dict[str, tuple[str, ...]] = {
    # single-qubit
    "z": ("Z",), "s": ("Z",), "sdg": ("Z",), "t": ("Z",), "tdg": ("Z",), "rz": ("Z",), "p": ("Z",),
    "x": ("X",), "rx": ("X",),
    "y": ("Y",), "ry": ("Y",),
    # two-qubit
    "cx": ("Z", "X"), "crx": ("Z", "X"), "cry": ("Z", "Y"),
    "cz": ("Z", "Z"), "cp": ("Z", "Z"), "crz": ("Z", "Z"), "rzz": ("Z", "Z"),
    "rxx": ("X", "X"),
    # three-qubit
    "ccx": ("Z", "Z", "X"),
}


def validate_mode(mode: str) -> str:
    if mode not in COMMUTATION_MODES:
        raise ValueError(f"commutation must be one of {COMMUTATION_MODES}, got {mode!r}")
    return mode


def disjoint(a: Gate, b: Gate) -> bool:
    return set(a.qubits).isdisjoint(b.qubits)


def _wire_types(g: Gate) -> dict[int, str]:
    types = _WIRE_TYPES.get(g.name)
    if types is None or len(types) != len(g.qubits):
        return {}
    return dict(zip(g.qubits, types))


def wire_type_commute(a: Gate, b: Gate) -> bool:
    """True iff every shared qubit has the same non-general wire type in both gates."""
    ta, tb = _wire_types(a), _wire_types(b)
    if not ta or not tb:
        return False
    shared = set(a.qubits) & set(b.qubits)
    if not shared:
        return True
    return all(ta[q] == tb[q] for q in shared)


def commutes(a: Gate, b: Gate, mode: str = "full") -> bool:
    """True iff ``a`` and ``b`` may be swapped under ``mode`` without changing the unitary."""
    if disjoint(a, b):
        return True
    if mode == "disjoint":
        return False
    return wire_type_commute(a, b) or bool(_shared_gates_commute(a, b))
