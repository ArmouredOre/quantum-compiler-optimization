"""Module C — unitary-equivalence checking for proposed rewrites.

Every rewrite emitted by Module A or Module B must pass through here before the
pipeline accepts it (the verification invariant in docs/architecture.md).

Phase 3 (Sahib Singh):
* SMT path  : encode U(original) == U(rewrite) up to global phase in Z3/CVC5 over
              a symbolic gate algebra; return SAT/UNSAT + counterexample.
* Numeric path (validation): build both unitaries with Qiskit ``Operator`` and
  compare (only for small windows; used to cross-check the SMT encoding).

Phase 2 ships an exact numeric checker using a tiny built-in statevector/unitary
simulator (no third-party deps) valid up to ~6 qubits — enough for bounded
windows and for the test-suite ground truth.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from qco.ir.intermediate_representation import IntermediateRepresentation

Complex = complex


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    equivalent: bool
    method: str
    detail: str = ""


# -- minimal dependency-free unitary builder (<= ~6 qubits) -----------------

def _kron(a: list[list[Complex]], b: list[list[Complex]]) -> list[list[Complex]]:
    ra, ca, rb, cb = len(a), len(a[0]), len(b), len(b[0])
    out = [[0j] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    out[i * rb + k][j * cb + l] = a[i][j] * b[k][l]
    return out


def _matmul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    out = [[0j] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        for k in range(m):
            aik = ai[k]
            if aik == 0:
                continue
            bk = b[k]
            for j in range(p):
                out[i][j] += aik * bk[j]
    return out


_I = [[1, 0], [0, 1]]
_H = [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]
_X = [[0, 1], [1, 0]]
_Y = [[0, -1j], [1j, 0]]
_Z = [[1, 0], [0, -1]]
_S = [[1, 0], [0, 1j]]
_T = [[1, 0], [0, cmath.exp(1j * math.pi / 4)]]


def _one_qubit(name: str, params: tuple[float, ...]):
    table = {"h": _H, "x": _X, "y": _Y, "z": _Z, "s": _S, "t": _T,
             "sdg": [[1, 0], [0, -1j]], "tdg": [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]]}
    if name in table:
        return [row[:] for row in table[name]]
    th = params[0]
    if name == "rx":
        c, s = math.cos(th / 2), math.sin(th / 2)
        return [[c, -1j * s], [-1j * s, c]]
    if name == "ry":
        c, s = math.cos(th / 2), math.sin(th / 2)
        return [[c, -s], [s, c]]
    if name in ("rz", "p"):
        e = cmath.exp(1j * th) if name == "p" else 1
        return [[cmath.exp(-1j * th / 2) if name == "rz" else 1, 0],
                [0, cmath.exp(1j * th / 2) if name == "rz" else e]]
    raise NotImplementedError(f"1-qubit gate {name!r} not in the Phase 2 numeric checker")


def circuit_unitary(ir: IntermediateRepresentation) -> list[list[Complex]]:
    """Dense unitary of ``ir`` (little-endian qubit order). <= ~6 qubits."""
    n = ir.num_qubits
    if n > 8:
        raise ValueError("numeric unitary limited to 8 qubits; use the SMT path")
    dim = 1 << n
    u = [[1j * 0 if i != j else 1 + 0j for j in range(dim)] for i in range(dim)]
    for g in ir.gates:
        layer = _gate_on_register(g, n)
        u = _matmul(layer, u)
    return u


_CTRL_BASE = {"cx": "x", "cz": "z", "cp": "p", "crx": "rx", "cry": "ry", "crz": "rz"}


def _apply_1q(mat, target: int, n: int):
    """Dense operator applying 2x2 ``mat`` on ``target`` of an n-qubit register."""
    dim = 1 << n
    out = [[0j] * dim for _ in range(dim)]
    for basis in range(dim):
        b = (basis >> target) & 1
        other = basis & ~(1 << target)
        for nb in (0, 1):
            amp = mat[nb][b]
            if amp != 0:
                out[other | (nb << target)][basis] += amp
    return out


def _gate_on_register(g, n):
    dim = 1 << n
    if len(g.qubits) == 1:
        return _apply_1q(_one_qubit(g.name, g.params), g.qubits[0], n)

    if g.name == "swap":
        c, t = g.qubits
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            bits = [(basis >> q) & 1 for q in range(n)]
            bits[c], bits[t] = bits[t], bits[c]
            out[sum(b << q for q, b in enumerate(bits))][basis] += 1
        return out

    if g.name in _CTRL_BASE:
        c, t = g.qubits
        base = _one_qubit(_CTRL_BASE[g.name], g.params)
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            if (basis >> c) & 1 == 0:
                out[basis][basis] += 1
            else:
                b = (basis >> t) & 1
                other = basis & ~(1 << t)
                for nb in (0, 1):
                    amp = base[nb][b]
                    if amp != 0:
                        out[other | (nb << t)][basis] += amp
        return out

    if g.name == "ccx":
        c1, c2, t = g.qubits
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            bits = [(basis >> q) & 1 for q in range(n)]
            if bits[c1] and bits[c2]:
                bits[t] ^= 1
            out[sum(b << q for q, b in enumerate(bits))][basis] += 1
        return out

    if g.name == "rzz":
        # exp(-i theta/2 * Z⊗Z): diagonal, eigenvalue +1 if bits equal else -1.
        th = g.params[0]
        q0, q1 = g.qubits
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            parity = ((basis >> q0) & 1) ^ ((basis >> q1) & 1)
            out[basis][basis] += cmath.exp(-1j * th / 2 if parity == 0 else 1j * th / 2)
        return out

    raise NotImplementedError(
        f"gate {g.name!r} not in the Phase 2 numeric checker (use the SMT path)"
    )


class EquivalenceChecker:
    def __init__(self, backend: str = "auto", atol: float = 1e-8):
        self.backend = backend
        self.atol = atol

    def check(self, original: IntermediateRepresentation, rewrite: IntermediateRepresentation) -> EquivalenceResult:
        if self.backend in ("smt", "z3", "cvc5"):  # pragma: no cover - Phase 3
            raise NotImplementedError("SMT equivalence encoding lands in Phase 3 (Sahib Singh)")
        if original.num_qubits != rewrite.num_qubits:
            return EquivalenceResult(False, "numeric", "qubit count differs")
        ua, ub = circuit_unitary(original), circuit_unitary(rewrite)
        # Compare up to a global phase.
        pivot = next(((i, j) for i in range(len(ua)) for j in range(len(ua))
                      if abs(ua[i][j]) > 1e-6), None)
        if pivot is None:
            return EquivalenceResult(True, "numeric", "both ~zero")
        i, j = pivot
        if abs(ub[i][j]) < 1e-6:
            return EquivalenceResult(False, "numeric", "phase pivot mismatch")
        ph = ua[i][j] / ub[i][j]
        ok = all(abs(ua[r][c] - ph * ub[r][c]) <= self.atol
                 for r in range(len(ua)) for c in range(len(ua)))
        return EquivalenceResult(ok, "numeric", "" if ok else "unitaries differ")
