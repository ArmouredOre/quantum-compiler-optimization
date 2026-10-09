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
import functools
import math
from dataclasses import dataclass
from fractions import Fraction

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import z3 as _z3_module
    _HAS_Z3 = True
except ImportError:  # pragma: no cover
    _z3_module = None  # type: ignore
    _HAS_Z3 = False

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
    table = {
        "id": _I, "i": _I,
        "h": _H, "x": _X, "y": _Y, "z": _Z, "s": _S, "t": _T,
        "sdg": [[1, 0], [0, -1j]], "tdg": [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]],
    }
    if name in table:
        return [row[:] for row in table[name]]
    th = params[0] if params else 0.0
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


_CTRL_BASE = {
    "cx": "x", "cz": "z", "cp": "p", "crx": "rx", "cry": "ry", "crz": "rz",
    "ch": "h", "cy": "y", "cs": "s", "ct": "t", "csdg": "sdg", "ctdg": "tdg",
}


def _two_qubit(name: str, params: tuple[float, ...]):
    """Return local 4x4 matrix for a 2-qubit gate in basis |00>, |01>, |10>, |11>."""
    if name == "swap":
        return [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    if name in _CTRL_BASE:
        u = _one_qubit(_CTRL_BASE[name], params)
        return [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, u[0][0], u[0][1]],
            [0, 0, u[1][0], u[1][1]],
        ]
    if name == "rxx":
        th = params[0] if params else 0.0
        c, s = math.cos(th / 2), -1j * math.sin(th / 2)
        return [[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]]
    if name == "rzz":
        th = params[0] if params else 0.0
        em, ep = cmath.exp(-1j * th / 2), cmath.exp(1j * th / 2)
        return [[em, 0, 0, 0], [0, ep, 0, 0], [0, 0, ep, 0], [0, 0, 0, em]]
    if name == "ryy":
        th = params[0] if params else 0.0
        c, s = math.cos(th / 2), 1j * math.sin(th / 2)
        return [[c, 0, 0, s], [0, c, -s, 0], [0, -s, c, 0], [s, 0, 0, c]]
    raise NotImplementedError(f"2-qubit gate {name!r} not supported in numeric checker")


def _three_qubit(name: str, params: tuple[float, ...]):
    """Return local 8x8 matrix for a 3-qubit gate."""
    if name == "ccx":
        m = [[1j * 0 if i != j else 1 + 0j for j in range(8)] for i in range(8)]
        m[6][6] = 0j; m[6][7] = 1 + 0j; m[7][7] = 0j; m[7][6] = 1 + 0j
        return m
    if name == "cswap":
        m = [[1j * 0 if i != j else 1 + 0j for j in range(8)] for i in range(8)]
        m[5][5] = 0j; m[5][6] = 1 + 0j; m[6][6] = 0j; m[6][5] = 1 + 0j
        return m
    raise NotImplementedError(f"3-qubit gate {name!r} not supported in numeric checker")


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


def _gate_on_register(g, n: int):
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

    if g.name == "rxx":
        th = g.params[0] if g.params else 0.0
        q0, q1 = g.qubits
        c = math.cos(th / 2)
        s = -1j * math.sin(th / 2)
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            flipped = basis ^ (1 << q0) ^ (1 << q1)
            out[basis][basis] += c
            out[flipped][basis] += s
        return out

    if g.name == "rzz":
        th = g.params[0] if g.params else 0.0
        q0, q1 = g.qubits
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            parity = ((basis >> q0) & 1) ^ ((basis >> q1) & 1)
            out[basis][basis] += cmath.exp(-1j * th / 2 if parity == 0 else 1j * th / 2)
        return out

    if g.name == "ryy":
        th = g.params[0] if g.params else 0.0
        q0, q1 = g.qubits
        c = math.cos(th / 2)
        s = 1j * math.sin(th / 2)
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            b0 = (basis >> q0) & 1
            b1 = (basis >> q1) & 1
            flipped = basis ^ (1 << q0) ^ (1 << q1)
            sign = 1 if b0 == b1 else -1
            out[basis][basis] += c
            out[flipped][basis] += sign * s
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

    if g.name == "cswap":
        c, t1, t2 = g.qubits
        out = [[0j] * dim for _ in range(dim)]
        for basis in range(dim):
            bits = [(basis >> q) & 1 for q in range(n)]
            if bits[c]:
                bits[t1], bits[t2] = bits[t2], bits[t1]
            out[sum(b << q for q, b in enumerate(bits))][basis] += 1
        return out

    raise NotImplementedError(
        f"gate {g.name!r} not in the numeric checker (use the SMT path)"
    )


def circuit_unitary(ir: IntermediateRepresentation) -> list[list[Complex]]:
    """Dense unitary of ``ir`` (little-endian qubit order). <= 8 qubits."""
    n = ir.num_qubits
    if n > 8:
        raise ValueError("numeric unitary limited to 8 qubits; use the SMT path")
    dim = 1 << n
    if n == 0:
        return [[1 + 0j]]

    if _HAS_NUMPY:
        u = np.eye(dim, dtype=complex).reshape((2,) * n + (dim,))
        for g in ir.gates:
            k = len(g.qubits)
            if k == 1:
                mat = np.array(_one_qubit(g.name, g.params), dtype=complex)
                ax = n - 1 - g.qubits[0]
                res = np.tensordot(mat, u, axes=([1], [ax]))
                u = np.moveaxis(res, 0, ax)
            elif k == 2:
                mat = np.array(_two_qubit(g.name, g.params), dtype=complex).reshape(2, 2, 2, 2)
                ax = [n - 1 - q for q in g.qubits]
                res = np.tensordot(mat, u, axes=([2, 3], ax))
                u = np.moveaxis(res, (0, 1), ax)
            elif k == 3:
                mat = np.array(_three_qubit(g.name, g.params), dtype=complex).reshape(2, 2, 2, 2, 2, 2)
                ax = [n - 1 - q for q in g.qubits]
                res = np.tensordot(mat, u, axes=([3, 4, 5], ax))
                u = np.moveaxis(res, (0, 1, 2), ax)
            else:
                layer = _gate_on_register(g, n)
                u = (np.array(layer, dtype=complex) @ u.reshape(dim, dim)).reshape((2,) * n + (dim,))
        return u.reshape(dim, dim).tolist()

    # Optimized pure-Python in-place fallback
    u = [[1j * 0 if i != j else 1 + 0j for j in range(dim)] for i in range(dim)]
    for g in ir.gates:
        if len(g.qubits) == 1:
            mat = _one_qubit(g.name, g.params)
            target = g.qubits[0]
            bit = 1 << target
            m00, m01 = mat[0]
            m10, m11 = mat[1]
            for base in range(0, dim, 2 * bit):
                for offset in range(bit):
                    r0 = base + offset
                    r1 = r0 + bit
                    row0 = u[r0]
                    row1 = u[r1]
                    u[r0] = [m00 * a + m01 * b for a, b in zip(row0, row1)]
                    u[r1] = [m10 * a + m11 * b for a, b in zip(row0, row1)]
        elif g.name == "cx":
            c_bit = 1 << g.qubits[0]
            t_bit = 1 << g.qubits[1]
            for base in range(dim):
                if (base & c_bit) and not (base & t_bit):
                    r0 = base
                    r1 = base | t_bit
                    u[r0], u[r1] = u[r1], u[r0]
        elif g.name == "swap":
            bit0 = 1 << g.qubits[0]
            bit1 = 1 << g.qubits[1]
            for base in range(dim):
                if (base & bit0) and not (base & bit1):
                    r0 = base
                    r1 = (base & ~bit0) | bit1
                    u[r0], u[r1] = u[r1], u[r0]
        elif g.name == "ccx":
            c1_bit = 1 << g.qubits[0]
            c2_bit = 1 << g.qubits[1]
            t_bit = 1 << g.qubits[2]
            for base in range(dim):
                if (base & c1_bit) and (base & c2_bit) and not (base & t_bit):
                    r0 = base
                    r1 = base | t_bit
                    u[r0], u[r1] = u[r1], u[r0]
        else:
            layer = _gate_on_register(g, n)
            u = _matmul(layer, u)
    return u


@functools.lru_cache(maxsize=4096)
def _to_z3_real(val_rounded: float):
    """Convert a rounded float to an exact Z3 RealVal via rational approximation."""
    frac = Fraction(val_rounded).limit_denominator(1_000_000)
    return _z3_module.RealVal(f"{frac.numerator}/{frac.denominator}")


def _z3_val_to_float(val) -> float:
    """Safely convert a Z3 Real model value (RatNumRef, AlgebraicNumRef) to Python float."""
    if hasattr(val, "as_fraction"):
        return float(val.as_fraction())
    if hasattr(val, "approx"):
        return float(val.approx(20).as_fraction())
    if hasattr(val, "as_decimal"):
        return float(val.as_decimal(15).rstrip("?"))
    return float(str(val))


def check_smt(
    ua: list[list[complex]],
    ub: list[list[complex]],
    *,
    atol: float = 1e-6,
    timeout: float = 5.0,
    solver_backend: str = "z3",
) -> EquivalenceResult:
    """Check whether ``ua == e^{i*phi} * ub`` for some global phase using Z3.

    Returns an :class:`EquivalenceResult` with ``method="smt"``.

    Raises
    ------
    ImportError
        If Z3 (or CVC5) is not installed.
    RuntimeError
        If an unsupported ``solver_backend`` is requested.
    """
    if solver_backend == "cvc5":
        try:
            import cvc5  # noqa: F401  # pragma: no cover
        except ImportError:
            raise ImportError(
                "cvc5 Python bindings are not installed. "
                "Install with: pip install cvc5"
            )
        raise RuntimeError(  # pragma: no cover
            "CVC5 backend is not yet implemented; use solver_backend='z3'."
        )

    if not _HAS_Z3:
        raise ImportError(
            "z3-solver is not installed. "
            "Install with: pip install 'qco[smt]' or pip install z3-solver"
        )

    dim = len(ua)

    # Fast invariant check: |U_a[i][j]| must equal |U_b[i][j]| up to tolerance
    # because |e^{i*phi} * z| == |z|. If any entry norm violates this,
    # the unitaries cannot be equivalent under any global phase.
    tol_norm = max(atol * 5, 1e-4)
    for i in range(dim):
        row_a = ua[i]
        row_b = ub[i]
        for j in range(dim):
            if abs(abs(row_a[j]) - abs(row_b[j])) > tol_norm:
                return EquivalenceResult(False, "smt", "unitaries not equivalent (UNSAT)")

    # Find pivot entry in ub with maximum amplitude
    max_val = -1.0
    pivot = (0, 0)
    for i in range(dim):
        row_b = ub[i]
        for j in range(dim):
            v = abs(row_b[j])
            if v > max_val:
                max_val = v
                pivot = (i, j)

    # Declare real-valued phase variables: cos(phi) and sin(phi) with c^2+s^2=1
    c = _z3_module.Real("c")
    s = _z3_module.Real("s")

    solver = _z3_module.Solver()
    solver.set("timeout", int(timeout * 1000))
    solver.add(c * c + s * s == _z3_module.RealVal("1"))

    tol = _to_z3_real(round(atol, 8))

    def _add_constraint(r: int, cl: int):
        a_val = ua[r][cl]
        b_val = ub[r][cl]
        ra = _to_z3_real(round(a_val.real, 8))
        ia = _to_z3_real(round(a_val.imag, 8))
        rb = _to_z3_real(round(b_val.real, 8))
        ib = _to_z3_real(round(b_val.imag, 8))
        diff_re = ra - (c * rb - s * ib)
        solver.add(_z3_module.And(diff_re >= -tol, diff_re <= tol))
        diff_im = ia - (c * ib + s * rb)
        solver.add(_z3_module.And(diff_im >= -tol, diff_im <= tol))

    # Add pivot constraint
    _add_constraint(pivot[0], pivot[1])

    # Counterexample-guided refinement loop (CEGAR)
    while True:
        status = solver.check()
        if status == _z3_module.unsat:
            return EquivalenceResult(False, "smt", "unitaries not equivalent (UNSAT)")
        elif status != _z3_module.sat:
            return EquivalenceResult(False, "smt", "SMT solver timeout/unknown")

        model = solver.model()
        c_val = _z3_val_to_float(model[c])
        s_val = _z3_val_to_float(model[s])

        # Check all entries against candidate model
        ph_c = complex(c_val, s_val)
        violated = None
        for i in range(dim):
            row_a = ua[i]
            row_b = ub[i]
            for j in range(dim):
                if abs(row_a[j] - ph_c * row_b[j]) > max(atol * 5, 1e-4):
                    violated = (i, j)
                    break
            if violated is not None:
                break

        if violated is None:
            phi = math.atan2(s_val, c_val)
            return EquivalenceResult(True, "smt", f"global phase phi={phi:.6f} rad")

        # Refine: add constraint for the violated entry
        _add_constraint(violated[0], violated[1])


class EquivalenceChecker:
    def __init__(
        self,
        backend: str = "auto",
        atol: float = 1e-8,
        rtol: float = 1e-5,
        timeout: float = 5.0,
        solver_backend: str = "z3",
    ):
        self.backend = backend
        self.atol = atol
        self.rtol = rtol
        self.timeout = timeout
        self.solver_backend = solver_backend

    def _use_smt(self, original: "IntermediateRepresentation", rewrite: "IntermediateRepresentation") -> bool:
        """Decide whether to route to the SMT path under 'auto' backend."""
        n = original.num_qubits
        total_gates = len(original.gates) + len(rewrite.gates)
        # Prefer numeric for small windows; SMT beyond
        return n > 5 or total_gates > 50

    def check(self, original: "IntermediateRepresentation", rewrite: "IntermediateRepresentation") -> EquivalenceResult:
        if original.num_qubits != rewrite.num_qubits:
            return EquivalenceResult(False, "numeric", "qubit count differs")

        # Determine effective backend
        backend = self.backend
        if backend == "auto":
            backend = "smt" if self._use_smt(original, rewrite) else "numeric"

        if backend in ("smt", "z3", "cvc5"):
            ua = circuit_unitary(original)
            ub = circuit_unitary(rewrite)
            return check_smt(
                ua, ub,
                atol=max(self.atol, 1e-6),  # SMT uses looser tolerance due to rational approximation
                timeout=self.timeout,
                solver_backend=self.solver_backend if backend == "smt" else backend,
            )
        ua, ub = circuit_unitary(original), circuit_unitary(rewrite)
        dim = len(ua)

        if _HAS_NUMPY:
            ua_arr = np.asarray(ua, dtype=complex)
            ub_arr = np.asarray(ub, dtype=complex)
            flat_idx = int(np.argmax(np.abs(ua_arr)))
            i, j = np.unravel_index(flat_idx, ua_arr.shape)
            pivot_val_a = ua_arr[i, j]
            pivot_val_b = ub_arr[i, j]

            if abs(pivot_val_a) <= self.atol or abs(pivot_val_b) <= self.atol:
                return EquivalenceResult(False, "numeric", "phase pivot mismatch")

            ph_raw = pivot_val_a / pivot_val_b
            if abs(abs(ph_raw) - 1.0) > 1e-3:
                return EquivalenceResult(False, "numeric", "unitaries differ (phase scale mismatch)")

            ph = ph_raw / abs(ph_raw)
            ok = np.allclose(ua_arr, ph * ub_arr, atol=self.atol, rtol=self.rtol)
            return EquivalenceResult(bool(ok), "numeric", "" if ok else "unitaries differ")

        # Pure-Python path
        max_val = -1.0
        pivot = (0, 0)
        for r in range(dim):
            ua_r = ua[r]
            for c in range(dim):
                val = abs(ua_r[c])
                if val > max_val:
                    max_val = val
                    pivot = (r, c)

        i, j = pivot
        pivot_val_a = ua[i][j]
        pivot_val_b = ub[i][j]

        if abs(pivot_val_a) <= self.atol or abs(pivot_val_b) <= self.atol:
            return EquivalenceResult(False, "numeric", "phase pivot mismatch")

        ph_raw = pivot_val_a / pivot_val_b
        if abs(abs(ph_raw) - 1.0) > 1e-3:
            return EquivalenceResult(False, "numeric", "unitaries differ (phase scale mismatch)")

        ph = ph_raw / abs(ph_raw)
        for r in range(dim):
            ua_r = ua[r]
            ub_r = ub[r]
            for c in range(dim):
                diff = abs(ua_r[c] - ph * ub_r[c])
                tol = self.atol + self.rtol * abs(ua_r[c])
                if diff > tol:
                    return EquivalenceResult(False, "numeric", "unitaries differ")
        return EquivalenceResult(True, "numeric", "")
