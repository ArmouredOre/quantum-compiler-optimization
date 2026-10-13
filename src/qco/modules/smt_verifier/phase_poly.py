"""Phase-polynomial SMT verifier for {CX, Rz, X} circuit windows.

Novelty N2 (Sahib Singh / Issue #27):
Enables exact symbolic equivalence checking beyond the 8-qubit dense-unitary ceiling.
Represents circuits over {CX, Rz, X, Z, S, T, P} as an affine parity transformation
A x + b and a phase polynomial Phi(x) over F_2^n.
Encodes equivalence into Z3 using bit-vectors and linear arithmetic.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import z3 as _z3
    _HAS_Z3 = True
except ImportError:  # pragma: no cover
    _z3 = None
    _HAS_Z3 = False

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.smt_verifier.equivalence import EquivalenceResult

_TWO_PI = 2.0 * math.pi
_PHASE_POLY_GATES = frozenset({"cx", "rz", "x", "z", "s", "sdg", "t", "tdg", "p"})


def is_phase_poly_circuit(ir: IntermediateRepresentation) -> bool:
    """Return True if all gates in ``ir`` belong to the {CX, Rz, X} family."""
    if not ir.gates:
        return True
    return all(g.name in _PHASE_POLY_GATES for g in ir.gates)


@dataclass(frozen=True, slots=True)
class PhasePolynomial:
    """Affine parity map and phase polynomial representation of a circuit.

    For basis state |x>, U|x> = e^{i * Phi(x)} |A x + b>, where:
    * ``parity_matrix``: tuple of integers (length n), row j is wire j's parity mask.
    * ``shifts``: tuple of {0, 1} (length n), row j is wire j's affine shift.
    * ``phase_terms``: tuple of (parity_mask, angle) with angle in (-pi, pi].
    * ``global_phase``: constant global phase in radians.
    """
    num_qubits: int
    parity_matrix: tuple[int, ...]
    shifts: tuple[int, ...]
    phase_terms: tuple[tuple[int, float], ...]
    global_phase: float


def _normalize_angle(angle: float) -> float:
    """Normalize angle to (-pi, pi]."""
    a = (angle + math.pi) % _TWO_PI - math.pi
    if abs(a - math.pi) < 1e-9:
        return math.pi
    return a


def extract_phase_polynomial(ir: IntermediateRepresentation) -> PhasePolynomial:
    """Extract the affine parity map and phase polynomial from ``ir``."""
    n = ir.num_qubits
    # Parity bitmask on wire j: initially 1 << j
    parities = [1 << j for j in range(n)]
    shifts = [0 for _ in range(n)]
    term_map: dict[int, float] = {}
    global_phase = 0.0

    for g in ir.gates:
        name = g.name
        if name == "cx":
            c, t = g.qubits
            parities[t] ^= parities[c]
            shifts[t] ^= shifts[c]

        elif name == "x":
            q = g.qubits[0]
            shifts[q] ^= 1

        elif name in ("rz", "z", "s", "sdg", "t", "tdg", "p"):
            q = g.qubits[0]
            if name == "rz":
                theta = float(g.params[0]) if g.params else 0.0
            elif name == "z":
                theta = math.pi
            elif name == "s":
                theta = math.pi / 2
            elif name == "sdg":
                theta = -math.pi / 2
            elif name == "t":
                theta = math.pi / 4
            elif name == "tdg":
                theta = -math.pi / 4
            elif name == "p":
                theta = float(g.params[0]) if g.params else 0.0

            p_mask = parities[q]
            s = shifts[q]

            if name == "p":
                # P(theta) applies phase 0 to |0> and theta to |1>
                # When wire has w_q(x) = p_mask*x ^ s:
                # phase = theta * w_q(x)
                if s == 0:
                    delta_angle = theta
                else:
                    global_phase += theta
                    delta_angle = -theta
            else:
                # Rz(theta) = exp(i * (-theta/2 + theta * w_q(x)))
                if s == 0:
                    global_phase += -theta / 2
                    delta_angle = theta
                else:
                    global_phase += theta / 2
                    delta_angle = -theta

            if p_mask == 0:
                global_phase += delta_angle
            else:
                term_map[p_mask] = term_map.get(p_mask, 0.0) + delta_angle

        else:
            raise ValueError(f"Gate {name!r} not supported in phase-polynomial extraction")

    # Clean and normalize non-zero terms
    cleaned_terms: list[tuple[int, float]] = []
    for mask, raw_angle in sorted(term_map.items()):
        norm = _normalize_angle(raw_angle)
        if abs(norm) > 1e-8:
            cleaned_terms.append((mask, norm))

    return PhasePolynomial(
        num_qubits=n,
        parity_matrix=tuple(parities),
        shifts=tuple(shifts),
        phase_terms=tuple(cleaned_terms),
        global_phase=_normalize_angle(global_phase),
    )


def check_phase_poly(
    ir_a: IntermediateRepresentation,
    ir_b: IntermediateRepresentation,
    *,
    atol: float = 1e-6,
    timeout: float = 5.0,
) -> EquivalenceResult:
    """Check equivalence of two {CX, Rz, X} circuits using Phase Polynomial SMT.

    Returns an :class:`EquivalenceResult` with ``method='phase_poly'``.
    """
    if ir_a.num_qubits != ir_b.num_qubits:
        return EquivalenceResult(
            False,
            "phase_poly",
            f"qubit count differs: {ir_a.num_qubits} vs {ir_b.num_qubits}",
        )

    n = ir_a.num_qubits
    poly_a = extract_phase_polynomial(ir_a)
    poly_b = extract_phase_polynomial(ir_b)

    # 1. Check affine transformation equality (A_a == A_b and b_a == b_b)
    if poly_a.parity_matrix != poly_b.parity_matrix or poly_a.shifts != poly_b.shifts:
        # Find a witness basis input where the output bitstrings disagree
        witness_x = 0
        diff_wire = 0
        for wire in range(n):
            pa = poly_a.parity_matrix[wire]
            pb = poly_b.parity_matrix[wire]
            sa = poly_a.shifts[wire]
            sb = poly_b.shifts[wire]
            if pa != pb or sa != sb:
                diff_wire = wire
                # If pa != pb, let bit k be where pa and pb differ
                diff_mask = pa ^ pb
                if diff_mask != 0:
                    k = (diff_mask & -diff_mask).bit_length() - 1
                    witness_x = 1 << k
                else:
                    witness_x = 0
                break

        out_a = sum(
            (((bin(poly_a.parity_matrix[w] & witness_x).count("1") + poly_a.shifts[w]) % 2) << w)
            for w in range(n)
        )
        out_b = sum(
            (((bin(poly_b.parity_matrix[w] & witness_x).count("1") + poly_b.shifts[w]) % 2) << w)
            for w in range(n)
        )
        cx = {
            "basis_input": witness_x,
            "input_bitstring": f"{witness_x:0{n}b}",
            "output_a": out_a,
            "output_b": out_b,
            "entry": (out_a, witness_x),
            "diff_wire": diff_wire,
            "reason": "affine parity transformation mismatch",
        }
        detail = (
            f"counterexample: basis input |{witness_x:0{n}b}> yields output |{out_a:0{n}b}> in A "
            f"vs |{out_b:0{n}b}> in B (affine parity mismatch)"
        )
        return EquivalenceResult(False, "phase_poly", detail, counterexample=cx)

    # 2. Check phase polynomial difference
    # Collect diff_map: parity_mask -> (coeff_a - coeff_b) mod 2pi
    diff_map: dict[int, float] = {}
    all_masks = set(dict(poly_a.phase_terms).keys()) | set(dict(poly_b.phase_terms).keys())
    dict_a = dict(poly_a.phase_terms)
    dict_b = dict(poly_b.phase_terms)

    for m in all_masks:
        d = _normalize_angle(dict_a.get(m, 0.0) - dict_b.get(m, 0.0))
        if abs(d) > atol:
            diff_map[m] = d

    if not diff_map:
        rel_phase = _normalize_angle(poly_a.global_phase - poly_b.global_phase)
        return EquivalenceResult(True, "phase_poly", f"global phase phi={rel_phase:.6f} rad")

    # Encode phase equality into Z3
    if _HAS_Z3:
        solver = _z3.Solver()
        solver.set("timeout", int(timeout * 1000))

        # Bits of the input state |x>
        x_bits = [_z3.Bool(f"x_{i}") for i in range(n)]

        # Phase difference expression
        phase_terms_expr = []
        for mask, d in diff_map.items():
            active_bits = [x_bits[i] for i in range(n) if (mask & (1 << i))]
            if len(active_bits) == 1:
                p_cond = active_bits[0]
            elif len(active_bits) == 2:
                p_cond = _z3.Xor(active_bits[0], active_bits[1])
            else:
                p_cond = functools.reduce(_z3.Xor, active_bits)
            d_val = _z3.RealVal(str(round(d, 8)))
            term_real = _z3.If(p_cond, d_val, _z3.RealVal("0"))
            phase_terms_expr.append(term_real)

        total_phase_diff = _z3.Sum(*phase_terms_expr) if phase_terms_expr else _z3.RealVal("0")

        two_pi_real = _z3.RealVal(str(round(_TWO_PI, 8)))
        tol_real = _z3.RealVal(str(round(atol, 8)))

        # Find x such that total_phase_diff != 2*pi*k for ALL integer multiples k
        max_sum = sum(abs(d) for d in diff_map.values())
        k_bound = max(int(math.ceil(max_sum / _TWO_PI)) + 1, 1)

        for kv in range(-k_bound, k_bound + 1):
            val_k = _z3.RealVal(str(round(kv * _TWO_PI, 8)))
            diff_k = total_phase_diff - val_k
            solver.add(_z3.Or(diff_k > tol_real, diff_k < -tol_real))

        check_status = solver.check()
        if check_status == _z3.unsat:
            rel_phase = _normalize_angle(poly_a.global_phase - poly_b.global_phase)
            return EquivalenceResult(True, "phase_poly", f"global phase phi={rel_phase:.6f} rad")
        elif check_status == _z3.sat:
            model = solver.model()
            witness_x = 0
            for i in range(n):
                val = model.eval(x_bits[i], model_completion=True)
                if _z3.is_true(val):
                    witness_x |= (1 << i)

            # Evaluate phase difference at witness_x
            eval_phase = sum(
                d for mask, d in diff_map.items()
                if (bin(mask & witness_x).count("1") % 2 == 1)
            )
            eval_phase_norm = _normalize_angle(eval_phase)
            cx = {
                "basis_input": witness_x,
                "input_bitstring": f"{witness_x:0{n}b}",
                "phase_diff": eval_phase_norm,
                "entry": (witness_x, witness_x),
                "reason": "phase polynomial angle mismatch",
            }
            detail = (
                f"counterexample: basis input |{witness_x:0{n}b}>: phase difference "
                f"{eval_phase_norm:.6f} rad != 0 mod 2pi"
            )
            return EquivalenceResult(False, "phase_poly", detail, counterexample=cx)
        else:
            return EquivalenceResult(False, "phase_poly", "SMT solver timeout/unknown")

    # Pure-Python fallback when Z3 is not installed:
    # Parity terms over F_2 are linearly independent basis functions;
    # pick the lowest mask with non-zero diff as witness
    first_mask, first_diff = next(iter(diff_map.items()))
    witness_x = 1 << ((first_mask & -first_mask).bit_length() - 1)
    eval_phase = sum(
        d for mask, d in diff_map.items()
        if (bin(mask & witness_x).count("1") % 2 == 1)
    )
    cx = {
        "basis_input": witness_x,
        "input_bitstring": f"{witness_x:0{n}b}",
        "phase_diff": _normalize_angle(eval_phase),
        "entry": (witness_x, witness_x),
        "reason": "phase polynomial angle mismatch",
    }
    detail = (
        f"counterexample: basis input |{witness_x:0{n}b}>: phase difference "
        f"{_normalize_angle(eval_phase):.6f} rad != 0 mod 2pi"
    )
    return EquivalenceResult(False, "phase_poly", detail, counterexample=cx)
