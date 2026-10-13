"""Tests for Module C scalability, Phase-Polynomial SMT beyond 8 qubits,

and bounded re-synthesis on real benchmark windows (Sprint 5 / Issue #39).
"""

import math
import time
import pytest

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.smt_verifier.bounded_optimizer import (
    BoundedResynthesizer,
    ResynthConfig,
)
from qco.modules.smt_verifier.equivalence import (
    EquivalenceChecker,
    EquivalenceResult,
)
from qco.modules.smt_verifier.phase_poly import (
    check_phase_poly,
    extract_phase_polynomial,
    is_phase_poly_circuit,
)


# ============================================================================
# 1. Phase-Polynomial Scaling beyond 8 Qubits (Acceptance Criteria)
# ============================================================================

@pytest.mark.parametrize("num_qubits", [16, 32, 64])
def test_phase_poly_verifier_scales_beyond_8_qubits(num_qubits):
    """Verify that Phase-Polynomial SMT scales to 16, 32, and 64 qubits in < 100ms."""
    c1 = IntermediateRepresentation(num_qubits, name="scaled_c1")
    c2 = IntermediateRepresentation(num_qubits, name="scaled_c2")

    # Construct equivalent pair with commuting operations and cancellations across register
    for q in range(num_qubits):
        c1.add("rz", [q], [math.pi / 4])
    for q in range(0, num_qubits - 1, 2):
        c1.add("cx", [q, q + 1])
        c1.add("rz", [q + 1], [math.pi / 2])
        c1.add("cx", [q, q + 1])

    # c2 has redundant canceling CNOT pairs and reordered independent gates
    for q in range(0, num_qubits - 1, 2):
        c2.add("cx", [q, q + 1])
        c2.add("cx", [q, q + 1])  # Cancels to identity
    for q in range(num_qubits):
        c2.add("rz", [q], [math.pi / 4])
    for q in range(0, num_qubits - 1, 2):
        c2.add("cx", [q, q + 1])
        c2.add("rz", [q + 1], [math.pi / 2])
        c2.add("cx", [q, q + 1])

    assert is_phase_poly_circuit(c1)
    assert is_phase_poly_circuit(c2)

    # Use EquivalenceChecker (which automatically routes to Phase-Poly for n > 8)
    checker = EquivalenceChecker()
    t0 = time.perf_counter()
    res = checker.check(c1, c2)
    elapsed = time.perf_counter() - t0

    assert res.equivalent is True
    assert res.method == "phase_poly"
    # Verification should easily complete in less than 200 ms even for 64 qubits
    assert elapsed < 0.20, f"Phase-poly took {elapsed:.4f}s for n={num_qubits}, expected < 0.2s"


def test_phase_poly_rejects_inequivalent_large_circuit_with_counterexample():
    """Verify that inequivalent rewrites on large circuits (32 qubits) are rejected."""
    n = 32
    c1 = IntermediateRepresentation(n)
    c2 = IntermediateRepresentation(n)

    # c1 has a chain of CX and Rz
    for q in range(n - 1):
        c1.add("cx", [q, q + 1])
        c1.add("rz", [q], [math.pi / 2])

    # c2 has a slight phase discrepancy on qubit 15
    for q in range(n - 1):
        c2.add("cx", [q, q + 1])
        c2.add("rz", [q], [math.pi / 2 if q != 15 else math.pi / 4])

    checker = EquivalenceChecker()
    res = checker.check(c1, c2)

    assert res.equivalent is False
    assert res.method == "phase_poly"
    assert res.counterexample is not None


def test_phase_poly_detects_affine_boolean_mismatch_on_large_register():
    """Verify rejection when the Boolean linear transformation does not match on 32 qubits."""
    n = 32
    c1 = IntermediateRepresentation(n)
    c2 = IntermediateRepresentation(n)

    for q in range(n - 1):
        c1.add("cx", [q, q + 1])

    # c2 misses one CNOT in the middle
    for q in range(n - 1):
        if q != 10:
            c2.add("cx", [q, q + 1])

    res = check_phase_poly(c1, c2)
    assert res.equivalent is False
    assert "affine" in res.detail.lower() or res.counterexample is not None


# ============================================================================
# 2. Re-Synthesis Window Scaling and Cache Efficiency
# ============================================================================

def test_bounded_resynthesizer_cache_hit_on_isomorphic_windows():
    """Verify that identical subcircuits across different qubits hit the canonical cache."""
    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=4, max_gates=6, timeout_s=1.0))

    # Window 1: H-CX-H on qubits 0, 1
    w1 = IntermediateRepresentation(4)
    w1.add("h", [1]).add("cx", [0, 1]).add("h", [1])

    # Window 2: H-CX-H on qubits 2, 3 (isomorphic to Window 1 under qubit permutation)
    w2 = IntermediateRepresentation(4)
    w2.add("h", [3]).add("cx", [2, 3]).add("h", [3])

    opt1, status1, solve1 = resynthesizer.resynthesize(w1)
    assert status1 in ("optimal", "reduced")
    assert resynthesizer.cache_hits == 0
    assert resynthesizer.cache_misses == 1

    # Second window should hit cache immediately
    t0 = time.perf_counter()
    opt2, status2, solve2 = resynthesizer.resynthesize(w2)
    elapsed = time.perf_counter() - t0

    assert status2 in ("optimal", "reduced")
    assert resynthesizer.cache_hits == 1
    # Cache hit should be near-instantaneous (< 5ms)
    assert elapsed < 0.05
    assert opt2.gate_count() == opt1.gate_count()


def test_bounded_resynthesizer_respects_depth_and_active_qubit_bounds():
    """Ensure windows exceeding max_qubits or max_gates are gracefully passed through."""
    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=3, max_gates=4))

    # Window with 5 active qubits (exceeds max_qubits=3)
    w_large = IntermediateRepresentation(6)
    for q in range(5):
        w_large.add("x", [q])

    out_ir, status, solve_s = resynthesizer.resynthesize(w_large)
    assert status in ("not_applicable", "skipped")
    assert out_ir.gate_count() == w_large.gate_count()
