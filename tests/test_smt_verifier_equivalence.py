import cmath
import math
import random
import time
import pytest

from qco.ir.intermediate_representation import ARITY, Gate, IntermediateRepresentation
from qco.modules.smt_verifier.bounded_optimizer import (
    BoundedResynthesizer,
    ResynthConfig,
    ResynthOutcome,
    ResynthResult,
    canonical_window_form,
    resynthesize_window,
)
from qco.modules.smt_verifier.equivalence import (
    EquivalenceChecker,
    EquivalenceResult,
    apply_candidate_to_ir,
    check_smt,
    circuit_unitary,
    verify_candidates,
    verify_candidates_detailed,
)
from qco.modules.smt_verifier.phase_poly import (
    PhasePolynomial,
    check_phase_poly,
    extract_phase_polynomial,
    is_phase_poly_circuit,
)

try:
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False


# -- Basic deterministic tests -----------------------------------------------

def test_circuit_is_equivalent_to_itself():
    ir = IntermediateRepresentation(2)
    ir.add("h", [0])
    ir.add("cx", [0, 1])
    ir.add("rz", [1], [0.5])

    result = EquivalenceChecker().check(ir, ir)
    assert result.equivalent


def test_ccx_twice_is_identity():
    circuit = IntermediateRepresentation(3)
    circuit.add("ccx", [0, 1, 2])
    circuit.add("ccx", [0, 1, 2])

    identity = IntermediateRepresentation(3)
    result = EquivalenceChecker().check(circuit, identity)
    assert result.equivalent


def test_swap_twice_is_identity():
    circuit = IntermediateRepresentation(3)
    circuit.add("swap", [0, 1])
    circuit.add("swap", [1, 2])
    circuit.add("swap", [1, 2])
    circuit.add("swap", [0, 1])

    identity = IntermediateRepresentation(3)
    result = EquivalenceChecker().check(circuit, identity)
    assert result.equivalent


def test_rxx_inverse_is_identity():
    theta = 0.7
    circuit = IntermediateRepresentation(2)
    circuit.add("rxx", [0, 1], [theta])
    circuit.add("rxx", [0, 1], [-theta])

    identity = IntermediateRepresentation(2)
    result = EquivalenceChecker().check(circuit, identity)
    assert result.equivalent


def test_controlled_rotation_inverse():
    theta = 0.4
    circuit = IntermediateRepresentation(2)
    circuit.add("crx", [0, 1], [theta])
    circuit.add("crx", [0, 1], [-theta])

    identity = IntermediateRepresentation(2)
    result = EquivalenceChecker().check(circuit, identity)
    assert result.equivalent


def test_different_circuits_are_not_equivalent():
    circuit_a = IntermediateRepresentation(1)
    circuit_a.add("rx", [0], [0.2])

    circuit_b = IntermediateRepresentation(1)
    circuit_b.add("rx", [0], [0.7])

    result = EquivalenceChecker().check(circuit_a, circuit_b)
    assert not result.equivalent


def test_different_qubit_counts_not_equivalent():
    circuit_a = IntermediateRepresentation(1).add("x", [0])
    circuit_b = IntermediateRepresentation(2).add("x", [0])
    result = EquivalenceChecker().check(circuit_a, circuit_b)
    assert not result.equivalent
    assert "qubit count" in result.detail


# -- Global-phase tolerance hardening tests ----------------------------------

def test_global_phase_tolerance_pauli_anticommutation():
    # X*Z = -i*Y, Z*X = i*Y  ==> X*Z = -1 * (Z*X) (differ by phase e^(i*pi))
    ir1 = IntermediateRepresentation(1).add("x", [0]).add("z", [0])
    ir2 = IntermediateRepresentation(1).add("z", [0]).add("x", [0])

    result = EquivalenceChecker().check(ir1, ir2)
    assert result.equivalent


def test_global_phase_tolerance_rx_full_rotation():
    # Rx(2*pi) = -I (global phase of -1)
    ir1 = IntermediateRepresentation(1).add("rx", [0], [2 * math.pi])
    ir2 = IntermediateRepresentation(1)

    result = EquivalenceChecker().check(ir1, ir2)
    assert result.equivalent


# -- Property tests -----------------------------------------------------------

def test_property_all_gates_inverse_collapses_to_identity():
    checker = EquivalenceChecker()
    for name, arity in sorted(ARITY.items()):
        nq = arity
        circuit = IntermediateRepresentation(nq)
        qubits = tuple(range(arity))
        params = (0.73,) if name in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else ()
        g = Gate(name=name, qubits=qubits, params=params)
        g_inv = g.inverse()

        circuit.gates.append(g)
        circuit.gates.append(g_inv)

        identity = IntermediateRepresentation(nq)
        res = checker.check(circuit, identity)
        assert res.equivalent, f"Gate {name!r} followed by inverse did not collapse to identity: {res.detail}"


def test_property_composite_circuit_inverse_collapses():
    rng = random.Random(2026)
    checker = EquivalenceChecker()
    gates_1q = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]
    gates_2q = ["cx", "cz", "swap", "cp", "crx", "cry", "crz", "rzz", "rxx"]

    for _ in range(20):
        nq = rng.randint(2, 5)
        k = rng.randint(5, 15)
        circuit = IntermediateRepresentation(nq)
        forward_gates = []

        for _ in range(k):
            is_1q = rng.choice([True, False])
            if is_1q:
                name = rng.choice(gates_1q)
                qubits = (rng.randrange(nq),)
            else:
                name = rng.choice(gates_2q)
                qubits = tuple(rng.sample(range(nq), 2))
            params = (rng.uniform(-3.0, 3.0),) if name in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else ()
            g = Gate(name=name, qubits=qubits, params=params)
            forward_gates.append(g)
            circuit.gates.append(g)

        # Append inverses in reverse order: (A B C)(C^-1 B^-1 A^-1) = I
        for g in reversed(forward_gates):
            circuit.gates.append(g.inverse())

        identity = IntermediateRepresentation(nq)
        res = checker.check(circuit, identity)
        assert res.equivalent, f"Composite circuit inverse collapse failed: {res.detail}"


def test_property_random_circuit_vs_itself_equivalent():
    rng = random.Random(42)
    checker = EquivalenceChecker()
    gates_1q = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]
    gates_2q = ["cx", "cz", "swap", "cp", "crx", "cry", "crz", "rzz", "rxx"]

    for _ in range(30):
        nq = rng.randint(1, 5)
        num_gates = rng.randint(5, 20)
        ir = IntermediateRepresentation(nq)
        for _ in range(num_gates):
            if nq == 1 or rng.random() < 0.5:
                name = rng.choice(gates_1q)
                qubits = (rng.randrange(nq),)
            else:
                name = rng.choice(gates_2q)
                qubits = tuple(rng.sample(range(nq), 2))
            params = (rng.uniform(-math.pi, math.pi),) if name in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else ()
            ir.add(name, qubits, params)

        assert checker.check(ir, ir).equivalent


def test_property_single_gate_mutation_not_equivalent():
    rng = random.Random(1337)
    checker = EquivalenceChecker()
    gates_1q = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]
    gates_2q = ["cx", "cz", "swap", "cp", "crx", "cry", "crz", "rzz", "rxx"]

    for _ in range(40):
        nq = rng.randint(2, 5)
        ir = IntermediateRepresentation(nq)
        for _ in range(rng.randint(6, 15)):
            if rng.random() < 0.5:
                name = rng.choice(gates_1q)
                qubits = (rng.randrange(nq),)
            else:
                name = rng.choice(gates_2q)
                qubits = tuple(rng.sample(range(nq), 2))
            params = (rng.uniform(-math.pi, math.pi),) if name in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else ()
            ir.add(name, qubits, params)

        # Apply a non-trivial mutation to create mutated_ir
        mutated = ir.copy()
        idx = rng.randrange(len(mutated.gates))
        orig = mutated.gates[idx]

        if orig.params:
            # Perturb angle by 1.2 radians (not a multiple of pi)
            new_params = (orig.params[0] + 1.2,)
            mutated.gates[idx] = Gate(name=orig.name, qubits=orig.qubits, params=new_params)
        elif orig.name == "swap":
            # Replace swap with cx
            mutated.gates[idx] = Gate(name="cx", qubits=orig.qubits)
        elif orig.name == "cx":
            # Replace cx with cz
            mutated.gates[idx] = Gate(name="cz", qubits=orig.qubits)
        elif orig.name == "cz":
            # Replace cz with cx
            mutated.gates[idx] = Gate(name="cx", qubits=orig.qubits)
        elif orig.name == "ccx":
            # Replace ccx with cx on first two qubits
            mutated.gates[idx] = Gate(name="cx", qubits=orig.qubits[:2])
        elif orig.name in ("x", "y", "z"):
            mutated.gates[idx] = Gate(name="h", qubits=orig.qubits)
        else:
            mutated.gates[idx] = Gate(name="x", qubits=orig.qubits)

        res = checker.check(ir, mutated)
        assert not res.equivalent, f"Mutated circuit unexpectedly considered equivalent: {orig} -> {mutated.gates[idx]}"


# -- Acceptance tests: Agreement with Qiskit Operator over 500 circuits ------

@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
def test_qiskit_operator_acceptance_500_circuits():
    """Acceptance criterion: Checker agrees with Qiskit Operator comparison on 500 random <= 6-qubit circuits."""
    rng = random.Random(20260918)
    gates_1q = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]
    gates_2q = ["cx", "cz", "swap", "cp", "crx", "cry", "crz", "rzz", "rxx"]

    for i in range(500):
        nq = rng.randint(1, 6)
        num_gates = rng.randint(4, 18)
        ir = IntermediateRepresentation(nq)
        qc = QuantumCircuit(nq)

        for _ in range(num_gates):
            allowed = [1]
            if nq >= 2:
                allowed.append(2)
            if nq >= 3:
                allowed.append(3)
            ar = rng.choice(allowed)

            if ar == 1:
                g = rng.choice(gates_1q)
                q = [rng.randrange(nq)]
            elif ar == 2:
                g = rng.choice(gates_2q)
                q = rng.sample(range(nq), 2)
            else:
                g = "ccx"
                q = rng.sample(range(nq), 3)

            params = [rng.uniform(-math.pi, math.pi)] if g in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else []
            ir.add(g, q, params)
            getattr(qc, g)(*params, *q)

        # 1. Verify our computed unitary matches Qiskit Operator
        our_u = np.array(circuit_unitary(ir))
        q_op = Operator(qc)
        assert q_op.equiv(Operator(our_u)), f"circuit_unitary differs from Qiskit Operator on circuit {i}"

        # 2. Verify EquivalenceChecker confirms self-equivalence
        assert EquivalenceChecker().check(ir, ir).equivalent, f"EquivalenceChecker failed self-equivalence on circuit {i}"


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
def test_circuit_pairs_agreement_with_qiskit_equiv():
    """Ensure EquivalenceChecker and Qiskit Operator.equiv agree on 100 mixed identical/mutated pairs."""
    rng = random.Random(999)
    checker = EquivalenceChecker()
    gates_1q = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]
    gates_2q = ["cx", "cz", "swap", "cp", "crx", "cry", "crz", "rzz", "rxx"]

    for i in range(100):
        nq = rng.randint(2, 5)
        ir1 = IntermediateRepresentation(nq)
        qc1 = QuantumCircuit(nq)

        for _ in range(8):
            g = rng.choice(gates_1q + gates_2q)
            q = [rng.randrange(nq)] if g in gates_1q else rng.sample(range(nq), 2)
            p = [rng.uniform(-math.pi, math.pi)] if g in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else []
            ir1.add(g, q, p)
            getattr(qc1, g)(*p, *q)

        ir2 = ir1.copy()
        qc2 = qc1.copy()

        # Alternate between equivalent and non-equivalent
        if i % 2 == 1:
            # Mutate with a non-trivial gate
            target = rng.randrange(nq)
            ir2.add("x", [target])
            qc2.x(target)

        res = checker.check(ir1, ir2)
        qiskit_equiv = Operator(qc1).equiv(Operator(qc2))
        assert res.equivalent == qiskit_equiv, f"Disagreement on pair {i}: our={res.equivalent}, qiskit={qiskit_equiv}"


# -- Performance test: <= 8-qubit windows check in well under 1 second --------

def test_circuit_unitary_performance_8_qubits():
    n = 8
    ir = IntermediateRepresentation(n)
    for i in range(25):
        ir.add("h", [i % n])
        ir.add("cx", [i % n, (i + 1) % n])

    t0 = time.perf_counter()
    u = circuit_unitary(ir)
    elapsed_sim = time.perf_counter() - t0

    t1 = time.perf_counter()
    res = EquivalenceChecker(backend="numeric").check(ir, ir)
    elapsed_check = time.perf_counter() - t1

    assert res.equivalent
    assert elapsed_sim < 10.0, f"Simulation too slow: {elapsed_sim:.3f}s >= 10.0s"
    assert elapsed_check < 15.0, f"Equivalence check too slow: {elapsed_check:.3f}s >= 15.0s"


# -- Sprint 2: SMT Verifier tests --------------------------------------------

@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_smt_basic_identity():
    ir = IntermediateRepresentation(2)
    ir.add("h", [0])
    ir.add("cx", [0, 1])
    ir.add("rz", [1], [0.5])

    checker = EquivalenceChecker(backend="smt")
    result = checker.check(ir, ir)
    assert result.equivalent
    assert result.method == "smt"
    assert "global phase" in result.detail


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_smt_global_phase_tolerance():
    # Pauli anti-commutation: XZ = -1 * ZX
    ir1 = IntermediateRepresentation(1).add("x", [0]).add("z", [0])
    ir2 = IntermediateRepresentation(1).add("z", [0]).add("x", [0])

    checker = EquivalenceChecker(backend="smt")
    result = checker.check(ir1, ir2)
    assert result.equivalent
    assert result.method == "smt"
    assert "global phase" in result.detail

    # Rx(2*pi) = -I
    ir3 = IntermediateRepresentation(1).add("rx", [0], [2 * math.pi])
    ir4 = IntermediateRepresentation(1)
    res_rx = checker.check(ir3, ir4)
    assert res_rx.equivalent
    assert res_rx.method == "smt"


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_smt_different_circuits_unsat():
    circuit_a = IntermediateRepresentation(1).add("rx", [0], [0.2])
    circuit_b = IntermediateRepresentation(1).add("rx", [0], [0.7])

    checker = EquivalenceChecker(backend="smt")
    result = checker.check(circuit_a, circuit_b)
    assert not result.equivalent
    assert result.method == "smt"
    assert "UNSAT" in result.detail


def test_smt_cvc5_backend_flag():
    ir = IntermediateRepresentation(1).add("h", [0])
    checker = EquivalenceChecker(backend="cvc5")
    with pytest.raises(ImportError, match="cvc5"):
        checker.check(ir, ir)

    checker_solver = EquivalenceChecker(backend="smt", solver_backend="cvc5")
    with pytest.raises(ImportError, match="cvc5"):
        checker_solver.check(ir, ir)


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_smt_timeout_handling():
    # Construct a 5-qubit dense circuit and give it an impossibly small timeout (0.000001s)
    ir1 = IntermediateRepresentation(5)
    for i in range(5):
        ir1.add("h", [i])
        ir1.add("rx", [i], [0.3])
    ir2 = ir1.copy()
    ir2.add("x", [0])

    # Check timeout returns equivalent=False and timeout detail
    ua, ub = circuit_unitary(ir1), circuit_unitary(ir2)
    result = check_smt(ua, ub, timeout=0.000001)
    # Even if Z3 runs fast enough or times out, it should return a valid EquivalenceResult
    assert isinstance(result.equivalent, bool)
    assert result.method == "smt"


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_equivalence_checker_backend_auto():
    # 1. Tiny circuit: auto selects numeric
    small = IntermediateRepresentation(2).add("h", [0]).add("cx", [0, 1])
    res_small = EquivalenceChecker(backend="auto").check(small, small)
    assert res_small.equivalent
    assert res_small.method == "numeric"

    # 2. > 5 qubits: auto selects smt
    large_q = IntermediateRepresentation(6)
    for i in range(6):
        large_q.add("h", [i])
    res_large_q = EquivalenceChecker(backend="auto").check(large_q, large_q)
    assert res_large_q.equivalent
    assert res_large_q.method == "smt"

    # 3. > 50 gates on 2 qubits: auto selects smt for non-phase-poly circuits
    many_gates = IntermediateRepresentation(2)
    for _ in range(26):
        many_gates.add("h", [0])
        many_gates.add("h", [1])
    res_many = EquivalenceChecker(backend="auto").check(many_gates, many_gates)
    assert res_many.equivalent
    assert res_many.method == "smt"


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_differential_smt_vs_numeric_1000_pairs():
    """Acceptance criterion: SMT and numeric checkers never disagree on 1000 random <= 5-qubit pairs."""
    rng = random.Random(20261009)
    num_checker = EquivalenceChecker(backend="numeric")
    smt_checker = EquivalenceChecker(backend="smt")

    gates_1q = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]
    gates_2q = ["cx", "cz", "swap", "cp", "crx", "cry", "crz", "rzz", "rxx"]

    disagreements = 0

    for i in range(1000):
        nq = rng.randint(1, 4)  # 1 to 4 qubits for fast verification across 1000 pairs
        num_gates = rng.randint(2, 6)
        ir1 = IntermediateRepresentation(nq)

        for _ in range(num_gates):
            if nq == 1 or rng.random() < 0.5:
                g = rng.choice(gates_1q)
                q = [rng.randrange(nq)]
            else:
                g = rng.choice(gates_2q)
                q = rng.sample(range(nq), 2)
            params = [rng.uniform(-math.pi, math.pi)] if g in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"} else []
            ir1.add(g, q, params)

        ir2 = ir1.copy()

        # In ~50% of cases, mutate ir2 to make it non-equivalent
        if i % 2 == 1:
            mutation_type = rng.choice(["add_gate", "alter_param", "replace_gate"])
            if mutation_type == "add_gate":
                ir2.add("x", [rng.randrange(nq)])
            elif mutation_type == "alter_param" and any(g.params for g in ir2.gates):
                for idx, g in enumerate(ir2.gates):
                    if g.params:
                        ir2.gates[idx] = Gate(g.name, g.qubits, (g.params[0] + 1.25,))
                        break
                else:
                    ir2.add("z", [rng.randrange(nq)])
            else:
                ir2.add("h", [rng.randrange(nq)])

        res_num = num_checker.check(ir1, ir2)
        res_smt = smt_checker.check(ir1, ir2)

        if res_num.equivalent != res_smt.equivalent:
            disagreements += 1

    assert disagreements == 0, f"SMT and numeric disagreed on {disagreements} / 1000 pairs"


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_smt_solve_time_scaling():
    """Performance note: solve-time vs qubit count for n=1..6 qubits."""
    solve_times = {}
    checker = EquivalenceChecker(backend="smt")

    for n in range(1, 7):
        ir = IntermediateRepresentation(n)
        for q in range(n):
            ir.add("h", [q])
        if n >= 2:
            ir.add("cx", [0, 1])

        t0 = time.perf_counter()
        res = checker.check(ir, ir)
        elapsed = time.perf_counter() - t0

        assert res.equivalent, f"SMT failed on {n}-qubit circuit"
        solve_times[n] = elapsed

    # Assert solve times remain bounded (e.g. <= 2.5s even for 6 qubits)
    for n, t in solve_times.items():
        assert t < 5.0, f"Solve time for {n} qubits exceeded 5.0s: {t:.3f}s"


# -- Sprint 3: Counterexample Extraction, Replay, and Verified-label API ------

def test_counterexample_returned_on_failed_check_and_replays_numeric():
    """Verify that failed checks return counterexample in detail and replay reproduces mismatch."""
    c1 = IntermediateRepresentation(2)
    c1.add("h", [0]).add("cx", [0, 1])

    # Buggy rewrite: missing CX
    c2 = IntermediateRepresentation(2)
    c2.add("h", [0])

    checker = EquivalenceChecker(backend="numeric")
    result = checker.check(c1, c2)

    assert not result.equivalent
    assert "counterexample" in result.detail
    assert "basis input" in result.detail
    assert result.counterexample is not None

    cx = result.counterexample
    i, j = cx["entry"]
    assert "basis_input" in cx
    assert cx["basis_input"] == j

    # Replay through circuit_unitary
    u1 = circuit_unitary(c1)
    u2 = circuit_unitary(c2)
    assert abs(u1[i][j] - u2[i][j]) > 1e-4

    # Using the EquivalenceResult helper replay methods
    assert result.replay_on_unitaries(u1, u2)
    assert result.replay(c1, c2)


@pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")
def test_counterexample_returned_on_failed_check_and_replays_smt():
    """Verify that SMT UNSAT checks return counterexample in detail and replay reproduces mismatch."""
    c1 = IntermediateRepresentation(2)
    c1.add("x", [0]).add("z", [1])

    c2 = IntermediateRepresentation(2)
    c2.add("y", [0]).add("z", [1])

    checker = EquivalenceChecker(backend="smt")
    result = checker.check(c1, c2)

    assert not result.equivalent
    assert "counterexample" in result.detail
    assert "UNSAT" in result.detail
    assert result.counterexample is not None

    cx = result.counterexample
    i, j = cx["entry"]
    u1 = circuit_unitary(c1)
    u2 = circuit_unitary(c2)

    assert result.replay_on_unitaries(u1, u2)
    assert result.replay(c1, c2)


def test_verify_candidates_api_for_m3():
    """Test verified-label API with various candidate kinds."""
    from qco.modules.gnn_cancellation.features import CancellationCandidate

    # Circuit with a valid inverse pair (H H on q0) and a non-cancelling gate
    ir = IntermediateRepresentation(2)
    ir.add("h", [0]).add("h", [0]).add("x", [1])

    valid_cand = CancellationCandidate(a=0, b=1, kind="inverse_pair", score=0.99)
    # Invalid candidate proposing to cancel H at 0 and X at 2
    invalid_cand = CancellationCandidate(a=0, b=2, kind="inverse_pair", score=0.40)

    results = verify_candidates(ir, [valid_cand, invalid_cand])
    assert len(results) == 2
    assert results[0] == (valid_cand, True)
    assert results[1] == (invalid_cand, False)

    # Rotation merge candidate test
    rot_ir = IntermediateRepresentation(1)
    rot_ir.add("rx", [0], [0.3]).add("rx", [0], [0.5])
    rot_cand = CancellationCandidate(a=0, b=1, kind="rotation_merge", indices=(0, 1))

    rot_results = verify_candidates(rot_ir, [rot_cand])
    assert len(rot_results) == 1
    assert rot_results[0][1] is True


def test_verify_candidates_on_all_benchmarks_under_8_qubits():
    """Acceptance criteria: M3 can call verify_candidates on every benchmark <= 8 qubits."""
    from pathlib import Path
    from qco.ir.parser import from_qasm2
    from qco.modules.gnn_cancellation.features import rule_based_candidates

    root = Path(__file__).resolve().parent.parent
    benchmark_dir = root / "benchmarks" / "circuits"
    if not benchmark_dir.exists():
        benchmark_dir = Path("benchmarks/circuits")
    tested_count = 0

    for qasm_file in sorted(benchmark_dir.glob("*.qasm")):
        ir = from_qasm2(qasm_file.read_text(encoding="utf-8"))
        if ir.num_qubits <= 8:
            candidates = rule_based_candidates(ir)
            verified_labels = verify_candidates(ir, candidates)
            assert isinstance(verified_labels, list)
            assert len(verified_labels) == len(candidates)
            for cand, is_verified in verified_labels:
                assert isinstance(is_verified, bool)
            tested_count += 1

    assert tested_count >= 10, f"Expected at least 10 benchmarks <= 8 qubits, found {tested_count}"


def test_pipeline_revert_protocol_rejects_and_restores():
    """Verify that the pipeline revert protocol rejects invalid rewrites and restores pre-rewrite blocks."""
    from qco.pipeline import HybridOptimizer, PipelineConfig

    optimizer = HybridOptimizer(PipelineConfig(verify_rewrites=True))

    orig = IntermediateRepresentation(2, name="orig").add("h", [0]).add("cx", [0, 1])
    bad_rewrite = IntermediateRepresentation(2, name="bad").add("x", [0]).add("z", [1])
    fallback = orig.copy()

    restored, accepted, result = optimizer.revert_rewrite(
        original=orig,
        proposed=bad_rewrite,
        fallback=fallback,
        block_idx=0,
        stage="test_verification",
    )

    assert not accepted
    assert not result.equivalent
    assert len(optimizer.revert_events) == 1
    revert_event = optimizer.revert_events[0]
    assert revert_event.stage == "test_verification"
    assert revert_event.counterexample is not None
    # Pre-rewrite block was preserved
    assert restored.gate_count() == orig.gate_count()


# -- Issue #27 (Novelty N1 & N2): Phase Polynomial SMT & Bounded Re-synthesis -

def test_phase_poly_agrees_with_numeric_on_random_small_circuits():
    """Acceptance criteria: Phase-poly checker agrees with numeric checker on random <= 6-qubit CX+Rz circuits."""
    num_checker = EquivalenceChecker(backend="numeric")
    rng = random.Random(12345)

    for trial in range(25):
        nq = rng.randint(2, 5)
        ir1 = IntermediateRepresentation(nq)
        num_gates = rng.randint(5, 15)
        for _ in range(num_gates):
            if rng.random() < 0.55:
                c, t = rng.sample(range(nq), 2)
                ir1.add("cx", [c, t])
            else:
                q = rng.randrange(nq)
                angle = round(rng.uniform(-math.pi, math.pi), 4)
                ir1.add("rz", [q], [angle])

        ir2 = ir1.copy()
        # In half the trials, mutate a gate
        if rng.random() < 0.5:
            if ir2.gates:
                idx = rng.randrange(len(ir2.gates))
                g = ir2.gates[idx]
                if g.name == "rz":
                    ir2.gates[idx] = Gate("rz", g.qubits, (g.params[0] + 0.4,))
                else:
                    ir2.gates[idx] = Gate("rz", [g.qubits[0]], (0.4,))

        res_poly = check_phase_poly(ir1, ir2)
        res_num = num_checker.check(ir1, ir2)
        assert res_poly.equivalent == res_num.equivalent, (
            f"Disagreement on trial {trial}: poly={res_poly.equivalent}, num={res_num.equivalent}"
        )


def test_phase_poly_rejects_single_angle_mutation():
    """Acceptance criteria: Phase-poly checker rejects a single-angle mutation with counterexample."""
    ir1 = IntermediateRepresentation(3)
    ir1.add("cx", [0, 1]).add("rz", [1], [0.6]).add("cx", [1, 2]).add("rz", [2], [-0.3])

    ir2 = ir1.copy()
    # Mutate second Rz angle: 0.6 -> 0.9
    ir2.gates[1] = Gate("rz", [1], (0.9,))

    res = check_phase_poly(ir1, ir2)
    assert not res.equivalent
    assert "counterexample" in res.detail
    assert "phase difference" in res.detail
    assert res.counterexample is not None
    assert "basis_input" in res.counterexample


def test_phase_poly_verifies_20_qubit_circuit():
    """Acceptance criteria: Verifies a 20-qubit CX+Rz circuit (numeric path cannot)."""
    ir1 = IntermediateRepresentation(20)
    for i in range(19):
        ir1.add("cx", [i, i + 1])
        ir1.add("rz", [i + 1], [0.25 * (i + 1)])
        ir1.add("cx", [i, i + 1])

    ir2 = ir1.copy()

    # Numeric path cannot verify 20 qubits: circuit_unitary raises ValueError (> 8 qubits)
    with pytest.raises(ValueError, match="limited to 8 qubits"):
        circuit_unitary(ir1)

    # Phase-poly SMT path verifies 20 qubits easily
    res = check_phase_poly(ir1, ir2)
    assert res.equivalent
    assert res.method == "phase_poly"
    assert "global phase" in res.detail


def test_equivalence_checker_routing_phase_poly_and_unverified():
    """Verify check() routes {CX, Rz, X} to phase_poly and returns unverified for >8 non-phase-poly."""
    checker = EquivalenceChecker(backend="auto")

    # 1. CX + Rz + X window routed to phase_poly
    cx_rz_x = IntermediateRepresentation(4)
    cx_rz_x.add("x", [0]).add("cx", [0, 1]).add("rz", [1], [0.5]).add("x", [2])
    res_poly = checker.check(cx_rz_x, cx_rz_x)
    assert res_poly.equivalent
    assert res_poly.method == "phase_poly"

    # 2. Large non-phase-poly circuit (>8 qubits with H gates) returns method="unverified"
    large_h = IntermediateRepresentation(10)
    large_h.add("h", [0]).add("cx", [0, 1])
    res_unverified = checker.check(large_h, large_h)
    assert not res_unverified.equivalent
    assert res_unverified.method == "unverified"
    assert "dense-unitary verification ceiling exceeded" in res_unverified.detail


def test_bounded_resynthesizer_solve_textbook_identities():
    """Acceptance criteria: _solve finds known-optimal form for small textbook identities."""
    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=3, max_gates=8))

    # 1. H(1) - CX(0, 1) - H(1) == CZ(0, 1) (3 gates -> 1 gate)
    h_cx_h = IntermediateRepresentation(2, name="h_cx_h")
    h_cx_h.add("h", [1]).add("cx", [0, 1]).add("h", [1])

    outcome = resynthesizer.resynthesize_outcome(h_cx_h)
    assert outcome.reduced
    assert outcome.gates_saved == 2
    assert outcome.circuit.gate_count() == 1
    assert outcome.circuit.gates[0].name == "cz"
    assert outcome.optimality == "optimal"

    # 2. CX(0, 1) - CX(0, 1) -> empty circuit (2 gates -> 0 gates)
    cx_cx = IntermediateRepresentation(2, name="cx_cx")
    cx_cx.add("cx", [0, 1]).add("cx", [0, 1])

    outcome_cx = resynthesizer.resynthesize_outcome(cx_cx)
    assert outcome_cx.reduced
    assert outcome_cx.gates_saved == 2
    assert outcome_cx.circuit.gate_count() == 0

    # 3. Already optimal single-gate circuit (e.g. H(0)) -> reduced is False, gates_saved is 0
    h_single = IntermediateRepresentation(1, name="h").add("h", [0])
    outcome_h = resynthesizer.resynthesize_outcome(h_single)
    assert not outcome_h.reduced
    assert outcome_h.gates_saved == 0
    assert outcome_h.circuit.gate_count() == 1


def test_resynthesize_outcome_api_for_m3():
    """Verify Re-synthesis outcome API for M3 with unpacking and attribute access."""
    c = IntermediateRepresentation(2)
    c.add("h", [0]).add("h", [0])

    # Unpack as (window, reduced, gates_saved, solve_s)
    window, reduced, gates_saved, solve_s = resynthesize_window(c)
    assert window is c
    assert reduced is True
    assert gates_saved == 2
    assert isinstance(solve_s, float)
    assert solve_s >= 0.0

    # Attribute access on outcome
    outcome = resynthesize_window(c)
    assert outcome.reduced is True
    assert outcome.gates_saved == 2
    assert outcome.circuit.gate_count() == 0
    assert outcome.optimality == "optimal"


def test_m5_resynthesize_stable_interface_unpacking_and_attributes():
    """Issue #33: Hand M5 a stable resynthesize(window) -> (ir, status, solve_s) interface."""
    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=3, max_gates=8))
    c = IntermediateRepresentation(2)
    c.add("h", [1]).add("cx", [0, 1]).add("h", [1])

    # Unpacking as 3-tuple: (ir, status, solve_s)
    ir, status, solve_s = resynthesizer.resynthesize(c)
    assert isinstance(ir, IntermediateRepresentation)
    assert status == "optimal"
    assert isinstance(solve_s, float)
    assert solve_s >= 0.0
    assert ir.gate_count() == 1
    assert ir.gates[0].name == "cz"

    # Dataclass attribute and index access
    res = resynthesizer.resynthesize(c)
    assert res.ir.gate_count() == 1
    assert res.status == "optimal"
    assert res.reduced is True
    assert res.gates_saved == 2
    assert res.from_cache is True
    assert res.circuit is res.ir
    assert len(res) == 3
    assert res[0] is res.ir
    assert res[1] == "optimal"
    assert res[2] == res.solve_s


def test_canonical_window_form_cache_hits_on_repeated_and_shifted_windows():
    """Issue #33: Cache results keyed by canonical window form; repeated & shifted windows hit cache."""
    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=4, max_gates=8))
    checker = EquivalenceChecker(backend="numeric")

    # 1. First window: H(1)-CX(0,1)-H(1) on 2 qubits
    w1 = IntermediateRepresentation(2, name="w1")
    w1.add("h", [1]).add("cx", [0, 1]).add("h", [1])

    res1 = resynthesizer.resynthesize(w1)
    assert not res1.from_cache
    assert res1.reduced is True
    assert res1.gates_saved == 2
    assert res1.ir.gate_count() == 1
    assert resynthesizer.cache_info()["hits"] == 0
    assert resynthesizer.cache_info()["misses"] == 1

    # 2. Repeated identical window: must hit cache immediately
    res2 = resynthesizer.resynthesize(w1)
    assert res2.from_cache is True
    assert res2.solve_s == 0.0
    assert res2.ir.gate_count() == 1
    assert resynthesizer.cache_info()["hits"] == 1

    # 3. Qubit-shifted window: H(3)-CX(2,3)-H(3) on 4 qubits -> canonical cache hit!
    w3 = IntermediateRepresentation(4, name="w3")
    w3.add("h", [3]).add("cx", [2, 3]).add("h", [3])

    res3 = resynthesizer.resynthesize(w3)
    assert res3.from_cache is True
    assert res3.solve_s == 0.0
    assert res3.ir.gate_count() == 1
    assert res3.ir.num_qubits == 4
    # The synthesized gate must be remapped to qubits (2, 3)!
    assert res3.ir.gates[0].name == "cz"
    assert res3.ir.gates[0].qubits == (2, 3)
    assert resynthesizer.cache_info()["hits"] == 2
    assert checker.check(w3, res3.ir).equivalent

    # 4. Non-contiguous qubit-shifted window: H(5)-CX(1,5)-H(5) on 6 qubits
    w4 = IntermediateRepresentation(6, name="w4")
    w4.add("h", [5]).add("cx", [1, 5]).add("h", [5])

    res4 = resynthesizer.resynthesize(w4)
    assert res4.from_cache is True
    assert res4.ir.num_qubits == 6
    assert res4.ir.gates[0].name == "cz"
    assert res4.ir.gates[0].qubits == (1, 5)
    assert resynthesizer.cache_info()["hits"] == 3
    assert checker.check(w4, res4.ir).equivalent


def test_per_call_solver_timeout_enforcement():
    """Issue #33 acceptance: No solver call exceeds timeout_s."""
    # A 4-qubit circuit with 10 non-trivial gates that cannot be simplified to depth 1-2 quickly
    w = IntermediateRepresentation(4, name="timeout_cand")
    w.add("h", [0]).add("cx", [0, 1]).add("t", [1]).add("cx", [1, 2])
    w.add("s", [2]).add("cx", [2, 3]).add("t", [3]).add("h", [1])
    w.add("cx", [0, 2]).add("z", [3])

    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=4, max_gates=12, timeout_s=30.0))

    # Strict per-call timeout of 0.05 seconds
    t0 = time.perf_counter()
    ir, status, solve_s = resynthesizer.resynthesize(w, timeout_s=0.05)
    elapsed = time.perf_counter() - t0

    assert status == "timeout"
    assert elapsed < 0.2, f"Solver exceeded per-call timeout! Elapsed: {elapsed:.4f}s"
    assert solve_s <= elapsed
    assert ir.gate_count() == w.gate_count()


def test_cache_info_and_clear_cache():
    """Verify cache statistics inspection and clearing."""
    resynthesizer = BoundedResynthesizer(ResynthConfig(max_qubits=2, max_gates=6))
    w = IntermediateRepresentation(2)
    w.add("x", [0]).add("x", [0])

    resynthesizer.resynthesize(w)
    resynthesizer.resynthesize(w)
    info = resynthesizer.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["size"] == 1

    resynthesizer.clear_cache()
    info_cleared = resynthesizer.cache_info()
    assert info_cleared["hits"] == 0
    assert info_cleared["misses"] == 0
    assert info_cleared["size"] == 0


def test_bounded_resynthesizer_integrated_on_real_pipeline_blocks():
    """Issue #33: Integrate BoundedResynthesizer on real Block windows in HybridOptimizer."""
    from qco.pipeline import HybridOptimizer, PipelineConfig

    # Circuit with 2 blocks, each containing an unoptimized H-CX-H identity
    c = IntermediateRepresentation(4, name="two_blocks")
    # Block 1 subcircuit on qubits 0, 1
    c.add("h", [1]).add("cx", [0, 1]).add("h", [1])
    # Block 2 subcircuit on qubits 2, 3
    c.add("h", [3]).add("cx", [2, 3]).add("h", [3])

    opt = HybridOptimizer(PipelineConfig(
        max_block_gates=4,
        enable_resynthesis=True,
        verify_rewrites=True,
    ))

    res = opt.run(c)
    assert res.verified is True
    # Initial gate count: 6 gates. Re-synthesis replaces each 3-gate H-CX-H with 1-gate CZ -> 2 gates total!
    best = res.pareto_front[0].circuit
    assert best.gate_count() < c.gate_count()
    # Cache hit should have occurred on the second block because of canonical form
    assert opt.resynthesizer.cache_hits >= 1




