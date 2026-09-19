import cmath
import math
import random
import time
import pytest

from qco.ir.intermediate_representation import ARITY, Gate, IntermediateRepresentation
from qco.modules.smt_verifier.equivalence import EquivalenceChecker, circuit_unitary

try:
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


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
    res = EquivalenceChecker().check(ir, ir)
    elapsed_check = time.perf_counter() - t1

    assert res.equivalent
    assert elapsed_sim < 0.5, f"Simulation too slow: {elapsed_sim:.3f}s >= 0.5s"
    assert elapsed_check < 0.8, f"Equivalence check too slow: {elapsed_check:.3f}s >= 0.8s"


