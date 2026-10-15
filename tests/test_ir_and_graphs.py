"""Phase 2 smoke tests: IR, parser, DAG, hypergraph, partitioning."""

import math

from qco.ir import IntermediateRepresentation, from_cirq, from_qasm2, from_qiskit
from qco.graphs import build_dag, build_hypergraph, partition_circuit


def test_ir_metrics():
    ir = IntermediateRepresentation(3)
    ir.add("h", [0]).add("cx", [0, 1]).add("cx", [1, 2])
    assert ir.gate_count() == 3
    assert ir.two_qubit_gate_count() == 2
    assert ir.depth() == 3


def test_qasm_roundtrip():
    src = IntermediateRepresentation(2, name="c")
    src.add("h", [0]).add("cx", [0, 1]).add("rz", [1], [math.pi / 2])
    parsed = from_qasm2(src.to_qasm2())
    assert parsed.num_qubits == 2
    assert [g.name for g in parsed] == ["h", "cx", "rz"]
    assert abs(parsed.gates[2].params[0] - math.pi / 2) < 1e-9


def test_dag_dependencies():
    ir = IntermediateRepresentation(2)
    ir.add("h", [0]).add("x", [1]).add("cx", [0, 1])
    dag = build_dag(ir)
    assert dag.succ[0] == {2}
    assert dag.succ[1] == {2}
    assert dag.critical_path_length() == 2
    assert dag.topological_order()[-1] == 2


def test_hypergraph_only_multiqubit_edges():
    ir = IntermediateRepresentation(3)
    ir.add("h", [0]).add("cx", [0, 1]).add("ccx", [0, 1, 2])
    hg = build_hypergraph(ir)
    assert len(hg.hyperedges) == 2
    assert hg.qubit_degrees()[2] == 6  # weighted by CX-cost: ccx = 6


def test_partition_covers_all_gates():
    ir = IntermediateRepresentation(4)
    for i in range(30):
        ir.add("h", [i % 4])
    blocks = partition_circuit(ir, max_block_gates=8)
    assert sum(b.ir.gate_count() for b in blocks) == 30


# --- #4: front ends, gate set, commutation table --------------------------------

import glob
import itertools

import pytest

from qco.graphs import commutes
from qco.ir import Gate, parse
from qco.ir.intermediate_representation import ARITY, PARAMETRIC
from qco.modules.smt_verifier.equivalence import circuit_unitary

BENCHMARKS = sorted(glob.glob("benchmarks/circuits/*.qasm"))


def _allclose(a, b, tol=1e-9):
    """Elementwise closeness for flat or nested (matrix) sequences; no numpy in CI."""
    rows_a = [list(r) for r in a] if a and hasattr(a[0], "__len__") else [list(a)]
    rows_b = [list(r) for r in b] if b and hasattr(b[0], "__len__") else [list(b)]
    return len(rows_a) == len(rows_b) and all(
        len(ra) == len(rb) and all(abs(x - y) <= tol for x, y in zip(ra, rb))
        for ra, rb in zip(rows_a, rows_b)
    )


def _same_gates(a, b):
    return a.num_qubits == b.num_qubits and [(g.name, g.qubits) for g in a] == [
        (g.name, g.qubits) for g in b
    ] and all(_allclose(g.params, h.params) for g, h in zip(a, b))


def test_every_ir_gate_has_inverse_and_checker_matrix():
    for name, n in ARITY.items():
        g = Gate(name, tuple(range(n)), (0.3,) if name in PARAMETRIC else ())
        ir = IntermediateRepresentation(n, [g, g.inverse()])
        eye = [[1.0 if i == j else 0.0 for j in range(2**n)] for i in range(2**n)]
        assert _allclose(circuit_unitary(ir), eye), name


def test_qasm_multi_register_aliases_and_custom_gate_bodies():
    src = """OPENQASM 2.0;
include "qelib1.inc";
gate foo a, b {
  cx a, b;
}
qreg a[2];
qreg b[1];
creg c[1];
u1(pi/2) a[1];
cnot a[0], b[0];
barrier a[0], b[0];
measure b[0] -> c[0];
"""
    ir = from_qasm2(src)
    assert ir.num_qubits == 3
    assert [(g.name, g.qubits) for g in ir] == [("p", (1,)), ("cx", (0, 2))]


def test_unsupported_gate_raises():
    with pytest.raises(ValueError, match="not supported"):
        from_qasm2('OPENQASM 2.0;\nqreg q[1];\nsx q[0];\n')


@pytest.mark.parametrize("path", BENCHMARKS)
def test_benchmark_qasm_roundtrip(path):
    ir = parse(path)
    assert _same_gates(ir, from_qasm2(ir.to_qasm2()))


@pytest.mark.parametrize("path", BENCHMARKS)
def test_benchmark_loads_from_qiskit(path):
    qiskit = pytest.importorskip("qiskit")
    from qiskit import qasm2

    qc = qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    assert _same_gates(parse(qc), parse(path))


def test_qiskit_multi_register_and_param_binding():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter

    th = Parameter("th")
    qc = QuantumCircuit(QuantumRegister(2, "a"), QuantumRegister(1, "b"))
    qc.h(0)
    qc.rz(th, 2)
    qc.cx(1, 2)
    qc.measure_all()
    with pytest.raises(ValueError, match="unbound"):
        parse(qc)
    ir = from_qiskit(qc, {th: 0.5})
    assert [(g.name, g.qubits, g.params) for g in ir] == [
        ("h", (0,), ()), ("rz", (2,), (0.5,)), ("cx", (1, 2), ()),
    ]


def test_cirq_import_and_param_resolution():
    cirq = pytest.importorskip("cirq")
    import sympy

    q = cirq.LineQubit.range(3)
    th = sympy.Symbol("th")
    c = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[2]), cirq.rz(th).on(q[1]), cirq.CCX(*q)])
    with pytest.raises(ValueError, match="unbound"):
        from_cirq(c)
    ir = from_cirq(c, {"th": 0.25})
    assert ir.num_qubits == 3
    # cirq packs gates into moments, so rz (disjoint qubit) moves up next to h.
    assert [(g.name, g.qubits) for g in ir] == [("h", (0,)), ("rz", (1,)), ("cx", (0, 2)), ("ccx", (0, 1, 2))]
    assert abs(ir.gates[1].params[0] - 0.25) < 1e-9


def test_commutation_table_is_sound():
    """commutes(a, b) must imply a.b == b.a exactly (not up to phase: X.Z = -Z.X)."""
    names = sorted(ARITY)
    for a_name, b_name in itertools.product(names, repeat=2):
        na, nb = ARITY[a_name], ARITY[b_name]
        width = na + nb - 1  # overlap >= 1; relabel so `a` sits on 0..na-1
        a = Gate(a_name, tuple(range(na)), (0.7,) if a_name in PARAMETRIC else ())
        for qb in itertools.permutations(range(width), nb):
            b = Gate(b_name, qb, (-1.1,) if b_name in PARAMETRIC else ())
            if set(a.qubits) & set(qb) and commutes(a, b):
                ab = circuit_unitary(IntermediateRepresentation(width, [a, b]))
                ba = circuit_unitary(IntermediateRepresentation(width, [b, a]))
                assert _allclose(ab, ba), (a, b)


def test_commutation_table_hand_cases():
    assert commutes(Gate("cx", (0, 1)), Gate("rz", (0,), (0.3,)))   # Z on control
    assert commutes(Gate("cx", (0, 1)), Gate("x", (1,)))            # X on target
    assert commutes(Gate("cx", (0, 1)), Gate("cx", (0, 2)))         # shared control
    assert commutes(Gate("cx", (0, 2)), Gate("cx", (1, 2)))         # shared target
    assert commutes(Gate("ccx", (0, 1, 2)), Gate("cz", (0, 1)))
    assert not commutes(Gate("cx", (0, 1)), Gate("cx", (1, 0)))
    assert not commutes(Gate("cx", (0, 1)), Gate("z", (1,)))
    assert not commutes(Gate("h", (0,)), Gate("t", (0,)))


def test_dag_critical_path_hand_cases():
    # GHZ chain: fully sequential.
    ghz = IntermediateRepresentation(4).add("h", [0]).add("cx", [0, 1]).add("cx", [1, 2]).add("cx", [2, 3])
    assert build_dag(ghz).critical_path_length() == 4 == ghz.depth()
    # Two independent wires: depth is the longer one, two roots.
    par = IntermediateRepresentation(2).add("h", [0]).add("x", [1]).add("t", [1])
    dag = build_dag(par)
    assert dag.critical_path_length() == 2
    assert dag.roots() == [0, 1]
    assert dag.pred[2] == {1}


# --- #9: hypergraph + partitioner -----------------------------------------------

from qco.graphs.partitioning import cut_size


def test_hypergraph_merges_duplicates_and_windows_by_interaction():
    ir = IntermediateRepresentation(4).add("cx", [0, 1]).add("cx", [0, 1]).add("swap", [2, 3])
    assert sorted((sorted(q), w) for q, w in build_hypergraph(ir).hyperedges) == [([0, 1], 2), ([2, 3], 3)]
    # Window of 3: cx(0,1) and swap(2,3) don't interact -> stay separate edges.
    windowed = build_hypergraph(ir.copy().add("cx", [1, 2]), window=3)
    assert sorted((sorted(q), w) for q, w in windowed.hyperedges) == [([0, 1], 2), ([1, 2], 1), ([2, 3], 3)]


def _check_blocks(ir, blocks, max_gates, max_qubits=None):
    ids = [g for b in blocks for g in b.source_gate_ids]
    assert sorted(ids) == list(range(len(ir)))  # every gate exactly once
    for q in range(ir.num_qubits):  # concatenation keeps each wire's order
        on_q = [g for g in ids if q in ir.gates[g].qubits]
        assert on_q == sorted(on_q)
    for b in blocks:
        assert len(b.ir) <= max_gates
        width = {q for g in b.ir for q in g.qubits}
        if max_qubits is not None and len(b.ir) > 1:
            assert len(width) <= max_qubits
        others = {q for o in blocks if o is not b for g in o.ir for q in g.qubits}
        assert b.boundary_qubits == width & others


@pytest.mark.parametrize("backend", ["auto", "fallback", "metis"])
@pytest.mark.parametrize("path", BENCHMARKS)
def test_partition_contract_on_benchmarks(path, backend):
    if backend == "metis":
        pytest.importorskip("pymetis")
    ir = parse(path)
    blocks = partition_circuit(ir, max_block_gates=32, backend=backend, max_block_qubits=6)
    _check_blocks(ir, blocks, 32, None if backend == "fallback" else 6)


def test_partition_unknown_backend_and_missing_kahypar_config(monkeypatch):
    ir = IntermediateRepresentation(20).add("cx", [0, 19])
    with pytest.raises(ValueError):
        partition_circuit(ir, backend="nope")
    monkeypatch.delenv("KAHYPAR_INI", raising=False)
    with pytest.raises(ImportError):
        partition_circuit(ir, backend="kahypar")


def test_greedy_bounds_width_at_modest_cut_cost():
    """Fallback blocks span the whole circuit; greedy keeps them <= 8 qubits
    (verifiable / re-synthesizable) at a modest wire-cut cost."""
    ir = parse("benchmarks/circuits/qaoa_n16_p1.qasm")
    greedy = partition_circuit(ir)
    widths = [len({q for g in b.ir for q in g.qubits}) for b in greedy]
    assert max(widths) <= 8
    assert cut_size(ir, greedy) <= 2 * cut_size(ir, partition_circuit(ir, backend="fallback"))

from qco.graphs.partitioning import _make_blocks, schedule_blocks, stitch_blocks
from qco.modules.smt_verifier.equivalence import EquivalenceChecker


def _per_wire(ir):
    return {q: [(g.name, g.qubits, g.params) for g in ir.gates if q in g.qubits]
            for q in range(ir.num_qubits)}


@pytest.mark.parametrize("backend", ["auto", "fallback", "metis"])
@pytest.mark.parametrize("path", BENCHMARKS)
def test_partition_schedule_stitch_equivalent(path, backend):
    if backend == "metis":
        pytest.importorskip("pymetis")
    ir = parse(path)
    if ir.num_qubits > 8:
        pytest.skip("dense equivalence only up to 8 qubits")
    blocks = schedule_blocks(
        ir, partition_circuit(ir, max_block_gates=8, backend=backend, max_block_qubits=4)
    )
    _check_blocks(ir, blocks, 8, None if backend == "fallback" else 4)  # exactly once, wire order, boundaries
    assert [b.index for b in blocks] == list(range(len(blocks)))
    stitched = stitch_blocks(ir, blocks)
    assert _per_wire(stitched) == _per_wire(ir)
    assert EquivalenceChecker().check(ir, stitched).equivalent


def test_schedule_reorders_out_of_order_blocks():
    ir = IntermediateRepresentation(3).add("h", [0]).add("cx", [0, 1]).add("cx", [1, 2])
    blocks = schedule_blocks(ir, _make_blocks(ir, [[2], [0], [1]]))
    assert [b.source_gate_ids for b in blocks] == [[0], [1], [2]]
    assert [b.index for b in blocks] == [0, 1, 2]
    assert blocks[0].boundary_qubits == frozenset({0})
    assert blocks[1].boundary_qubits == frozenset({0, 1})
    assert _per_wire(stitch_blocks(ir, _make_blocks(ir, [[2], [0], [1]]))) == _per_wire(ir)


def test_schedule_is_identity_on_valid_order():
    ir = IntermediateRepresentation(2).add("h", [0]).add("h", [1]).add("x", [0])
    blocks = _make_blocks(ir, [[1], [0, 2]])  # independent blocks, deliberately not min-id order
    assert [b.source_gate_ids for b in schedule_blocks(ir, blocks)] == [[1], [0, 2]]


def test_schedule_rejects_non_convex_block():
    ir = IntermediateRepresentation(2).add("h", [0]).add("cx", [0, 1]).add("x", [0])
    with pytest.raises(ValueError, match="cyclic"):
        schedule_blocks(ir, _make_blocks(ir, [[0, 2], [1]]))


def test_schedule_rejects_bad_coverage():
    ir = IntermediateRepresentation(2).add("h", [0]).add("x", [1]).add("t", [1])
    with pytest.raises(ValueError, match="in blocks"):
        schedule_blocks(ir, _make_blocks(ir, [[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match="in no block"):
        schedule_blocks(ir, _make_blocks(ir, [[0], [1]]))


def test_stitch_rewritten_blocks_equivalent_and_confined():
    ir = IntermediateRepresentation(2).add("h", [0]).add("h", [0]).add("cx", [0, 1])
    blocks = _make_blocks(ir, [[0, 1], [2]])
    stitched = stitch_blocks(ir, blocks, [IntermediateRepresentation(2), blocks[1].ir])  # h.h -> identity
    assert len(stitched) == 1
    assert EquivalenceChecker().check(ir, stitched).equivalent
    stray = IntermediateRepresentation(2).add("x", [1])
    with pytest.raises(ValueError, match="outside the block"):
        stitch_blocks(ir, blocks, [stray, blocks[1].ir])
        