"""Phase 2 smoke tests: IR, parser, DAG, hypergraph, partitioning."""

import math

from qco.ir import IntermediateRepresentation, from_qasm2
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
    assert hg.qubit_degrees()[2] == 1


def test_partition_covers_all_gates():
    ir = IntermediateRepresentation(4)
    for i in range(30):
        ir.add("h", [i % 4])
    blocks = partition_circuit(ir, max_block_gates=8)
    assert sum(b.ir.gate_count() for b in blocks) == 30
