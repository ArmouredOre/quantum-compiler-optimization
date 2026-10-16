"""Tests for the graph-layer / parser fixes (#48). Drop into tests/.

Written against the source pasted in the chat; not run against the repo.
The first group fails on current main and passes after the patches in graph_layer_fixes.md.
The last group documents current behaviour and should pass before and after.
"""
import subprocess
import sys

import pytest

from qco.graphs import build_dag, build_hypergraph, partition_circuit
from qco.graphs.partitioning import _make_blocks, cut_size, stitch_blocks
from qco.ir import Gate, IntermediateRepresentation, from_qasm2

HEADER = "OPENQASM 2.0;\nqreg q[2];\n"


# --- fail before the patches -------------------------------------------------------
def test_gate_rejects_repeated_qubit():
    with pytest.raises(ValueError):
        Gate("cx", (0, 0))


def test_gate_rejects_negative_qubit():
    with pytest.raises(ValueError):
        Gate("h", (-1,))


@pytest.mark.parametrize("backend", ["auto", "fallback"])
def test_partition_rejects_nonpositive_caps(backend):
    ir = IntermediateRepresentation(2).add("h", [0])
    with pytest.raises(ValueError):
        partition_circuit(ir, max_block_gates=0, backend=backend)
    with pytest.raises(ValueError):
        partition_circuit(ir, max_block_qubits=0, backend=backend)


def test_inline_comment_does_not_drop_gate():
    assert len(from_qasm2(HEADER + "h q[0]; // note\n")) == 1


def test_commented_out_qreg_is_ignored():
    assert from_qasm2("// qreg x[3];\n" + HEADER + "h q[0];\n").num_qubits == 2


def test_unparseable_gate_line_raises_instead_of_vanishing():
    with pytest.raises(ValueError):
        from_qasm2(HEADER + "rz((pi)/2) q[0];\n")


def test_power_expression_rejected_quickly():
    # run in a subprocess: on the unpatched parser eval never returns, which would hang pytest
    code = ("from qco.ir import from_qasm2; "
            "from_qasm2('OPENQASM 2.0;\\nqreg q[1];\\nrz(9**9**9**9) q[0];\\n')")
    try:
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=5)
    except subprocess.TimeoutExpired:
        pytest.fail("parser hangs on a power expression")
    assert res.returncode != 0 and b"ValueError" in res.stderr


# --- should pass before and after ----------------------------------------------------
def test_cut_size_counts_dag_edges_not_wires():
    # two CX on the same pair share two qubits but are one DAG edge
    ir = IntermediateRepresentation(2).add("cx", [0, 1]).add("cx", [0, 1])
    assert cut_size(ir, _make_blocks(ir, [[0], [1]])) == 1
    assert len(build_dag(ir).succ[0]) == 1


@pytest.mark.parametrize("backend", ["auto", "fallback"])
def test_empty_circuit_partitions_to_no_blocks(backend):
    ir = IntermediateRepresentation(3)
    assert partition_circuit(ir, backend=backend) == []
    assert len(stitch_blocks(ir, [])) == 0


def test_hypergraph_without_multiqubit_gates_is_empty():
    ir = IntermediateRepresentation(3).add("h", [0]).add("x", [1])
    hg = build_hypergraph(ir)
    assert hg.hyperedges == []
    assert hg.to_kahypar_hmetis() == "0 3 1\n"
    