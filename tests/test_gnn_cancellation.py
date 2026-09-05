"""Sprint 1 (#6): Module B feature encoding, rule-based candidates, dataset dump."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qco.ir.intermediate_representation import ARITY, IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.dataset import dump_candidates, dump_suite, load_candidates
from qco.modules.gnn_cancellation.features import (
    EDGE_FEATURE_DIM,
    FEATURE_DIM,
    FEATURE_LAYOUT,
    GATE_VOCAB,
    IDX_ARITY,
    IDX_DEPTH,
    IDX_PARAMETRIC,
    IDX_QUBITS,
    MAX_ARITY,
    are_inverses,
    build_pyg_data,
    gate_node_features,
    gates_commute,
    rule_based_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "benchmarks" / "circuits"


def _load_qasm(name: str) -> IntermediateRepresentation:
    ir = from_qasm2((CIRCUITS / name).read_text(encoding="utf-8"))
    ir.name = Path(name).stem
    return ir


def _obvious_wire_inverse_pairs(ir: IntermediateRepresentation) -> list[tuple[int, int]]:
    """Ground truth: inverse pairs with nothing on the shared wires between them."""
    pairs: list[tuple[int, int]] = []
    n = len(ir.gates)
    for i in range(n):
        for j in range(i + 1, n):
            if not are_inverses(ir.gates[i], ir.gates[j]):
                continue
            involved = set(ir.gates[i].qubits) | set(ir.gates[j].qubits)
            blocked = any(set(ir.gates[k].qubits) & involved for k in range(i + 1, j))
            if not blocked:
                pairs.append((i, j))
    return pairs


# -- feature layout ----------------------------------------------------------

def test_feature_layout_is_frozen():
    assert GATE_VOCAB == tuple(sorted(ARITY))
    assert len(FEATURE_LAYOUT) == FEATURE_DIM
    assert FEATURE_DIM == len(GATE_VOCAB) + 1 + MAX_ARITY + 1 + 1
    assert FEATURE_LAYOUT[IDX_ARITY] == "arity"
    assert FEATURE_LAYOUT[IDX_PARAMETRIC] == "parametric"
    assert FEATURE_LAYOUT[IDX_DEPTH] == "depth_position"


def test_gate_node_features_hand_circuit_h_and_cx():
    ir = IntermediateRepresentation(2, name="hand")
    ir.add("h", [0]).add("cx", [0, 1])
    h = gate_node_features(ir.gates[0], 2, index=0, num_gates=2, layer=1, circuit_depth=2)
    cx = gate_node_features(ir.gates[1], 2, index=1, num_gates=2, layer=2, circuit_depth=2)

    assert len(h) == FEATURE_DIM == len(cx)
    assert h[GATE_VOCAB.index("h")] == 1.0
    assert sum(h[: len(GATE_VOCAB)]) == 1.0
    assert h[IDX_ARITY] == 1.0
    assert h[IDX_QUBITS : IDX_QUBITS + MAX_ARITY] == [0.0, -1.0, -1.0]
    assert h[IDX_PARAMETRIC] == 0.0
    assert h[IDX_DEPTH] == 0.0

    assert cx[GATE_VOCAB.index("cx")] == 1.0
    assert cx[IDX_ARITY] == 2.0
    assert cx[IDX_QUBITS : IDX_QUBITS + MAX_ARITY] == [0.0, 1.0, -1.0]
    assert cx[IDX_PARAMETRIC] == 0.0
    assert cx[IDX_DEPTH] == 1.0


def test_gate_node_features_parametric_and_ccx_qubit_slots():
    ir = IntermediateRepresentation(3)
    ir.add("rz", [1], [math.pi / 3]).add("ccx", [0, 1, 2])
    rz = gate_node_features(ir.gates[0], 3, index=0, num_gates=2, layer=1, circuit_depth=2)
    ccx = gate_node_features(ir.gates[1], 3, index=1, num_gates=2, layer=2, circuit_depth=2)
    assert rz[IDX_PARAMETRIC] == 1.0
    assert rz[IDX_QUBITS] == pytest.approx(1 / 2)
    assert ccx[IDX_ARITY] == 3.0
    assert ccx[IDX_QUBITS : IDX_QUBITS + MAX_ARITY] == [0.0, 0.5, 1.0]


def test_depth_position_differs_along_a_wire():
    ir = IntermediateRepresentation(1)
    ir.add("h", [0]).add("x", [0]).add("h", [0])
    feats = [
        gate_node_features(g, 1, index=i, num_gates=3, layer=i + 1, circuit_depth=3)
        for i, g in enumerate(ir.gates)
    ]
    depths = [f[IDX_DEPTH] for f in feats]
    assert depths[0] < depths[1] < depths[2]
    assert depths[0] == 0.0 and depths[2] == 1.0


# -- rule-based candidates ---------------------------------------------------

def test_adjacent_inverse_pair_hh():
    ir = IntermediateRepresentation(2, name="hh")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1])
    cands = rule_based_candidates(ir)
    assert any(c.kind == "inverse_pair" and {c.a, c.b} == {0, 1} for c in cands)


def test_commuting_through_inverse_pair_x_cx_x():
    """X on the CNOT target commutes through, so the two X gates cancel."""
    ir = IntermediateRepresentation(2, name="xcx")
    ir.add("x", [1]).add("cx", [0, 1]).add("x", [1])
    assert gates_commute(ir.gates[0], ir.gates[1])
    cands = rule_based_candidates(ir)
    assert any(c.kind == "inverse_pair" and {c.a, c.b} == {0, 2} for c in cands)


def test_non_commuting_h_x_h_is_not_an_inverse_pair():
    ir = IntermediateRepresentation(1)
    ir.add("h", [0]).add("x", [0]).add("h", [0])
    cands = [c for c in rule_based_candidates(ir) if c.kind == "inverse_pair"]
    assert not any({c.a, c.b} == {0, 2} for c in cands)


def test_rotation_chain_merge():
    ir = IntermediateRepresentation(1, name="rzs")
    ir.add("rz", [0], [0.3]).add("rz", [0], [0.2])
    cands = rule_based_candidates(ir)
    assert any(c.kind == "rotation_merge" and c.gate_indices == (0, 1) for c in cands)


def test_rotation_chain_that_sums_to_identity():
    ir = IntermediateRepresentation(1)
    ir.add("rz", [0], [math.pi / 2]).add("rz", [0], [-math.pi / 2])
    cands = rule_based_candidates(ir)
    assert any(c.kind == "identity_chain" for c in cands)
    assert any(c.kind == "inverse_pair" for c in cands)


def test_identity_chain_four_x_and_four_s():
    xs = IntermediateRepresentation(1)
    for _ in range(4):
        xs.add("x", [0])
    ss = IntermediateRepresentation(1)
    for _ in range(4):
        ss.add("s", [0])
    x_cands = rule_based_candidates(xs)
    s_cands = rule_based_candidates(ss)
    assert any(c.kind == "identity_chain" and c.gate_indices == (0, 1, 2, 3) for c in x_cands)
    assert any(c.kind == "identity_chain" and c.gate_indices == (0, 1, 2, 3) for c in s_cands)


def test_s_sdg_inverse_pair():
    ir = IntermediateRepresentation(1)
    ir.add("s", [0]).add("sdg", [0])
    cands = rule_based_candidates(ir)
    assert any(c.kind == "inverse_pair" for c in cands)


def _assert_recall_on(names: list[str]) -> None:
    for name in names:
        ir = _load_qasm(name)
        found = {(c.a, c.b) for c in rule_based_candidates(ir) if c.kind == "inverse_pair"}
        obvious = _obvious_wire_inverse_pairs(ir)
        missed = [pair for pair in obvious if pair not in found]
        assert not missed, f"{name}: missed obvious pairs {missed}"


def test_candidate_recall_ghz_qft_grover():
    ghz = sorted(p.name for p in CIRCUITS.glob("ghz_*.qasm"))
    qft = sorted(p.name for p in CIRCUITS.glob("qft_*.qasm"))
    grover = sorted(p.name for p in CIRCUITS.glob("grover_*.qasm"))
    _assert_recall_on(ghz + qft + grover)


def test_grover_finds_all_obvious_cancellations():
    grover = sorted(p.name for p in CIRCUITS.glob("grover_*.qasm"))
    assert grover, "benchmark grover circuits missing"
    for name in grover:
        ir = _load_qasm(name)
        obvious = _obvious_wire_inverse_pairs(ir)
        assert obvious, f"{name}: expected at least one obvious HH cancellation"
        found = {(c.a, c.b) for c in rule_based_candidates(ir) if c.kind == "inverse_pair"}
        assert set(obvious) <= found


# -- dataset dump ------------------------------------------------------------

def test_dump_and_load_candidates(tmp_path: Path):
    ir = IntermediateRepresentation(1, name="hh")
    ir.add("h", [0]).add("h", [0])
    path = tmp_path / "hh.json"
    dumped = dump_candidates(ir, path)
    assert path.is_file()
    assert dumped and dumped[0].kind == "inverse_pair"
    loaded_ir, loaded = load_candidates(path)
    assert loaded_ir.num_qubits == 1
    assert [g.name for g in loaded_ir] == ["h", "h"]
    assert loaded[0].a == dumped[0].a and loaded[0].kind == "inverse_pair"
    assert loaded[0].score == pytest.approx(1.0)


def test_dump_suite_writes_one_json_per_qasm(tmp_path: Path):
    written = dump_suite(CIRCUITS, tmp_path)
    qasms = list(CIRCUITS.glob("*.qasm"))
    assert len(written) == len(qasms)
    assert all(p.suffix == ".json" and p.is_file() for p in written)


# -- PyTorch Geometric -------------------------------------------------------

def test_build_pyg_data_on_every_benchmark_circuit():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric")
    del pyg

    qasms = sorted(CIRCUITS.glob("*.qasm"))
    assert qasms
    for path in qasms:
        ir = from_qasm2(path.read_text(encoding="utf-8"))
        ir.name = path.stem
        data = build_pyg_data(ir)
        assert data.x.shape == (len(ir.gates), FEATURE_DIM)
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape == (data.edge_index.shape[1], EDGE_FEATURE_DIM)
        assert data.x.dtype == torch.float32
        if len(ir.gates) > 0 and data.edge_index.numel() > 0:
            assert int(data.edge_index.max()) < len(ir.gates)
        assert data.num_qubits == ir.num_qubits


def test_build_pyg_data_empty_circuit():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    ir = IntermediateRepresentation(2, name="empty")
    data = build_pyg_data(ir)
    assert data.x.shape == (0, FEATURE_DIM)
    assert data.edge_index.shape == (2, 0)
