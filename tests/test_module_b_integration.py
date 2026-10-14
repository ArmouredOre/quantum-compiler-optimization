import pytest
from types import SimpleNamespace
from qco.ir.intermediate_representation import IntermediateRepresentation

pytest.importorskip("torch")

from qco.ir.parser import from_qasm2
from qco.graphs.partitioning import partition_circuit
from qco.modules.gnn_cancellation.dataset import WindowLabel
from qco.modules.gnn_cancellation.features import rule_based_candidates
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor
from qco.pipeline import (
    DEFAULT_CHECKPOINT, PipelineConfig, _load_gnn, _window_spans, optimize,
)

QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
    "h q[0];\nh q[0];\ncx q[0],q[1];\ncx q[0],q[1];\n"
    "t q[2];\nx q[1];\ncx q[1],q[2];\nh q[2];\n"
)


def _ir(name="t"):
    ir = from_qasm2(QASM)
    ir.name = name
    return ir


def _keys(cands):
    return {(c.kind, tuple(c.gate_indices or ()) or (c.a, c.b)) for c in cands}


@pytest.fixture(scope="module")
def trained():
    ir = _ir()
    p = GNNCancellationPredictor()
    p.fit([ir])
    wl = [(ir, [WindowLabel(0, 2, 1, 2, 0.1, "optimal"),
                WindowLabel(2, 4, 0, 0, 0.1, "optimal")])]
    p.fit_windows(wl, wl, epochs=1)
    return p, ir


def test_checkpoint_round_trip_keeps_window_head(trained, tmp_path):
    p, ir = trained
    assert p.has_window_head
    path = p.save(tmp_path / "m.pt")
    q = GNNCancellationPredictor().load(path)
    assert q.has_window_head
    a = [(c.a, c.b, round(c.score, 5)) for c in p.predict(ir, threshold=0.0)]
    b = [(c.a, c.b, round(c.score, 5)) for c in q.predict(ir, threshold=0.0)]
    assert a == b


def test_pair_only_checkpoint_has_no_window_head(tmp_path):
    ir = _ir()
    p = GNNCancellationPredictor()
    p.fit([ir])
    q = GNNCancellationPredictor().load(p.save(tmp_path / "pair.pt"))
    assert not q.has_window_head


def test_rank_candidates_shapes(trained):
    p, ir = trained
    r = p.rank_candidates(ir)
    assert len(r.pairs) > 0
    for c in r.pairs:
        assert 0 <= c.a < len(ir.gates) and 0 <= c.b < len(ir.gates)
        assert isinstance(c.kind, str) and 0.0 <= c.score <= 1.0


def test_none_checkpoint_is_rule_based():
    g = _load_gnn(None)
    ir = _ir()
    assert not g.has_window_head
    assert _keys(g.rank_candidates(ir).pairs) == _keys(rule_based_candidates(ir))


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="no shipped checkpoint")
def test_module_b_never_worse_than_rules():
    def best(ckpt):
        cfg = PipelineConfig(gnn_checkpoint=ckpt, enable_resynthesis=False)
        res = optimize(QASM, cfg)
        return min(pt.circuit.gate_count() for pt in res.pareto_front)

    assert best("auto") <= best(None)


def _slice_windows(ir, cuts):
    return [
        SimpleNamespace(
            ir=IntermediateRepresentation(ir.num_qubits, list(ir.gates[s:e]), ir.name)
        )
        for s, e in zip(cuts, cuts[1:])
    ]


def test_window_spans_tiled():
    ir = _ir()
    n = len(ir.gates)
    wins = _slice_windows(ir, [0, 3, 5, n])
    assert _window_spans(ir, wins) == [(0, 3), (3, 5), (5, n)]


def test_window_spans_untiled_returns_none():
    ir = _ir()
    n = len(ir.gates)
    wins = _slice_windows(ir, [0, 3, 5, n])
    assert _window_spans(ir, wins[::-1]) is None                  # wrong order
    assert _window_spans(ir, wins[:-1]) is None                   # missing tail
    assert _window_spans(ir, _slice_windows(ir, [0, 3, 3, n])) is None  # empty window


def test_window_spans_on_real_partition_is_none_or_valid():
    ir = _ir()
    wins = partition_circuit(ir, max_block_gates=3, max_block_qubits=3)
    spans = _window_spans(ir, wins)
    if spans is not None:
        assert spans[0][0] == 0 and spans[-1][1] == len(ir.gates)
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))