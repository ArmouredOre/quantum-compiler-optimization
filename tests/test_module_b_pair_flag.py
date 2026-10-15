"""fit_windows() alone must not make predict() use an untrained pair head (#42)."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from qco.ir.parser import from_qasm2  # noqa: E402
from qco.modules.gnn_cancellation.features import rule_based_candidates  # noqa: E402
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor, GNNConfig  # noqa: E402

QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\nh q[0];\ncx q[0],q[1];\n'


def _window_only_predictor(ir):
    rows = [SimpleNamespace(start=0, end=2, label=1), SimpleNamespace(start=1, end=3, label=0)]
    p = GNNCancellationPredictor(GNNConfig(epochs=1))
    p.fit_windows([(ir, rows)], [(ir, rows)], epochs=1)
    return p


def test_window_only_predictor_falls_back_to_rules_for_pairs():
    ir = from_qasm2(QASM)
    p = _window_only_predictor(ir)
    got = {(c.a, c.b) for c in p.predict(ir, threshold=0.0)}
    want = {(c.a, c.b) for c in rule_based_candidates(ir)}
    assert got == want


def test_window_only_checkpoint_stays_window_only(tmp_path):
    ir = from_qasm2(QASM)
    q = GNNCancellationPredictor().load(_window_only_predictor(ir).save(tmp_path / "w.pt"))
    got = {(c.a, c.b) for c in q.predict(ir, threshold=0.0)}
    want = {(c.a, c.b) for c in rule_based_candidates(ir)}
    assert got == want
    