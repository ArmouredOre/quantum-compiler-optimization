import random
import pytest
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.gnn_cancellation.dataset import dump_resynth_labels, extract_windows
from qco.modules.gnn_cancellation.eval import (
    expected_calibration_error as ece, fit_temperature, precision_at_k, sigmoid)

def test_temperature_fixes_overconfidence():
    rng = random.Random(0); z, y = [], []
    for _ in range(3000):
        t = rng.gauss(0, 1)
        y.append(int(rng.random() < sigmoid(t))); z.append(3 * t)   # 3x overconfident
    T = fit_temperature(z, y)
    assert 2.0 < T < 4.0
    assert ece([sigmoid(v / T) for v in z], y) < ece([sigmoid(v) for v in z], y)

def test_precision_at_k_ignores_unknown_and_ranks():
    items = [[(0.9, 1), (0.8, None), (0.1, 0)]]
    assert precision_at_k(items, 1) == 1.0
    assert precision_at_k([[(0.1, 1), (0.9, 0)]], 1) == 0.0

def test_resynth_labels_hh_reducible():
    ir = IntermediateRepresentation(1, name="hh"); ir.add("h", [0]).add("h", [0])
    [(_, rows)] = dump_resynth_labels([ir], max_gates=4, stride=1, timeout_s=5)
    assert rows[0].label == 1 and rows[0].gates_saved == 2

def test_window_head_shapes():
    torch = pytest.importorskip("torch"); pytest.importorskip("torch_geometric")
    from qco.modules.gnn_cancellation.model import GNNCancellationPredictor, GNNConfig
    ir = IntermediateRepresentation(2, name="t")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1]).add("cx", [0, 1])
    p = GNNCancellationPredictor(GNNConfig(epochs=1, hidden=16))
    from qco.modules.gnn_cancellation.model import CancellationGNN
    p._net = CancellationGNN(p.config)
    assert len(p.score_windows(ir, extract_windows(ir, stride=1))) == len(extract_windows(ir, stride=1))
    