"""Sprint 2 (#11): GraphSAGE/GAT training loop, checkpoints, held-out F1."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.dataset import dump_suite, load_training_dataset
from qco.modules.gnn_cancellation.eval import (
    adjacent_inverse_baseline,
    evaluate_baseline,
    evaluate_predictor,
    held_out_split,
)
from qco.modules.gnn_cancellation.features import overlapping_gate_pairs, rule_based_candidates
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor, GNNConfig

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "benchmarks" / "circuits"

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def _tiny_config(**kwargs) -> GNNConfig:
    defaults = dict(conv="sage", hidden=32, layers=2, dropout=0.0, epochs=40, lr=5e-3)
    defaults.update(kwargs)
    return GNNConfig(**defaults)


def _synthetic_circuits() -> list[IntermediateRepresentation]:
    """Small labelled circuits covering every Sprint-1 candidate kind."""
    out: list[IntermediateRepresentation] = []

    for name in ("h", "x", "y", "z"):
        ir = IntermediateRepresentation(1, name=f"{name}{name}")
        ir.add(name, [0]).add(name, [0])
        out.append(ir)

    ir = IntermediateRepresentation(1, name="ssdg")
    ir.add("s", [0]).add("sdg", [0])
    out.append(ir)

    ir = IntermediateRepresentation(1, name="ttdg")
    ir.add("t", [0]).add("tdg", [0])
    out.append(ir)

    ir = IntermediateRepresentation(1, name="rz_merge")
    ir.add("rz", [0], [0.3]).add("rz", [0], [0.2])
    out.append(ir)

    ir = IntermediateRepresentation(1, name="rz_id")
    ir.add("rz", [0], [math.pi / 2]).add("rz", [0], [-math.pi / 2])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="xcx")
    ir.add("x", [1]).add("cx", [0, 1]).add("x", [1])
    out.append(ir)

    ir = IntermediateRepresentation(1, name="hxh")
    ir.add("h", [0]).add("x", [0]).add("h", [0])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="h_idle_h")
    ir.add("h", [0]).add("x", [1]).add("h", [0])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="hhcx")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="cxcx")
    ir.add("cx", [0, 1]).add("cx", [0, 1])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="czcz")
    ir.add("cz", [0, 1]).add("cz", [0, 1])
    out.append(ir)

    xs = IntermediateRepresentation(1, name="xxxx")
    for _ in range(4):
        xs.add("x", [0])
    out.append(xs)

    ss = IntermediateRepresentation(1, name="ssss")
    for _ in range(4):
        ss.add("s", [0])
    out.append(ss)

    ir = IntermediateRepresentation(2, name="ghz_like")
    ir.add("h", [0]).add("cx", [0, 1])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="ryry")
    ir.add("ry", [0], [0.4]).add("ry", [0], [0.1]).add("cx", [0, 1])
    out.append(ir)

    ir = IntermediateRepresentation(2, name="sdgs")
    ir.add("sdg", [1]).add("s", [1]).add("h", [0])
    out.append(ir)

    ir = IntermediateRepresentation(3, name="spaced_hh")
    ir.add("h", [2]).add("cx", [0, 1]).add("h", [2])
    out.append(ir)

    ir = IntermediateRepresentation(1, name="rx_inv")
    ir.add("rx", [0], [0.7]).add("rx", [0], [-0.7])
    out.append(ir)

    # Duplicates with a tag so train/test can both see the pattern.
    for i in range(3):
        ir = IntermediateRepresentation(1, name=f"hh_dup_{i}")
        ir.add("h", [0]).add("h", [0])
        out.append(ir)
        ir = IntermediateRepresentation(2, name=f"spaced_dup_{i}")
        ir.add("h", [0]).add("x", [1]).add("h", [0])
        out.append(ir)

    return out


@pytest.fixture(scope="module")
def trained() -> GNNCancellationPredictor:
    model = GNNCancellationPredictor(_tiny_config())
    model.fit(_synthetic_circuits(), negative_ratio=6, seed=0, val_fraction=0.2)
    return model


def test_sage_and_gat_forward_on_hand_circuit():
    from qco.modules.gnn_cancellation.model import CancellationGNN
    from qco.modules.gnn_cancellation.features import build_pyg_data, overlapping_gate_pairs
    from qco.modules.gnn_cancellation.model import GNNCancellationPredictor as Pred

    ir = IntermediateRepresentation(2, name="hh")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1])
    data = build_pyg_data(ir)
    pairs = overlapping_gate_pairs(ir)
    dummy = Pred(_tiny_config())
    for conv in ("sage", "gat"):
        net = CancellationGNN(_tiny_config(conv=conv, epochs=1))
        net.eval()
        h = net.encode(data.x, data.edge_index)
        assert h.shape == (3, 32)
        idx, attr = dummy._pair_tensors(ir, pairs)
        logits = net.score_pairs(h, idx, attr)
        assert logits.shape == (len(pairs),)


def test_untrained_predict_still_uses_rule_fallback():
    ir = IntermediateRepresentation(2, name="hh")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1])
    cands = GNNCancellationPredictor().predict(ir)
    assert any(c.kind == "inverse_pair" and {c.a, c.b} == {0, 1} for c in cands)


def test_fit_predict_finds_inverse_pair(trained: GNNCancellationPredictor):
    ir = IntermediateRepresentation(1, name="hh")
    ir.add("h", [0]).add("h", [0])
    cands = trained.predict(ir)
    assert cands, "trained model returned no candidates for HH"
    assert cands == sorted(cands, key=lambda c: -c.score)
    assert any({c.a, c.b} == {0, 1} for c in cands)
    assert all(0.0 <= c.score <= 1.0 for c in cands)


def test_checkpoint_roundtrip(trained: GNNCancellationPredictor, tmp_path: Path):
    ir = IntermediateRepresentation(1, name="xx")
    ir.add("x", [0]).add("x", [0])
    before = trained.predict(ir)
    path = tmp_path / "gnn.pt"
    trained.save(path)
    loaded = GNNCancellationPredictor().load(path)
    after = loaded.predict(ir)
    assert [pair_key(c) for c in after] == [pair_key(c) for c in before]
    assert [c.score for c in after] == pytest.approx([c.score for c in before], abs=1e-5)


def pair_key(c) -> tuple[int, int]:
    return (c.a, c.b)


def test_predict_ranked_on_every_benchmark(trained: GNNCancellationPredictor):
    qasms = sorted(CIRCUITS.glob("*.qasm"))
    assert qasms
    for path in qasms:
        ir = from_qasm2(path.read_text(encoding="utf-8"))
        ir.name = path.stem
        cands = trained.predict(ir)
        assert isinstance(cands, list)
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_held_out_f1_beats_adjacent_baseline():
    circuits = _synthetic_circuits()
    train, test = held_out_split(circuits, test_fraction=0.3, seed=1)
    assert train and test
    model = GNNCancellationPredictor(_tiny_config(epochs=50))
    model.fit(train, negative_ratio=6, seed=1, val_fraction=0.2)
    gnn = evaluate_predictor(model, test)
    baseline = evaluate_baseline(test)
    assert gnn.f1 >= baseline.f1 - 1e-9
    assert gnn.f1 >= 0.7


def test_fit_from_dumped_json_suite(tmp_path: Path):
    dump_suite(CIRCUITS, tmp_path)
    labelled = load_training_dataset(tmp_path)
    assert len(labelled) == len(list(CIRCUITS.glob("*.qasm")))
    tiny = [ir for ir, _ in labelled if ir.gate_count() <= 20][:6]
    assert tiny
    model = GNNCancellationPredictor(_tiny_config(epochs=3, hidden=16, layers=1))
    result = model.fit(tiny, val_fraction=0.0, seed=0)
    assert result.n_epochs == 3
    assert result.n_circuits == len(tiny)
    assert model.save(tmp_path / "ckpt.pt").is_file()


def test_overlapping_pairs_required_for_shared_support():
    ir = IntermediateRepresentation(2)
    ir.add("h", [0]).add("h", [1]).add("cx", [0, 1])
    pairs = overlapping_gate_pairs(ir)
    assert (0, 2) in pairs and (1, 2) in pairs
    assert (0, 1) not in pairs


def test_adjacent_baseline_misses_spaced_inverses():
    ir = IntermediateRepresentation(2, name="spaced")
    ir.add("h", [0]).add("x", [1]).add("h", [0])
    gold = {(c.a, c.b) for c in rule_based_candidates(ir)}
    naive = {(c.a, c.b) for c in adjacent_inverse_baseline(ir)}
    assert (0, 2) in gold
    assert (0, 2) not in naive


def test_empty_circuit_predict(trained: GNNCancellationPredictor):
    ir = IntermediateRepresentation(2, name="empty")
    assert trained.predict(ir) == []


def test_save_without_fit_raises():
    with pytest.raises(RuntimeError, match="nothing to save"):
        GNNCancellationPredictor(_tiny_config()).save("unused.pt")


def test_unknown_conv_rejected():
    from qco.modules.gnn_cancellation.model import CancellationGNN

    with pytest.raises(ValueError, match="sage"):
        CancellationGNN(GNNConfig(conv="gcn"))
