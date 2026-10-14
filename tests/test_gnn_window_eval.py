import importlib.util
import math
import random
from pathlib import Path
from types import SimpleNamespace

from qco.modules.gnn_cancellation.eval import (
    expected_calibration_error, fit_temperature, precision_at_k, sigmoid)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_windows.py"
_spec = importlib.util.spec_from_file_location("eval_windows", _SCRIPT)
ew = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ew)


def _circ(name, labels):
    return (SimpleNamespace(name=name), [SimpleNamespace(label=l) for l in labels])


def test_temperature_single_class_is_identity():
    assert fit_temperature([0.3, -1.0, 2.0], [0, 0, 0]) == 1.0
    assert fit_temperature([0.3, -1.0, 2.0], [1, 1, 1]) == 1.0


def test_temperature_near_one_on_calibrated_logits():
    rng = random.Random(0)
    z = [rng.gauss(0, 2) for _ in range(5000)]
    y = [1 if rng.random() < sigmoid(v) else 0 for v in z]
    assert abs(fit_temperature(z, y) - 1.0) < 0.3


def test_temperature_scaling_preserves_ranking():
    z = [-3.0, -0.5, 0.0, 0.7, 4.0]
    for T in (0.3, 1.0, 2.5):
        s = [sigmoid(v / T) for v in z]
        assert s == sorted(s)


def test_ece_near_zero_when_calibrated():
    rng = random.Random(1)
    p = [rng.random() for _ in range(20000)]
    y = [1 if rng.random() < q else 0 for q in p]
    assert expected_calibration_error(p, y) < 0.03


def test_precision_at_k_ignores_unknown_labels():
    assert precision_at_k([[(0.9, None), (0.5, 1), (0.1, 0)]], 1) == 1.0


def test_folds_never_split_a_group_family_mode():
    data = [_circ(n, [0]) for n in
            ["grover_w5_i1", "grover_w5_i2", "grover_w7_i1", "qft_n4", "qft_n5", "ghz_n3", "qaoa_n4"]]
    folds = ew.make_folds(data, 4, seed=0, by_family=True)
    where = {}
    for f, held in enumerate(folds):
        for i in held:
            where.setdefault(ew.family(data[i][0].name), set()).add(f)
    assert all(len(v) == 1 for v in where.values())
    assert sorted(i for f in folds for i in f) == list(range(len(data)))


def test_folds_keep_siblings_together_in_workload_mode():
    data = [_circ(n, [0]) for n in
            ["grover_w5_i1", "grover_w5_i2", "grover_w7_i1", "grover_w7_i2", "qft_n4", "ghz_n3"]]
    for seed in range(5):
        folds = ew.make_folds(data, 3, seed=seed, by_family=False)
        where = {}
        for f, held in enumerate(folds):
            for i in held:
                where.setdefault(ew.workload(data[i][0].name), set()).add(f)
        assert all(len(v) == 1 for v in where.values())


def test_auroc_basic():
    assert ew.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert math.isnan(ew.auroc([0.1, 0.2], [0, 0]))
    