"""Sprint 1 (#8): tests for the Stage 6 metrics, calibration profiles, and the
benchmark runner (markdown/CSV export, baseline column stub).
"""

import csv
import io

import pytest

from qco.evaluation.calibration import CALIBRATIONS, get_calibration, list_calibrations
from qco.evaluation.metrics import (
    depth_reduction,
    estimated_fidelity,
    execution_time,
    gate_count_reduction,
    runtime_complexity,
    scalarized_reward,
)
from qco.evaluation.benchmark_runner import run_benchmarks
from qco.ir.intermediate_representation import IntermediateRepresentation


# -- calibration -------------------------------------------------------------

def test_list_and_get_calibration():
    names = list_calibrations()
    assert {"default", "superconducting_ibm_like", "trapped_ion_like"} <= set(names)
    for name in names:
        cal = get_calibration(name)
        assert cal.name == name
        assert cal.duration(1) > 0 and cal.duration(2) > 0
        assert 0 <= cal.error(1) < 1 and 0 <= cal.error(2) < 1


def test_get_calibration_unknown_name_raises():
    with pytest.raises(KeyError):
        get_calibration("does-not-exist")


def test_get_calibration_passthrough():
    cal = CALIBRATIONS["default"]
    assert get_calibration(cal) is cal


def test_trapped_ion_is_slower_but_more_accurate_than_superconducting():
    sc = get_calibration("superconducting_ibm_like")
    ti = get_calibration("trapped_ion_like")
    assert ti.duration(2) > sc.duration(2)
    assert ti.error(2) < sc.error(2)


# -- metrics -------------------------------------------------------------

def _ghz(n: int) -> IntermediateRepresentation:
    ir = IntermediateRepresentation(n)
    ir.add("h", [0])
    for i in range(n - 1):
        ir.add("cx", [i, i + 1])
    return ir


def test_gate_count_and_depth_reduction():
    before = _ghz(5)
    after = IntermediateRepresentation(5)
    after.add("h", [0]).add("cx", [0, 1])  # a stand-in "optimized" circuit
    gr = gate_count_reduction(before, after)
    dr = depth_reduction(before, after)
    assert gr.value == pytest.approx((5 - 2) / 5 * 100)
    assert dr.value == pytest.approx((5 - 2) / 5 * 100)
    assert gr.unit == "%" and dr.unit == "%"


def test_gate_count_reduction_zero_before_is_zero_not_div_by_zero():
    empty = IntermediateRepresentation(1)
    assert gate_count_reduction(empty, empty).value == 0.0


def test_execution_time_matches_hand_calculation_default_calibration():
    ir = IntermediateRepresentation(2)
    ir.add("h", [0]).add("cx", [0, 1])
    cal = get_calibration("default")
    expected = cal.duration(1) + cal.duration(2)  # serial: h then cx on q0
    assert execution_time(ir).value == pytest.approx(expected)


def test_execution_time_varies_by_calibration():
    ir = _ghz(5)
    t_default = execution_time(ir, "default").value
    t_ion = execution_time(ir, "trapped_ion_like").value
    assert t_ion > t_default  # ion gates are far slower


def test_estimated_fidelity_is_product_of_success_probs():
    ir = IntermediateRepresentation(2)
    ir.add("h", [0]).add("cx", [0, 1])
    cal = get_calibration("default")
    expected = (1 - cal.error(1)) * (1 - cal.error(2))
    assert estimated_fidelity(ir).value == pytest.approx(expected)


def test_estimated_fidelity_empty_circuit_is_one():
    empty = IntermediateRepresentation(1)
    assert estimated_fidelity(empty).value == pytest.approx(1.0)


def test_runtime_complexity_per_gate():
    r = runtime_complexity(seconds=2.0, circuit_size=100)
    assert r.value == pytest.approx(0.02)
    assert runtime_complexity(seconds=1.0, circuit_size=0).value == 0.0


def test_scalarized_reward_zero_for_identity():
    ir = _ghz(5)
    assert scalarized_reward(ir, ir.copy()) == pytest.approx(0.0)


def test_scalarized_reward_positive_for_strict_improvement():
    before = _ghz(5)
    after = IntermediateRepresentation(5)
    after.add("h", [0]).add("cx", [0, 1])  # fewer gates, shallower, higher fidelity
    assert scalarized_reward(before, after) > 0


# -- benchmark runner ------------------------------------------------------

def _identity(ir: IntermediateRepresentation) -> IntermediateRepresentation:
    return ir


def test_run_benchmarks_identity_gives_zero_reduction_and_correct_absolutes():
    report = run_benchmarks(_identity)
    assert report.rows, "expected the committed benchmark suite to be non-empty"
    for row in report.rows:
        assert row.gate_count_before == row.gate_count_after
        assert row.depth_before == row.depth_after
        assert row.gate_reduction_pct == pytest.approx(0.0)
        assert row.depth_reduction_pct == pytest.approx(0.0)
        assert row.reward == pytest.approx(0.0)
        assert row.baseline_gate_count is None  # stubbed until Phase 5

    ghz5 = next(r for r in report.rows if r.circuit == "ghz_n5.qasm")
    assert ghz5.gate_count_before == 5
    assert ghz5.depth_before == 5


def test_run_benchmarks_calibration_selectable_by_name():
    default_report = run_benchmarks(_identity, calibration="default")
    ion_report = run_benchmarks(_identity, calibration="trapped_ion_like")
    assert default_report.calibration_name == "default"
    assert ion_report.calibration_name == "trapped_ion_like"
    assert ion_report.mean("exec_time_ns") > default_report.mean("exec_time_ns")


def test_run_benchmarks_with_baseline_fills_stub_columns():
    def fake_baseline(ir):  # pretend "optimizer" that halves nothing, just a stand-in
        return ir

    report = run_benchmarks(_identity, baseline=fake_baseline, baseline_name="qiskit-O2")
    row = report.rows[0]
    assert row.baseline_name == "qiskit-O2"
    assert row.baseline_gate_count == row.gate_count_before
    assert row.baseline_depth == row.depth_before


def test_benchmark_report_to_markdown_and_csv():
    report = run_benchmarks(_identity)
    md = report.to_markdown()
    assert "gate red %" in md
    assert "default" in md  # calibration profile noted in the header
    assert "**mean**" in md

    csv_text = report.to_csv()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == len(report.rows)
    assert rows[0]["circuit"]
    assert "baseline_gate_count" in rows[0]
