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


# -- #23 / #32: baselines, solver column, ablations, one-command report ---------

import os
import shutil

from qco.evaluation.baselines import Unsupported, cx_cost
from qco.ir.parser import from_qasm2
from qco.evaluation.benchmark_runner import BENCH_DIR, main, pipeline_optimizer
from qco.modules.smt_verifier.equivalence import EquivalenceChecker
from qco.pipeline import PipelineConfig, optimize


def _tiny_suite(tmp_path, names=("ghz_n5.qasm", "grover_w3_i1.qasm")):
    for n in names:
        shutil.copy(os.path.join(BENCH_DIR, n), tmp_path / n)
    return str(tmp_path)


def test_cx_cost_is_basis_independent_weighting():
    ir = IntermediateRepresentation(3).add("h", [0]).add("cx", [0, 1]).add("ccx", [0, 1, 2]).add("swap", [1, 2])
    assert cx_cost(ir) == 1 + 6 + 3


def test_runner_records_solver_seconds_and_skips_unsupported(tmp_path):
    suite = _tiny_suite(tmp_path)

    def picky(ir):
        if ir.num_qubits == 5:
            raise Unsupported("no ghz today")
        return ir, 1.5

    report = run_benchmarks(picky, suite)
    assert report.skipped == ["ghz_n5.qasm"]
    assert [r.solver_seconds for r in report.rows] == [1.5]
    assert report.rows[0].cx_cost_before == report.rows[0].cx_cost_after > 0
    small = {r.circuit for r in run_benchmarks(_identity, max_qubits=5).rows}
    assert "ghz_n5.qasm" in small and "ghz_n10.qasm" not in small


@pytest.mark.parametrize("tool", ["qiskit-O0", "qiskit-O1", "qiskit-O2", "qiskit-O3", "tket", "pyzx", "bqskit"])
def test_baselines_return_equivalent_circuits(tool):
    from qco.evaluation.baselines import available_baselines

    opt = available_baselines().get(tool)
    if opt is None:
        pytest.skip(f"{tool} not installed")
    ir = from_qasm2(open(os.path.join(BENCH_DIR, "grover_w3_i1.qasm")).read())
    out = opt(ir.copy())
    assert out.num_qubits == ir.num_qubits
    assert EquivalenceChecker(backend="numeric").check(ir, out).equivalent


def test_qiskit_levels_improve_on_o0():
    from qco.evaluation.baselines import available_baselines

    b = available_baselines()
    if "qiskit-O0" not in b:
        pytest.skip("qiskit not installed")
    ir = from_qasm2(open(os.path.join(BENCH_DIR, "grover_w3_i2.qasm")).read())
    assert b["qiskit-O3"](ir.copy()).gate_count() < b["qiskit-O0"](ir.copy()).gate_count()


def test_ablation_flags_switch_modules_off(monkeypatch):
    from qco.modules.rl_scheduler.agent import RLScheduler

    def boom(self, circuit):
        raise AssertionError("Module A should be off")

    monkeypatch.setattr(RLScheduler, "schedule", boom)
    ir = IntermediateRepresentation(2).add("h", [0]).add("h", [0]).add("cx", [0, 1])
    res = optimize(ir, PipelineConfig(use_rl=False, use_ea=False, solver_budget_s=0.5))
    assert len(res.pareto_front) == 1
    assert res.pareto_front[0].circuit.gate_count() == 1


@pytest.mark.parametrize("mode", ["gnn", "rule", "random", "exhaustive"])
def test_window_ranking_modes_stay_equivalent(mode):
    c = IntermediateRepresentation(4)
    c.add("h", [1]).add("cx", [0, 1]).add("h", [1]).add("h", [3]).add("cx", [2, 3]).add("h", [3])
    best, solver_s = pipeline_optimizer(window_ranking=mode, solver_budget_s=5.0)(c.copy())
    assert EquivalenceChecker().check(c, best).equivalent
    assert best.gate_count() < c.gate_count()
    assert 0 < solver_s < float("inf")


def test_one_command_builds_every_table(tmp_path):
    # Tiny suite whose windows re-synthesize instantly (H-CX-H == CZ), so even
    # the unbudgeted "exhaustive" mode stays fast.
    suite = tmp_path / "suite"
    suite.mkdir()
    c = IntermediateRepresentation(4)
    c.add("h", [1]).add("cx", [0, 1]).add("h", [1]).add("h", [3]).add("cx", [2, 3]).add("h", [3])
    (suite / "hcxh.qasm").write_text(c.to_qasm2())
    suite = str(suite)
    out = tmp_path / "results"
    summary = main(["--suite", suite, "--out", str(out), "--budget", "0.5", "--window-max-qubits", "4",
                    "--gnn-checkpoint", "none"])
    assert "Module B ran as the rule-based stand-in" in summary  # no --gnn-model given
    for title in ("qco vs baselines", "Ablations", "N1 window selection", "N5: confidence-gated verification"):
        assert title in summary
    for row in ("minus A (RL)", "minus D (NSGA-II)", "exhaustive", "| qco |"):
        assert row in summary
    assert (out / "summary.md").exists() and (out / "qco.csv").exists()
    # #40: Pareto front per circuit (CSV always; PNG when matplotlib is installed)
    assert "Pareto fronts" in summary
    rows = list(csv.DictReader(open(out / "pareto" / "hcxh.csv")))
    assert {r["source"] for r in rows} >= {"input", "qco"}
    assert min(int(r["gate_count"]) for r in rows if r["source"] == "qco") < 6
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (out / "pareto" / "hcxh.png").stat().st_size > 0


def test_tool_crash_becomes_unsupported_row():
    from qco.evaluation.baselines import _guard

    @_guard("flaky")
    def crashes(ir):
        raise RuntimeError("Can only build replacement circuits for basic gates")

    with pytest.raises(Unsupported, match="flaky: RuntimeError"):
        crashes(IntermediateRepresentation(1))
