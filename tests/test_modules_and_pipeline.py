"""Phase 2 smoke tests: modules A-D, equivalence checker, end-to-end pipeline."""

import math

from qco.ir import IntermediateRepresentation
from qco.modules.rl_scheduler import RLScheduler
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor
from qco.modules.smt_verifier.equivalence import EquivalenceChecker
from qco.modules.evolutionary.nsga2 import NSGA2Compiler
from qco.evaluation.metrics import gate_count_reduction, scalarized_reward
from qco.pipeline import optimize


def _hh_circuit():
    ir = IntermediateRepresentation(2, name="hh")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1])   # first two H cancel
    return ir


def test_equivalence_checker_detects_identity_pair():
    ir = _hh_circuit()
    reduced = IntermediateRepresentation(2)
    reduced.add("cx", [0, 1])
    assert EquivalenceChecker().check(ir, reduced).equivalent


def test_equivalence_checker_rejects_wrong_rewrite():
    ir = _hh_circuit()
    wrong = IntermediateRepresentation(2)
    wrong.add("x", [0]).add("cx", [0, 1])
    assert not EquivalenceChecker().check(ir, wrong).equivalent


def test_gnn_rule_baseline_finds_inverse_pair():
    cands = GNNCancellationPredictor().predict(_hh_circuit())
    assert any(c.kind == "inverse_pair" for c in cands)


def test_rl_scheduler_preserves_gate_count():
    ir = IntermediateRepresentation(3)
    ir.add("h", [0]).add("x", [2]).add("cx", [0, 1]).add("h", [2])
    out = RLScheduler().schedule(ir)
    assert out.gate_count() == ir.gate_count()
    assert EquivalenceChecker().check(ir, out).equivalent


def test_nsga2_returns_non_dominated_front():
    a = IntermediateRepresentation(2)
    a.add("h", [0]).add("h", [0])
    b = IntermediateRepresentation(2)
    b.add("h", [0])
    front = NSGA2Compiler().evolve([a, b])
    assert len(front) == 1
    assert front[0].circuit.gate_count() == 1


def test_pipeline_end_to_end_qft_like():
    ir = IntermediateRepresentation(3, name="mix")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1]).add("cx", [0, 1]).add("rz", [2], [math.pi / 4])
    res = optimize(ir)
    assert res.verified
    assert res.pareto_front
    best = res.pareto_front[0].circuit
    assert best.gate_count() <= ir.gate_count()
    assert gate_count_reduction(ir, best).value >= 0


def test_pipeline_on_generated_benchmark(tmp_path):
    import benchmarks.generate_circuits as gen

    c = gen.ghz(5)
    p = tmp_path / "ghz_n5.qasm"
    p.write_text(c.text())
    res = optimize(str(p))
    assert res.pareto_front
