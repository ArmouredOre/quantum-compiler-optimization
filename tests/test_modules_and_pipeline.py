"""Phase 2 smoke tests: modules A-D, equivalence checker, end-to-end pipeline."""

import math
import random

from qco.ir import IntermediateRepresentation
from qco.modules.rl_scheduler import RLScheduler
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor
from qco.modules.smt_verifier.equivalence import EquivalenceChecker
from qco.modules.evolutionary.nsga2 import (
    NSGA2Compiler,
    NSGA2Config,
    dominates,
    mutate_validity_preserving,
    structure_aware_fidelity_surrogate,
)
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


def test_nsga2_genetic_search_finds_non_dominated_front():
    # Sprint 2: real pymoo NSGA-II with crossover/mutation over gate sequences,
    # so the exact front is no longer deterministic (search can beat the seed) -
    # check the search properties instead of an exact circuit.
    seed = IntermediateRepresentation(2)
    seed.add("h", [0]).add("h", [0]).add("cx", [0, 1])
    cfg = NSGA2Config(pop_size=12, generations=6)
    front = NSGA2Compiler(config=cfg).evolve([seed])

    assert front
    assert all(p.circuit.num_qubits == 2 for p in front)
    # mutually non-dominated: no point in the front is beaten on every objective
    # by another (pymoo's final population can still hold exact duplicates since
    # eliminate_duplicates=False, so compare pairwise rather than dedupe-and-count)
    assert not any(
        dominates(other.objective, p.objective)
        for p in front for other in front if other is not p
    )
    # search should never end up worse than the seed it started from
    assert min(p.objective.gate_count for p in front) <= seed.gate_count()


def test_mutation_operators_never_emit_an_invalid_circuit():
    # commute / drop-candidate / reorder are equivalence-preserving by
    # construction (Module B's gates_commute / rule_based_candidates, issues
    # #6 and #11, closed). Crossover is excluded from this claim - splicing
    # two different circuits together fundamentally isn't equivalence-
    # preserving, which is why the pipeline still re-verifies NSGA-II's front
    # before returning it (see test_pipeline_end_to_end_qft_like below).
    checker = EquivalenceChecker()
    rng = random.Random(7)

    ghz = IntermediateRepresentation(4, name="ghz4")
    ghz.add("h", [0]).add("cx", [0, 1]).add("cx", [1, 2]).add("cx", [2, 3])

    mix = IntermediateRepresentation(3, name="mix")
    mix.add("h", [0]).add("h", [0]).add("cx", [0, 1]).add("cx", [0, 1])
    mix.add("rz", [2], [math.pi / 4]).add("rz", [2], [-math.pi / 4])

    for base in (_hh_circuit(), ghz, mix):
        current = base.copy()
        for _ in range(25):
            current = mutate_validity_preserving(current, rng)
            assert checker.check(base, current).equivalent


def test_nsga2_falls_back_without_pymoo(monkeypatch):
    import qco.modules.evolutionary.nsga2 as nsga2

    monkeypatch.setattr(nsga2, "_HAVE_PYMOO", False)
    a = IntermediateRepresentation(2)
    a.add("h", [0]).add("h", [0])
    b = IntermediateRepresentation(2)
    b.add("h", [0])
    front = NSGA2Compiler().evolve([a, b])
    assert len(front) == 1
    assert front[0].circuit.gate_count() == 1


def test_structure_aware_surrogate_is_sensitive_to_layout():
    # Same gate multiset (two 2-qubit gates, nothing else) but different qubit
    # layout: disjoint pairs land in the same timing layer (crosstalk-charged),
    # a shared qubit forces them into separate layers (idle-decay-charged
    # instead). A flat per-gate-type product would score these identically;
    # the structure-aware surrogate must not.
    concurrent = IntermediateRepresentation(4)
    concurrent.add("cx", [0, 1]).add("cx", [2, 3])
    sequential = IntermediateRepresentation(4)
    sequential.add("cx", [0, 1]).add("cx", [1, 2])

    f_concurrent = structure_aware_fidelity_surrogate(concurrent)
    f_sequential = structure_aware_fidelity_surrogate(sequential)
    assert 0.0 < f_concurrent <= 1.0
    assert 0.0 < f_sequential <= 1.0
    assert f_concurrent != f_sequential


def test_pipeline_end_to_end_qft_like():
    ir = IntermediateRepresentation(3, name="mix")
    ir.add("h", [0]).add("h", [0]).add("cx", [0, 1]).add("cx", [0, 1]).add("rz", [2], [math.pi / 4])
    res = optimize(ir)
    assert res.verified
    assert res.pareto_front
    best = res.pareto_front[0].circuit
    assert best.gate_count() <= ir.gate_count()
    assert gate_count_reduction(ir, best).value >= 0
    # NSGA-II's genetic search can propose non-equivalent (even empty) circuits;
    # the pipeline must only ever surface verified ones.
    checker = EquivalenceChecker()
    assert all(checker.check(ir, p.circuit).equivalent for p in res.pareto_front)


def test_pipeline_on_generated_benchmark(tmp_path):
    import benchmarks.generate_circuits as gen

    c = gen.ghz(5)
    p = tmp_path / "ghz_n5.qasm"
    p.write_text(c.text())
    res = optimize(str(p))
    assert res.pareto_front
