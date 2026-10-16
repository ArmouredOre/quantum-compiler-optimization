"""Phase 2 smoke tests: modules A-D, equivalence checker, end-to-end pipeline."""

import math
import random

import pytest

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
from qco.pipeline import PipelineConfig, optimize


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
    # Real pymoo NSGA-II over block-variant genomes (N3), so the exact front
    # isn't deterministic (search can beat the seed) - check the search
    # properties instead of an exact circuit.
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
    # #6 and #11, closed). Block-level crossover is covered below.
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


def test_nsga2_front_is_equivalent_without_filtering():
    # N3 (#30): crossover only recombines per-block variant indices, so every
    # front member is equivalent by construction - no post-hoc filter.
    pytest.importorskip("pymoo")
    checker = EquivalenceChecker(backend="numeric")
    ir = IntermediateRepresentation(4, name="mix4")
    for _ in range(3):
        ir.add("h", [0]).add("h", [0]).add("cx", [0, 1]).add("t", [2]).add("cx", [1, 2])
        ir.add("rz", [3], [0.3]).add("cx", [2, 3]).add("cx", [2, 3]).add("x", [1])
    cfg = NSGA2Config(pop_size=16, generations=10)
    front = NSGA2Compiler(config=cfg).evolve([ir])
    assert front
    assert all(checker.check(ir, p.circuit).equivalent for p in front)
    assert min(p.objective.gate_count for p in front) < ir.gate_count()


def test_nsga2_block_variants_argument():
    # Caller-supplied verified variants: block 0 has a cheaper equivalent
    # variant, block 1 only itself. The search must find the cheap combination
    # and never produce anything non-equivalent.
    pytest.importorskip("pymoo")
    checker = EquivalenceChecker(backend="numeric")
    b0 = IntermediateRepresentation(2).add("h", [0]).add("h", [0]).add("cx", [0, 1])
    b0_cheap = IntermediateRepresentation(2).add("cx", [0, 1])
    b1 = IntermediateRepresentation(2).add("t", [1]).add("cx", [1, 0])
    full = IntermediateRepresentation(2, gates=b0.gates + b1.gates)
    front = NSGA2Compiler(config=NSGA2Config(pop_size=8, generations=5)).evolve(
        [], block_variants=[[b0, b0_cheap], [b1]]
    )
    assert all(checker.check(full, p.circuit).equivalent for p in front)
    assert min(p.objective.gate_count for p in front) == 3


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


# --- #31: novelty wiring in HybridOptimizer -------------------------------------

from qco.ir import parse
from qco.pipeline import HybridOptimizer, PipelineConfig, _compact


def test_pipeline_reports_block_count_not_front_size():
    ir = parse("benchmarks/circuits/qaoa_n16_p1.qasm")
    res = optimize(ir, PipelineConfig(solver_budget_s=1.0))
    from qco.graphs import partition_circuit

    assert res.blocks == len(partition_circuit(ir))


def test_pipeline_verifies_blocks_of_wide_circuits():
    # N2: 16 qubits is past the dense ceiling, but each block is checked on its
    # own <= 8 active qubits, so nothing is left unverified.
    ir = parse("benchmarks/circuits/qft_n16.qasm")
    res = optimize(ir, PipelineConfig(solver_budget_s=1.0))
    assert res.verified
    assert res.unverified_blocks == 0
    assert all(p.circuit.num_qubits == 16 for p in res.pareto_front)
    assert min(p.objective.gate_count for p in res.pareto_front) <= ir.gate_count()


def test_compact_relabels_onto_active_qubits():
    a = IntermediateRepresentation(20).add("cx", [3, 17]).add("h", [9])
    b = IntermediateRepresentation(20).add("h", [9]).add("cx", [3, 17])
    ca, cb = _compact(a, b)
    assert ca.num_qubits == cb.num_qubits == 3
    assert [g.qubits for g in ca] == [(0, 2), (1,)]
    assert EquivalenceChecker().check(ca, cb).equivalent


def test_unverifiable_rewrite_is_rejected_not_accepted():
    # > 8 active qubits, non-phase-poly gates: the checker can only say
    # "unverified", and the pipeline must keep the original.
    opt = HybridOptimizer()
    orig = IntermediateRepresentation(10)
    for q in range(10):
        orig.add("h", [q])
    for q in range(9):
        orig.add("cx", [q, q + 1])
    proposed = IntermediateRepresentation(10, list(orig.gates[:-1]))
    kept, accepted, result = opt.revert_rewrite(orig, proposed, block_idx=3)
    assert not accepted and result.method == "unverified"
    assert kept is orig
    assert opt._unverified == {3}


def test_feedback_loop_stops_when_reward_stops_improving():
    ir = _hh_circuit()
    res = optimize(ir, PipelineConfig(feedback_iterations=5, calibration="default"))
    assert 1 <= len(res.reward_history) < 5      # second pass has nothing left to gain
    assert res.reward_history[-1] <= 0 or len(res.reward_history) == 1
    assert res.reward == res.reward_history[-1]


def test_resynthesized_windows_are_stitched_back_in():
    # N1: H-CX-H == CZ on each pair; re-synthesis shrinks both windows.
    c = IntermediateRepresentation(4, name="two_windows")
    c.add("h", [1]).add("cx", [0, 1]).add("h", [1])
    c.add("h", [3]).add("cx", [2, 3]).add("h", [3])
    opt = HybridOptimizer(PipelineConfig(solver_budget_s=10.0))
    res = opt.run(c)
    best = min(res.pareto_front, key=lambda p: p.objective.gate_count).circuit
    assert best.gate_count() < c.gate_count()
    assert res.solver_time_s > 0
    assert EquivalenceChecker().check(c, best).equivalent


def test_pipeline_merges_adjacent_rotations():
    # Module B emits rotation_merge candidates; the pipeline used to apply only inverse pairs.
    ir = IntermediateRepresentation(2, name="rz_rz")
    ir.add("cx", [0, 1]).add("rz", [1], [0.3]).add("rz", [1], [0.4]).add("cx", [0, 1])
    res = optimize(ir, PipelineConfig(solver_budget_s=0.0))
    best = min(res.pareto_front, key=lambda p: p.objective.as_tuple()).circuit
    rz = [g for g in best.gates if g.name == "rz"]
    assert len(rz) == 1 and rz[0].params[0] == pytest.approx(0.7)
    assert EquivalenceChecker(backend="numeric").check(ir, best).equivalent


def test_controlled_rotation_pi_pair_is_not_cancelled():
    # crz(pi)·crz(pi) = controlled(-I) = Z on the control, not the identity (period 4*pi).
    ir = IntermediateRepresentation(2, name="crz_pi")
    ir.add("h", [0]).add("crz", [0, 1], [math.pi]).add("crz", [0, 1], [math.pi]).add("h", [1])
    res = optimize(ir, PipelineConfig(solver_budget_s=0.0))
    checker = EquivalenceChecker(backend="numeric")
    assert all(checker.check(ir, p.circuit).equivalent for p in res.pareto_front)


# --- #36: trained Stage-4 modules + reward hook ----------------------------------

import os

from qco.pipeline import RewardEvent

RL_POLICY = "docs/rl_training/ppo_5q/best.json"


def test_reward_hook_gets_block_and_iteration_events():
    events: list[RewardEvent] = []
    ir = parse("benchmarks/circuits/grover_w5_i2.qasm")
    res = optimize(ir, PipelineConfig(reward_hook=events.append, solver_budget_s=0.5, calibration="default"))
    per_block = [e for e in events if e.block_index >= 0]
    whole = [e for e in events if e.block_index == -1]
    assert len(per_block) == res.blocks
    assert [e.reward for e in whole] == res.reward_history
    assert all(e.calibration == "default" and e.iteration == 0 for e in events)
    from qco.evaluation.metrics import scalarized_reward

    e = per_block[0]
    assert e.reward == scalarized_reward(e.before, e.after, calibration="default")


@pytest.mark.skipif(not os.path.exists(RL_POLICY), reason="committed RL policy missing")
def test_pipeline_uses_trained_rl_policy():
    ir = parse("benchmarks/circuits/rand_cliffordt_n5.qasm")
    cfg = dict(solver_budget_s=0.5, use_ea=False)
    trained = optimize(ir, PipelineConfig(rl_policy_path=RL_POLICY, **cfg))
    untrained = optimize(ir, PipelineConfig(rl_policy_path=None, **cfg))  # None = untrained stand-in (default is now "auto")
    best = trained.pareto_front[0].circuit
    assert EquivalenceChecker().check(ir, best).equivalent
    assert best.gate_count() < untrained.pareto_front[0].circuit.gate_count()


def test_pipeline_loads_trained_gnn(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from qco.modules.gnn_cancellation.model import GNNConfig, GNNCancellationPredictor

    train = [parse(f"benchmarks/circuits/{n}.qasm") for n in ("grover_w3_i1", "rand_cliffordt_n5")]
    model = GNNCancellationPredictor(GNNConfig(epochs=2))
    model.fit(train, val_fraction=0.0, seed=0)
    path = model.save(tmp_path / "gnn.pt")

    ir = parse("benchmarks/circuits/grover_w3_i2.qasm")
    opt = HybridOptimizer(PipelineConfig(gnn_checkpoint=str(path), solver_budget_s=0.5))
    assert opt.gnn._net is not None
    res = opt.run(ir)
    assert all(EquivalenceChecker().check(ir, p.circuit).equivalent for p in res.pareto_front)


# --- N5: confidence-gated verification is measurable ------------------------------

def test_confidence_gating_skips_rule_proven_checks_at_equal_correctness():
    ir = parse("benchmarks/circuits/grover_w3_i2.qasm")
    # Module A off-policy (None): the trained agent's CR pass already removes the rule-proven
    # cancellations, which would leave nothing for the N5 gate to skip. This test isolates Module B/C.
    cfg = dict(solver_budget_s=0.0, use_ea=False, gnn_checkpoint=None, rl_policy_path=None)
    on = optimize(ir, PipelineConfig(confidence_gating=True, **cfg))
    off = optimize(ir, PipelineConfig(confidence_gating=False, **cfg))
    assert on.verifications_skipped > 0 and off.verifications_skipped == 0
    assert off.verification_calls > on.verification_calls
    assert on.verified and off.verified and on.unverified_blocks == off.unverified_blocks == 0
    checker = EquivalenceChecker()
    for res in (on, off):
        assert checker.check(ir, res.pareto_front[0].circuit).equivalent
    assert on.pareto_front[0].circuit.gate_count() == off.pareto_front[0].circuit.gate_count()
