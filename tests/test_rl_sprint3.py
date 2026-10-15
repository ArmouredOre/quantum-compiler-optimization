"""Sprint 3 / novelty N4 — Module A: calibrated-time reward, per-backend training,
cross-backend matrix, zero-shot generalization harness, SB3 model in ``RLScheduler``.

Pure Python except ``TestSB3Agent`` (skipped without the ``rl`` extra).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler import RLConfig, RLScheduler
from qco.modules.rl_scheduler.backend_matrix import (
    backend_config,
    cross_matrix,
    time_reduction_pct,
    to_markdown as matrix_md,
    train_backend_policies,
)
from qco.modules.rl_scheduler.circuits import sample_circuits
from qco.modules.rl_scheduler.environment import Action, ActionKind, RewardWeights, SchedulerEnv
from qco.modules.rl_scheduler.features import _lookahead
from qco.modules.rl_scheduler.generalization import (
    Policies,
    bucket_of,
    load_benchmarks,
    run_methods,
    study,
    summarize,
    to_markdown as gen_md,
)
from qco.modules.rl_scheduler.ppo import make_env
from qco.modules.rl_scheduler.training import train
from qco.modules.smt_verifier.equivalence import EquivalenceChecker

try:
    from qco.modules.rl_scheduler.sb3_backend import SB3_AVAILABLE
except Exception:  # pragma: no cover
    SB3_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(total_timesteps=100, n_steps=50, epochs=1, minibatch=25, n_train_circuits=2, n_eval_circuits=2,
            circuit_layers=4, eval_every=0, checkpoint_every=0, numeric_backend="python")


def tiny(**kw) -> RLConfig:
    return RLConfig(**{**TINY, **kw}).validate()


def two_q_chain() -> IntermediateRepresentation:
    c = IntermediateRepresentation(3)
    c.add("rz", [0], [0.3])
    c.add("cx", [0, 1])
    c.add("cx", [1, 2])
    return c


class TestTimeReward:
    def test_default_weights_leave_reward_unchanged(self):
        assert RewardWeights().time == 0.0
        env = SchedulerEnv(two_q_chain())
        _, _, _, _, info = env.step(Action(ActionKind.NOOP, -1))
        assert info["d_time"] == 0.0

    def test_time_term_enters_reward(self):
        c = two_q_chain()
        acts = [a for a in SchedulerEnv(c, commutation="full").legal_actions() if a.kind == ActionKind.COMMUTE]
        assert acts
        base = SchedulerEnv(c, RewardWeights(time=0.0, terminal=0.0), commutation="full", calibration="trapped_ion_like")
        timed = SchedulerEnv(c, RewardWeights(time=1.0, terminal=0.0), commutation="full", calibration="trapped_ion_like")
        _, r0, *_ , i0 = base.step(acts[0])
        _, r1, *_ , i1 = timed.step(acts[0])
        assert r1 == pytest.approx(r0 + i1["d_time"])

    def test_time_unit_positive(self):
        assert SchedulerEnv(two_q_chain(), calibration="trapped_ion_like").time_unit > 0

    def test_lookahead_time_metric_bounded(self):
        cfg = tiny(lookahead_metric="time", calibration="trapped_ion_like", commutation="full")
        env = make_env(sample_circuits(5, 1, 0, layers=6)[0], cfg)
        _, info = env.reset()
        vals = [_lookahead(env, a) for a in info["legal_actions"]]
        assert all(-1.0 <= v <= 1.0 for v in vals)

    def test_config_validation_new_fields(self):
        with pytest.raises(ValueError):
            RLConfig(lookahead_metric="nope").validate()
        with pytest.raises(KeyError):
            RLConfig(calibration="no_such_device").validate()
        c = RLConfig(lookahead_metric="time", weights=RewardWeights(time=2.0))
        assert RLConfig.from_dict(c.to_dict()) == c


class TestBackendMatrix:
    def test_backend_config(self):
        c = backend_config(RLConfig(), "trapped_ion_like", seed=4)
        assert (c.calibration, c.lookahead_metric, c.seed) == ("trapped_ion_like", "time", 4)
        assert c.weights.time == 1.0 and c.weights.depth == 0.0

    def test_time_reduction_pct(self):
        a = two_q_chain()
        assert time_reduction_pct(a, a, "default") == 0.0
        b = IntermediateRepresentation(3)
        assert time_reduction_pct(a, b, "default") == pytest.approx(100.0)

    def test_cross_matrix_shape_and_report(self):
        base = tiny()
        profiles = ("superconducting_ibm_like", "trapped_ion_like")
        pols = train_backend_policies(base, profiles, seeds=(0,))
        assert len(pols) == 2
        res = cross_matrix(pols, sample_circuits(5, 3, 1, layers=4), profiles)
        for t in profiles:
            for a in profiles:
                assert res["matrix"][t][a]["mean"] > -50
            assert "greedy_time" in res["matrix"][t]
        assert set(res["wins_own_backend"]) == set(profiles)
        assert 0.0 <= res["fraction_schedules_differ"] <= 1.0
        md = matrix_md(res)
        assert "train on A" in md and "trapped_ion_like" in md


class TestGeneralization:
    def test_bucket_of(self):
        assert [bucket_of(n) for n in (5, 6, 10, 20)] == ["5-6", "5-6", "8-12", "16-20"]

    def test_load_benchmarks_filters_families_and_size(self):
        cs = load_benchmarks(ROOT / "benchmarks" / "circuits", max_qubits=10)
        assert cs and all(c.num_qubits <= 10 for c in cs)
        assert not any(c.name.startswith(("grover", "ghz_n40")) for c in cs)

    def test_study_and_report(self):
        cfg = tiny()
        pol, _ = train(cfg, log=None)
        nocfg = tiny(deterministic_reduction=False)
        nopol, _ = train(nocfg, log=None)
        pols = {0: Policies(pol, cfg, nopol, nocfg)}
        circuits = sample_circuits(4, 2, 7, layers=5)
        for i, c in enumerate(circuits):
            c.name = f"c{i}"
        one = run_methods(circuits[0], pols[0], verify=EquivalenceChecker(backend="numeric"))
        assert set(one) == {"stub", "cr_only", "greedy", "agent_cr", "agent_cr_x", "agent_cr_off", "agent_nocr"}
        assert all(r["equivalent"] for r in one.values())
        assert one["stub"]["depth_red_pct"] == 0.0  # the Phase 2 stub only swaps disjoint gates
        res = study(circuits, pols)
        summ = summarize(res)
        md = gen_md(res, summ)
        assert "Zero-shot generalization" in md and "PPO + CR" in md

    def test_committed_checkpoint_loads_and_schedules(self):
        ck = ROOT / "models" / "rl_ppo_5q.json"
        if not ck.exists():
            pytest.skip("checkpoint not committed")
        s = RLScheduler().load(ck)
        c = sample_circuits(5, 1, 99, layers=6)[0]
        out = s.schedule(c)
        assert EquivalenceChecker(backend="numeric").check(c, out).equivalent
        assert out.depth() <= c.depth()


@pytest.mark.skipif(not SB3_AVAILABLE, reason="needs the `rl` extra (gymnasium, sb3-contrib)")
class TestSB3Agent:
    def test_train_schedule_save_load(self, tmp_path):
        cfg = tiny(backend="sb3", total_timesteps=128, n_steps=64, minibatch=32)
        s = RLScheduler(cfg)
        s.train(out_dir=tmp_path, log=None)
        c = sample_circuits(4, 1, 3, layers=5)[0]
        out = s.schedule(c)
        assert EquivalenceChecker(backend="numeric").check(c, out).equivalent
        s.save(tmp_path / "m.zip")
        s2 = RLScheduler().load(tmp_path / "m.zip")
        assert s2.config.backend == "sb3"
        assert s2.schedule(c).gates == out.gates
