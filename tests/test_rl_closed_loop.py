"""#35 — Module A closed loop: reward-hook consumer, real (Stage 6) reward for training,
end-to-end evaluation, and the trained policy as the pipeline default. Pure Python."""

from __future__ import annotations

import csv
import warnings

import pytest

from qco.evaluation.metrics import scalarized_reward
from qco.ir.parser import parse
from qco.modules.rl_scheduler import RLConfig
from qco.modules.rl_scheduler.circuits import sample_circuits
from qco.modules.rl_scheduler.closed_loop import (
    PipelineReward,
    RewardLog,
    evaluate_end_to_end,
    get_pipeline_reward,
    mean_curve,
    summarize_e2e,
    write_reward_curves,
)
from qco.modules.rl_scheduler.environment import Action, ActionKind, SchedulerEnv
from qco.modules.rl_scheduler.ppo import PPOPolicy, make_env
from qco.modules.rl_scheduler.training import train
from qco.modules.smt_verifier.equivalence import EquivalenceChecker
from qco.pipeline import DEFAULT_RL_POLICY, HybridOptimizer, PipelineConfig, optimize

warnings.simplefilter("ignore")
TINY = dict(total_timesteps=200, n_steps=100, epochs=1, minibatch=50, n_train_circuits=2, n_eval_circuits=2,
            circuit_layers=5, eval_every=1, checkpoint_every=0, numeric_backend="python")


def tiny(**kw) -> RLConfig:
    return RLConfig(**{**TINY, **kw}).validate()


class TestRewardLog:
    def test_collects_block_and_iteration_events(self, tmp_path):
        log = RewardLog()
        res = optimize(parse("benchmarks/circuits/rand_cliffordt_n5.qasm"),
                       PipelineConfig(reward_hook=log, use_ea=False, solver_budget_s=0.1, rl_policy_path=None))
        s = log.summary()
        assert s["block_events"] == res.blocks and s["iteration_events"] == len(res.reward_history)
        assert s["last_iteration_reward"] == pytest.approx(res.reward)
        log.to_csv(tmp_path / "r.csv")
        rows = list(csv.DictReader(open(tmp_path / "r.csv")))
        assert len(rows) == s["events"] and set(rows[0]) == set(RewardLog.FIELDS)
        log.clear()
        assert log.rows == []


class TestPipelineReward:
    def test_matches_stage6_reward_of_pipeline_output_and_caches(self):
        c = sample_circuits(5, 1, 3, layers=8)[0]
        pr = PipelineReward()
        out = pr.output(c)
        assert EquivalenceChecker(backend="numeric").check(c, out).equivalent
        r = pr(c, c)
        assert r == pytest.approx(scalarized_reward(c, out, calibration="default"))
        pr(c, c)
        assert pr.cache_hits == 1 and pr.calls == 2

    def test_hook_sees_inner_pipeline_events(self):
        log = RewardLog()
        PipelineReward(hook=log)(*(2 * [sample_circuits(5, 1, 3, layers=6)[0]]))
        assert log.rows

    def test_shared_instance_is_cached(self):
        assert get_pipeline_reward("default") is get_pipeline_reward("default")


class TestEnvTerminalReward:
    def _acts(self, env):
        return [a for a in env.legal_actions() if a.kind != ActionKind.NOOP]

    def test_bonus_is_relative_to_cr_reference(self):
        c = sample_circuits(5, 1, 2, layers=6)[0]
        env = SchedulerEnv(c, max_episode_steps=1, commutation="full", reduction=True,
                           terminal_reward_fn=lambda orig, circ: -1.0 * circ.depth())
        env.reset()
        act = (self._acts(env) or [Action(ActionKind.NOOP, -1)])[0]
        _, _, _, trunc, info = env.step(act)
        assert trunc
        assert info["terminal_reward"] == pytest.approx(-env.circuit.depth() + env.reference.depth())

    def test_constant_fn_gives_zero_bonus(self):
        c = sample_circuits(5, 1, 2, layers=6)[0]
        env = SchedulerEnv(c, max_episode_steps=1, terminal_reward_fn=lambda o, f: 3.0)
        env.reset()
        _, _, _, _, info = env.step(Action(ActionKind.NOOP, -1))
        assert info["terminal_reward"] == 0.0

    def test_default_is_proxy_bonus_and_key_present(self):
        env = SchedulerEnv(sample_circuits(5, 1, 2, layers=6)[0], max_episode_steps=1)
        env.reset()
        _, _, _, _, info = env.step(Action(ActionKind.NOOP, -1))
        assert "terminal_reward" in info and env.terminal_reward_fn is None


class TestConfigAndTraining:
    def test_reward_source_validation_and_roundtrip(self):
        with pytest.raises(ValueError):
            RLConfig(reward_source="nope").validate()
        c = RLConfig(reward_source="pipeline", eval_pipeline_reward=True)
        assert RLConfig.from_dict(c.to_dict()) == c

    def test_make_env_wires_pipeline_reward(self):
        c = sample_circuits(5, 1, 1, layers=5)[0]
        assert make_env(c, tiny()).terminal_reward_fn is None
        assert make_env(c, tiny(reward_source="pipeline")).terminal_reward_fn is not None

    def test_pipeline_reward_training_logs_real_reward(self, tmp_path):
        cfg = tiny(reward_source="pipeline")
        _, hist = train(cfg, tmp_path, log=None)
        rows = list(csv.DictReader(open(tmp_path / "metrics.csv")))
        assert "mean_terminal_reward" in rows[0] and rows[0]["eval_pipeline_reward"] != ""
        assert all(isinstance(h["mean_terminal_reward"], float) for h in hist)
        assert (tmp_path / "best.json").exists()


class TestEndToEnd:
    def test_evaluate_end_to_end_with_policy_and_stub(self):
        cfg = tiny()
        pol = PPOPolicy(cfg.hidden, 0, "python", True)
        cs = sample_circuits(5, 3, 8, layers=6)
        for i, c in enumerate(cs):
            c.name = f"c{i}"
        a = evaluate_end_to_end(cs, policy=pol, rl_config=cfg)
        b = evaluate_end_to_end(cs, rl_policy_path=None)
        assert [r.name for r in a] == ["c0", "c1", "c2"]
        s = summarize_e2e(a)
        assert s["n"] == 3 and set(s) == {"n", "reward", "depth_red_pct", "gate_red_pct"}
        assert all(r.depth_red_pct >= 0 for r in b)

    def test_curves(self, tmp_path):
        h = [{"timesteps": 100 * (i + 1), "eval_pipeline_reward": 0.1 * i, "mean_terminal_reward": 0.01 * i} for i in range(4)]
        xs, ys = mean_curve([h, h], "eval_pipeline_reward")
        assert xs == [100, 200, 300, 400] and ys[-1] == pytest.approx(0.3)
        write_reward_curves([h], [h], tmp_path / "c.svg")
        assert (tmp_path / "c.svg").read_text().startswith("<svg")


class TestPipelineDefaultPolicy:
    def test_default_is_auto_and_loads_committed_policy(self):
        assert PipelineConfig().rl_policy_path == "auto"
        opt = HybridOptimizer(PipelineConfig(gnn_checkpoint=None))
        assert (opt.rl._model is not None) == DEFAULT_RL_POLICY.exists()

    def test_none_forces_untrained_stand_in(self):
        assert HybridOptimizer(PipelineConfig(rl_policy_path=None, gnn_checkpoint=None)).rl._model is None

    def test_explicit_missing_path_errors(self):
        with pytest.raises(FileNotFoundError):
            HybridOptimizer(PipelineConfig(rl_policy_path="/no/such/policy.json", gnn_checkpoint=None))

    @pytest.mark.skipif(not DEFAULT_RL_POLICY.exists(), reason="committed policy missing")
    def test_pipeline_runs_end_to_end_with_default_trained_agent(self):
        ir = parse("benchmarks/circuits/rand_cliffordt_n5.qasm")
        res = optimize(ir, PipelineConfig(use_ea=False, solver_budget_s=0.1, gnn_checkpoint=None))
        best = res.pareto_front[0].circuit
        assert res.verified and EquivalenceChecker().check(ir, best).equivalent
        assert best.gate_count() < ir.gate_count()
