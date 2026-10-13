"""Sprint 2 — Module A: config, features, built-in PPO, training loop, agent wrapper
and the optional SB3 backend.

The built-in PPO is pure Python, so everything except the numpy-parity and SB3
classes runs in the minimal CI (``pip install -e ".[test]"``).
"""

from __future__ import annotations

import csv
import math
import random

import pytest

from qco.modules.rl_scheduler import RLConfig, RLScheduler
from qco.modules.rl_scheduler._nn import HAVE_NUMPY, Adam, ScalarMLP, softmax
from qco.modules.rl_scheduler.baselines import evaluate, greedy_depth_chooser, random_chooser
from qco.modules.rl_scheduler.circuits import FAMILIES, random_circuit, sample_circuits
from qco.modules.rl_scheduler.features import CANDIDATE_DIM, GLOBAL_DIM, encode_state
from qco.modules.rl_scheduler.ppo import (
    PPOPolicy,
    PPOTrainer,
    Transition,
    load_policy,
    make_env,
    run_policy,
    save_policy,
    select_candidates,
)
from qco.modules.rl_scheduler.training import compare, train
from qco.modules.smt_verifier.equivalence import EquivalenceChecker

try:
    from qco.modules.rl_scheduler.sb3_backend import SB3_AVAILABLE
except Exception:  # pragma: no cover
    SB3_AVAILABLE = False

SMOKE = dict(total_timesteps=300, n_steps=100, epochs=2, minibatch=32, n_train_circuits=3,
             n_eval_circuits=2, circuit_layers=5, eval_every=1, checkpoint_every=1, numeric_backend="python")


def smoke_cfg(**kw) -> RLConfig:
    return RLConfig(**{**SMOKE, **kw}).validate()


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

class TestConfig:
    def test_defaults_keep_phase2_interface(self):
        c = RLConfig()
        assert (c.algo, c.total_timesteps, c.train_qubits, c.eval_qubits, c.deterministic_reduction) == (
            "ppo", 200_000, (5,), (10, 20, 40), True)

    def test_agent_module_still_exports_rlconfig(self):
        from qco.modules.rl_scheduler.agent import RLConfig as Legacy
        assert Legacy is RLConfig

    def test_dict_round_trip(self):
        c = RLConfig(seed=3, train_qubits=(4, 5), n_steps=64)
        assert RLConfig.from_dict(c.to_dict()) == c

    def test_yaml_round_trip(self, tmp_path):
        c = RLConfig(seed=9, lr=0.01)
        c.to_yaml(tmp_path / "c.yaml")
        assert RLConfig.from_yaml(tmp_path / "c.yaml") == c

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            RLConfig.from_dict({"nope": 1})

    @pytest.mark.parametrize("bad", [{"algo": "x"}, {"backend": "x"}, {"commutation": "x"},
                                     {"n_steps": 0}, {"max_candidates": 1}, {"train_qubits": ()}])
    def test_validation(self, bad):
        with pytest.raises(ValueError):
            RLConfig.from_dict(bad)

    def test_committed_yaml_configs_load(self):
        from pathlib import Path
        for f in (Path(__file__).resolve().parents[1] / "configs").glob("rl_*.yaml"):
            RLConfig.from_yaml(f)


# --------------------------------------------------------------------------
# circuits & features
# --------------------------------------------------------------------------

class TestCircuitsAndFeatures:
    def test_generators_deterministic(self):
        a = sample_circuits(5, 4, seed=1)
        b = sample_circuits(5, 4, seed=1)
        assert [c.gates for c in a] == [c.gates for c in b]
        assert [c.gates for c in a] != [c.gates for c in sample_circuits(5, 4, seed=2)]

    @pytest.mark.parametrize("family", FAMILIES)
    def test_families_valid(self, family):
        c = random_circuit(5, random.Random(0), family, 6)
        assert c.num_qubits == 5 and len(c.gates) > 0

    def test_unknown_family(self):
        with pytest.raises(ValueError):
            random_circuit(3, random.Random(0), "nope")

    def test_encode_state_shapes_and_finite(self):
        cfg = smoke_cfg()
        env = make_env(sample_circuits(5, 1, 0, layers=6)[0], cfg)
        _, info = env.reset()
        rows, g = encode_state(env, info["legal_actions"])
        assert len(rows) == len(info["legal_actions"]) and len(g) == GLOBAL_DIM
        assert all(len(r) == CANDIDATE_DIM and all(math.isfinite(v) for v in r) for r in rows)
        assert rows[0][0] == 1.0  # NOOP is first

    def test_select_candidates_keeps_noop_and_caps(self):
        cfg = smoke_cfg()
        env = make_env(sample_circuits(5, 1, 0, layers=10)[0], cfg)
        _, info = env.reset()
        acts = info["legal_actions"]
        sel = select_candidates(acts, 4, random.Random(0))
        assert sel[0] == acts[0] and len(sel) <= 4


# --------------------------------------------------------------------------
# neural-net pieces
# --------------------------------------------------------------------------

class TestNN:
    def test_gradient_matches_finite_differences(self):
        rng = random.Random(0)
        net = ScalarMLP(4, 3, rng=rng, out_scale=0.5, backend="python")
        X = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(5)]
        w = [rng.uniform(-1, 1) for _ in range(5)]

        def loss(theta):
            net.set_theta(theta)
            y, _ = net.forward(X)
            return sum(a * b for a, b in zip(w, y))

        theta = list(net.theta)
        y, cache = net.forward(X)
        grad = [0.0] * net.n_params
        net.backward(cache, w, grad)
        for k in range(0, net.n_params, 3):
            e = 1e-6
            tp, tm = list(theta), list(theta)
            tp[k] += e
            tm[k] -= e
            assert grad[k] == pytest.approx((loss(tp) - loss(tm)) / (2 * e), abs=1e-5)

    @pytest.mark.skipif(not HAVE_NUMPY, reason="numpy not installed")
    def test_numpy_matches_python(self):
        rng = random.Random(1)
        a = ScalarMLP(6, 5, rng=rng, out_scale=0.5, backend="python")
        b = ScalarMLP(6, 5, theta=a.theta, backend="numpy")
        X = [[rng.uniform(-1, 1) for _ in range(6)] for _ in range(7)]
        ya, ca = a.forward(X)
        yb, cb = b.forward(X)
        assert yb == pytest.approx(ya, abs=1e-10)
        dy = [rng.uniform(-1, 1) for _ in range(7)]
        ga, gb = [0.0] * a.n_params, [0.0] * b.n_params
        a.backward(ca, dy, ga)
        b.backward(cb, dy, gb)
        assert gb == pytest.approx(ga, abs=1e-10)

    def test_numpy_backend_without_numpy_errors_clearly(self):
        if HAVE_NUMPY:
            pytest.skip("numpy installed")
        with pytest.raises(ImportError):
            ScalarMLP(2, 2, backend="numpy")

    def test_adam_minimizes_quadratic(self):
        th = [3.0, -2.0]
        opt = Adam(2, lr=0.1, max_grad_norm=None)
        for _ in range(300):
            opt.step(th, [2 * th[0], 2 * th[1]])
        assert abs(th[0]) < 0.05 and abs(th[1]) < 0.05

    def test_adam_clips_global_norm(self):
        th = [0.0]
        opt = Adam(1, lr=1.0, max_grad_norm=1.0)
        assert opt.step(th, [100.0]) == pytest.approx(100.0)

    def test_softmax_normalizes_and_is_stable(self):
        p = softmax([1000.0, 1001.0, 999.0])
        assert sum(p) == pytest.approx(1.0) and all(x >= 0 for x in p)


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------

class TestPPO:
    def test_gae_matches_hand_computation(self):
        cfg = smoke_cfg(gamma=0.5, gae_lambda=0.5)
        tr = PPOTrainer(cfg)
        ep = [Transition([], [], 0, 0.0, v, r) for v, r in ((1.0, 1.0), (2.0, 0.0))]
        tr._gae(ep, next_value=4.0)
        d1 = 0.0 + 0.5 * 4.0 - 2.0
        d0 = 1.0 + 0.5 * 2.0 - 1.0
        assert ep[1].adv == pytest.approx(d1)
        assert ep[0].adv == pytest.approx(d0 + 0.5 * 0.5 * d1)
        assert ep[0].ret == pytest.approx(ep[0].adv + 1.0)

    def test_policy_round_trip_and_clone(self, tmp_path):
        cfg = smoke_cfg()
        pol = PPOPolicy(cfg.hidden, 0, "python", True)
        rows = [[random.Random(k).uniform(-1, 1) for _ in range(CANDIDATE_DIM)] for k in range(4)]
        save_policy(pol, tmp_path / "p.json")
        loaded, _ = load_policy(tmp_path / "p.json", "python")
        assert loaded.logits(rows) == pytest.approx(pol.logits(rows))
        assert pol.clone().logits(rows) == pytest.approx(pol.logits(rows))

    def test_load_rejects_foreign_file(self, tmp_path):
        (tmp_path / "x.json").write_text("{}")
        with pytest.raises(ValueError):
            load_policy(tmp_path / "x.json")

    def test_collect_finishes_episodes_and_update_changes_weights(self):
        cfg = smoke_cfg()
        tr = PPOTrainer(cfg)
        pool = sample_circuits(5, 3, 0, layers=5)
        batch, eps = tr.collect(pool)
        assert len(batch) >= cfg.n_steps and eps and sum(e["length"] for e in eps) == len(batch)
        before = list(tr.policy.actor.theta)
        stats = tr.update(batch)
        assert tr.policy.actor.theta != before
        assert all(math.isfinite(v) for v in stats.values())

    def test_run_policy_never_worse_than_cr_and_equivalent(self):
        cfg = smoke_cfg()
        pol = PPOPolicy(cfg.hidden, 0, "python", True)
        chk = EquivalenceChecker(backend="numeric")
        for c in sample_circuits(4, 4, 5, layers=6):
            out, stats = run_policy(pol, c, cfg, greedy=True)
            assert chk.check(c, out).equivalent
            assert (out.depth(), out.gate_count()) <= (c.depth(), c.gate_count())
            assert stats["steps"] >= 1

    def test_training_is_deterministic_in_seed(self):
        a = train(smoke_cfg(total_timesteps=100), log=None)[0]
        b = train(smoke_cfg(total_timesteps=100), log=None)[0]
        assert a.actor.theta == b.actor.theta


# --------------------------------------------------------------------------
# training loop, baselines, agent
# --------------------------------------------------------------------------

class TestTraining:
    def test_smoke_run_writes_logs_and_checkpoints(self, tmp_path):
        cfg = smoke_cfg()
        policy, hist = train(cfg, tmp_path, log=None)
        assert len(hist) >= 3
        for f in ("metrics.csv", "latest.json", "final.json", "best.json", "curves.svg"):
            assert (tmp_path / f).exists(), f
        rows = list(csv.DictReader(open(tmp_path / "metrics.csv")))
        assert len(rows) == len(hist) and rows[0]["iteration"] == "1"
        assert (tmp_path / "curves.svg").read_text().startswith("<svg")
        loaded, payload = load_policy(tmp_path / "final.json", "python")
        assert loaded.actor.theta == policy.actor.theta
        assert RLConfig.from_dict(payload["config"]) == cfg

    def test_resume_continues_iteration_count(self, tmp_path):
        cfg = smoke_cfg(total_timesteps=100)
        _, h1 = train(cfg, tmp_path, log=None)
        _, h2 = train(cfg, tmp_path, resume=True, log=None)
        assert h2[0]["iteration"] == h1[-1]["iteration"] + 1
        assert len(list(csv.DictReader(open(tmp_path / "metrics.csv")))) == len(h1) + len(h2)

    def test_ddqn_not_implemented(self):
        with pytest.raises(NotImplementedError):
            train(smoke_cfg(algo="ddqn"), log=None)

    def test_compare_reports_all_methods_and_equivalence(self):
        cfg = smoke_cfg()
        pol, _ = train(cfg, log=None)
        res = compare(cfg, pol, verify=True)
        assert set(res) == {"cr_only", "random", "greedy_depth", "agent"}
        assert all(r["equivalent"] for r in res.values())
        # every method starts from the CR-reduced circuit, so gate count never regresses
        assert res["agent"]["gate_red_pct"] >= res["cr_only"]["gate_red_pct"] - 1e-9

    def test_baselines_run(self):
        cfg = smoke_cfg()
        pool = sample_circuits(4, 2, 3, layers=5)
        rng = random.Random(0)
        for ch in (random_chooser(rng), greedy_depth_chooser(rng)):
            assert evaluate(pool, cfg, chooser=ch)["n"] == 2


class TestAgentWrapper:
    def test_untrained_schedule_unchanged(self):
        c = sample_circuits(4, 1, 0, layers=5)[0]
        s = RLScheduler()
        assert s.schedule(c).gate_count() == c.gate_count()

    def test_train_save_load_schedule(self, tmp_path):
        cfg = smoke_cfg()
        s = RLScheduler(cfg)
        s.train(out_dir=tmp_path / "run", log=None)
        s.save(tmp_path / "m.json")
        s2 = RLScheduler().load(tmp_path / "m.json")
        c = sample_circuits(4, 1, 9, layers=6)[0]
        out = s2.schedule(c)
        assert EquivalenceChecker(backend="numeric").check(c, out).equivalent
        assert out.gate_count() <= s.reduce(c).gate_count()

    def test_save_without_model_errors(self, tmp_path):
        with pytest.raises(RuntimeError):
            RLScheduler().save(tmp_path / "x.json")

    def test_ddqn_train_raises(self):
        with pytest.raises(NotImplementedError):
            RLScheduler(RLConfig(algo="ddqn")).train()


# --------------------------------------------------------------------------
# optional SB3 backend
# --------------------------------------------------------------------------

class TestSB3Backend:
    @pytest.mark.skipif(SB3_AVAILABLE, reason="SB3 installed")
    def test_clear_error_without_extra(self):
        from qco.modules.rl_scheduler.sb3_backend import train_sb3
        with pytest.raises(ImportError, match="rl"):
            train_sb3(smoke_cfg())

    @pytest.mark.skipif(not SB3_AVAILABLE, reason="needs the `rl` extra (gymnasium, sb3-contrib)")
    def test_slot_env_masks_and_gym_api(self):
        from qco.modules.rl_scheduler.sb3_backend import SlotEnv
        env = SlotEnv(sample_circuits(5, 2, 0, layers=5), smoke_cfg())
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        mask = env.action_masks()
        assert mask[0] and mask.sum() == len(env._actions)
        obs, r, term, trunc, _ = env.step(0)
        assert isinstance(r, float)

    @pytest.mark.skipif(not SB3_AVAILABLE, reason="needs the `rl` extra (gymnasium, sb3-contrib)")
    def test_maskable_ppo_trains_and_saves(self, tmp_path):
        from qco.modules.rl_scheduler.sb3_backend import train_sb3
        cfg = smoke_cfg(total_timesteps=128, n_steps=64, minibatch=32, epochs=1)
        train_sb3(cfg, tmp_path, log=None)
        assert (tmp_path / "sb3_model.zip").exists()
