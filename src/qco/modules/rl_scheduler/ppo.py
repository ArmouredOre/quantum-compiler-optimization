"""Module A — built-in PPO (pure Python, optional numpy), the default learner.

Why a custom PPO and not only Stable-Baselines3
-----------------------------------------------
CI installs ``.[test]`` only (no torch / numpy), and the Sprint 2 issue asks for a
committed smoke check, so the default learner must run on the standard library.
``sb3_backend.py`` provides the SB3 ``MaskablePPO`` alternative behind the ``rl``
extra; both consume the same features (``features.py``) and environment.

Policy
------
Action-scoring policy: the *same* MLP scores every legal action from its feature
vector (``features.encode_state``) and the scores are soft-maxed. This

* handles a variable number of legal actions (no fixed action indices,
  no illegal-action sampling — masking is implicit), and
* is independent of circuit size, which is what the Sprint 3 5 -> 10/20/40-qubit
  generalization study needs.

A separate small MLP maps the global features to the state value.

Algorithm
---------
Standard PPO-clip: GAE advantages, per-batch advantage normalization, clipped
surrogate, value regression, entropy bonus, global-norm gradient clipping, Adam.
Rollouts always run episodes to completion; time-limit truncation bootstraps with
``V(final state)`` while true termination bootstraps with 0.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler._nn import Adam, ScalarMLP, resolve_backend, softmax
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.environment import Action, SchedulerEnv
from qco.modules.rl_scheduler.features import CANDIDATE_DIM, GLOBAL_DIM, encode_state

POLICY_FORMAT = "qco-rl-ppo-v1"


def make_env(circuit: IntermediateRepresentation, config: RLConfig) -> SchedulerEnv:
    """The environment every learner/evaluator uses, built from one ``RLConfig``."""
    terminal_fn = None
    if config.reward_source == "pipeline":
        from qco.modules.rl_scheduler.closed_loop import get_pipeline_reward  # lazy: pulls in the pipeline

        terminal_fn = get_pipeline_reward(config.calibration, config.pipeline_resynthesis)
    env = SchedulerEnv(
        circuit,
        config.weights,
        max_episode_steps=config.max_episode_steps,
        calibration=config.calibration,
        hardware=config.hardware,
        commutation=config.commutation,
        reduction=config.deterministic_reduction,
        cr_window=config.cr_window,
        stop_on_noop=True,
        terminal_reward_fn=terminal_fn,
    )
    env.lookahead_metric = config.lookahead_metric
    return env


class PPOPolicy:
    """Action-scoring actor + state-value critic (JSON-serializable)."""

    def __init__(self, hidden: int = 16, seed: int = 0, backend: str = "auto", lookahead: bool = True):
        rng = random.Random(f"qco-ppo-init-{seed}")
        self.hidden = hidden
        self.lookahead = lookahead
        self.backend = resolve_backend(backend)
        self.actor = ScalarMLP(CANDIDATE_DIM, hidden, rng=rng, out_scale=0.01, backend=self.backend)
        self.critic = ScalarMLP(GLOBAL_DIM, hidden, rng=rng, out_scale=0.1, backend=self.backend)

    # -- inference ---------------------------------------------------------
    def logits(self, rows: list[list[float]]) -> list[float]:
        return self.actor.forward(rows)[0]

    def value(self, g: list[float]) -> float:
        return self.critic.forward([g])[0][0]

    def sample(self, rows: list[list[float]], rng: random.Random) -> tuple[int, float]:
        probs = softmax(self.logits(rows))
        u, acc = rng.random(), 0.0
        idx = len(probs) - 1
        for k, p in enumerate(probs):
            acc += p
            if u < acc:
                idx = k
                break
        return idx, math.log(max(probs[idx], 1e-300))

    def greedy(self, rows: list[list[float]]) -> int:
        z = self.logits(rows)
        return max(range(len(z)), key=z.__getitem__)

    # -- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": POLICY_FORMAT,
            "hidden": self.hidden,
            "lookahead": self.lookahead,
            "candidate_dim": CANDIDATE_DIM,
            "global_dim": GLOBAL_DIM,
            "actor": self.actor.theta,
            "critic": self.critic.theta,
        }

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto") -> "PPOPolicy":
        if d.get("format") != POLICY_FORMAT:
            raise ValueError(f"not a {POLICY_FORMAT} policy (format={d.get('format')!r})")
        if d["candidate_dim"] != CANDIDATE_DIM or d["global_dim"] != GLOBAL_DIM:
            raise ValueError(
                "policy was saved with a different feature layout "
                f"({d['candidate_dim']}/{d['global_dim']} vs {CANDIDATE_DIM}/{GLOBAL_DIM}); retrain it"
            )
        p = cls(hidden=d["hidden"], backend=backend, lookahead=d["lookahead"])
        p.actor.set_theta(d["actor"])
        p.critic.set_theta(d["critic"])
        return p

    def clone(self) -> "PPOPolicy":
        return PPOPolicy.from_dict(self.to_dict(), backend=self.backend)


# ----------------------------------------------------------------------------
# Running a policy on a circuit (shared by training, evaluation and schedule())
# ----------------------------------------------------------------------------

def select_candidates(actions: list[Action], cap: int | None, rng: random.Random | None) -> list[Action]:
    """Keep ``NOOP`` (always ``actions[0]``) plus at most ``cap - 1`` random other moves."""
    if cap is None or len(actions) <= cap:
        return actions
    rng = rng or random.Random(0)
    return [actions[0]] + rng.sample(actions[1:], cap - 1)


def run_policy(
    policy: PPOPolicy,
    circuit: IntermediateRepresentation,
    config: RLConfig,
    *,
    rng: random.Random | None = None,
    greedy: bool = True,
    cap: int | None = None,
) -> tuple[IntermediateRepresentation, dict]:
    """Roll ``policy`` out on ``circuit``; return ``(best circuit seen, stats)``.

    "Best" is lowest ``(depth, gate_count)`` over every state visited, so the result
    is never worse than the (CR-reduced) starting point. ``stats`` has the
    undiscounted return, step count and the CR gates removed up front.
    """
    env = make_env(circuit, config)
    _, info = env.reset()
    best = env.circuit
    best_key = (best.depth(), best.gate_count())
    total, steps = 0.0, 0
    while True:
        actions = select_candidates(info["legal_actions"], cap, rng)
        rows, _ = encode_state(env, actions, policy.lookahead)
        idx = policy.greedy(rows) if greedy else policy.sample(rows, rng or random.Random(0))[0]
        _, r, term, trunc, info = env.step(actions[idx])
        total += r
        steps += 1
        key = (env.circuit.depth(), env.circuit.gate_count())
        if key < best_key:
            best, best_key = env.circuit, key
        if term or trunc:
            break
    return best, {"return": total, "steps": steps, "cr_removed": env.cr_initial_removed, "reference": env.reference}


# ----------------------------------------------------------------------------
# PPO trainer
# ----------------------------------------------------------------------------

@dataclass(slots=True)
class Transition:
    rows: list[list[float]]
    g: list[float]
    action: int
    logp: float
    value: float
    reward: float
    adv: float = 0.0
    ret: float = 0.0


class PPOTrainer:
    def __init__(self, config: RLConfig, policy: PPOPolicy | None = None):
        self.config = config.validate()
        self.policy = policy or PPOPolicy(
            hidden=config.hidden, seed=config.seed, backend=config.numeric_backend, lookahead=config.lookahead
        )
        self.rng = random.Random(f"qco-ppo-run-{config.seed}")
        self.opt_actor = Adam(self.policy.actor.n_params, lr=config.lr, max_grad_norm=config.max_grad_norm)
        self.opt_critic = Adam(self.policy.critic.n_params, lr=config.lr, max_grad_norm=config.max_grad_norm)
        self.timesteps = 0
        self._cursor = 0

    # -- rollout -------------------------------------------------------------
    def _next_circuit(self, pool: list[IntermediateRepresentation]) -> IntermediateRepresentation:
        c = pool[self._cursor % len(pool)]
        self._cursor += 1
        return c

    def collect(self, pool: list[IntermediateRepresentation]) -> tuple[list[Transition], list[dict]]:
        """Collect >= ``n_steps`` transitions, always finishing the last episode."""
        cfg, pol = self.config, self.policy
        batch: list[Transition] = []
        episodes: list[dict] = []
        while len(batch) < cfg.n_steps:
            env = make_env(self._next_circuit(pool), cfg)
            _, info = env.reset()
            ref_depth, ref_gates = env.reference.depth(), env.reference.gate_count()
            start = len(batch)
            ep_return = 0.0
            terminal_reward = 0.0
            term = trunc = False
            last_rows = last_g = None
            while not (term or trunc):
                actions = select_candidates(info["legal_actions"], cfg.max_candidates, self.rng)
                rows, g = encode_state(env, actions, pol.lookahead)
                idx, logp = pol.sample(rows, self.rng)
                v = pol.value(g)
                _, r, term, trunc, info = env.step(actions[idx])
                batch.append(Transition(rows, g, idx, logp, v, r))
                ep_return += r
                if term or trunc:
                    terminal_reward = info["terminal_reward"]
            # bootstrap value of the final state (0 on true termination)
            if trunc and not term:
                actions = select_candidates(info["legal_actions"], cfg.max_candidates, self.rng)
                _, g_last = encode_state(env, actions, pol.lookahead)
                next_v = pol.value(g_last)
            else:
                next_v = 0.0
            self._gae(batch[start:], next_v)
            fin = env.circuit
            episodes.append(
                {
                    "return": ep_return,
                    "length": len(batch) - start,
                    "depth_red_pct": 100.0 * (ref_depth - fin.depth()) / ref_depth if ref_depth else 0.0,
                    "gate_red_pct": 100.0 * (ref_gates - fin.gate_count()) / ref_gates if ref_gates else 0.0,
                    "cr_removed": env.cr_initial_removed,
                    "terminal_reward": terminal_reward,
                }
            )
        self.timesteps += len(batch)
        return batch, episodes

    def _gae(self, ep: list[Transition], next_value: float) -> None:
        gamma, lam = self.config.gamma, self.config.gae_lambda
        gae = 0.0
        for t in range(len(ep) - 1, -1, -1):
            nv = next_value if t == len(ep) - 1 else ep[t + 1].value
            delta = ep[t].reward + gamma * nv - ep[t].value
            gae = delta + gamma * lam * gae
            ep[t].adv = gae
            ep[t].ret = gae + ep[t].value

    # -- update ----------------------------------------------------------------
    def update(self, batch: list[Transition]) -> dict:
        cfg, pol = self.config, self.policy
        advs = [t.adv for t in batch]
        mean = sum(advs) / len(advs)
        std = math.sqrt(sum((a - mean) ** 2 for a in advs) / len(advs)) + 1e-8
        for t in batch:
            t.adv = (t.adv - mean) / std

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_seen = 0
        order = list(range(len(batch)))
        for _ in range(cfg.epochs):
            self.rng.shuffle(order)
            for s in range(0, len(order), cfg.minibatch):
                mb = [batch[i] for i in order[s:s + cfg.minibatch]]
                ga = [0.0] * pol.actor.n_params
                gc = [0.0] * pol.critic.n_params
                m = len(mb)
                for t in mb:
                    z, cache = pol.actor.forward(t.rows)
                    p = softmax(z)
                    logp = math.log(max(p[t.action], 1e-300))
                    ratio = math.exp(max(-20.0, min(20.0, logp - t.logp)))
                    clipped = (t.adv >= 0 and ratio > 1 + cfg.clip) or (t.adv < 0 and ratio < 1 - cfg.clip)
                    surr = min(ratio * t.adv, max(min(ratio, 1 + cfg.clip), 1 - cfg.clip) * t.adv)
                    ent = -sum(pk * math.log(max(pk, 1e-300)) for pk in p)
                    dz = [0.0] * len(p)
                    if not clipped:
                        c = -t.adv * ratio / m
                        for k, pk in enumerate(p):
                            dz[k] += c * ((1.0 if k == t.action else 0.0) - pk)
                    if cfg.ent_coef:
                        for k, pk in enumerate(p):
                            dz[k] += cfg.ent_coef / m * pk * (math.log(max(pk, 1e-300)) + ent)
                    pol.actor.backward(cache, dz, ga)

                    v, vcache = pol.critic.forward([t.g])
                    err = v[0] - t.ret
                    pol.critic.backward(vcache, [cfg.vf_coef * err / m], gc)

                    stats["policy_loss"] += -surr
                    stats["value_loss"] += 0.5 * err * err
                    stats["entropy"] += ent
                    stats["approx_kl"] += t.logp - logp
                    stats["clip_frac"] += 1.0 if clipped else 0.0
                    n_seen += 1
                th = list(pol.actor.theta)
                self.opt_actor.step(th, ga)
                pol.actor.set_theta(th)
                th = list(pol.critic.theta)
                self.opt_critic.step(th, gc)
                pol.critic.set_theta(th)
        return {k: v / max(n_seen, 1) for k, v in stats.items()}


def save_policy(policy: PPOPolicy, path, extra: dict | None = None) -> None:
    """Atomically write ``policy`` (+ optional metadata) as JSON."""
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"policy": policy.to_dict(), **(extra or {})}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def load_policy(path, backend: str = "auto") -> tuple[PPOPolicy, dict]:
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "policy" not in payload:
        raise ValueError(f"{path} is not a qco RL checkpoint (missing 'policy')")
    return PPOPolicy.from_dict(payload["policy"], backend=backend), payload
