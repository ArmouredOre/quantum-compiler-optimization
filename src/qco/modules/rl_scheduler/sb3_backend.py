"""Module A — optional Stable-Baselines3 backend (``pip install -e ".[rl]"``).

``SlotEnv`` exposes the scheduler as a standard Gymnasium env with a
``Discrete(K)`` action space where *slot k* means "the k-th legal action
(slot 0 is always NOOP)", plus ``action_masks()`` so sb3-contrib's
``MaskablePPO`` never samples an illegal slot. This replaces the Sprint 1
``MultiDiscrete`` adapter (``spaces.py``) for training, which wastes most
samples on illegal ``(kind, i, j)`` triples. Observation = ``K`` candidate
feature rows (see ``features.py``) flattened, plus the global features.

Imports of gymnasium / sb3 happen lazily so importing this module on a
bare install raises a clear ImportError only when the backend is used.
"""

from __future__ import annotations

import random
from pathlib import Path

from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.features import CANDIDATE_DIM, GLOBAL_DIM, encode_state
from qco.modules.rl_scheduler.ppo import make_env, select_candidates

try:
    import gymnasium as gym
    import numpy as np

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    gym = np = None
    GYMNASIUM_AVAILABLE = False

try:
    from sb3_contrib import MaskablePPO

    SB3_AVAILABLE = GYMNASIUM_AVAILABLE
except ImportError:  # pragma: no cover
    MaskablePPO = None
    SB3_AVAILABLE = False


def _require() -> None:
    if not SB3_AVAILABLE:
        raise ImportError("the 'sb3' backend needs gymnasium, numpy, torch, stable-baselines3 and sb3-contrib: pip install -e '.[rl]'")


_Base = gym.Env if GYMNASIUM_AVAILABLE else object


class SlotEnv(_Base):
    """Gymnasium wrapper: slot-indexed discrete actions with a legality mask."""

    metadata = {"render_modes": []}

    def __init__(self, circuits, config: RLConfig, slots: int | None = None):
        _require()
        super().__init__()
        self.pool = list(circuits)
        self.config = config
        self.K = slots or (config.max_candidates or 48)
        self.action_space = gym.spaces.Discrete(self.K)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.K * CANDIDATE_DIM + GLOBAL_DIM,), dtype=np.float32)
        self._rng = random.Random(config.seed)
        self._cursor = 0
        self.env = None
        self._actions: list = []

    def _observe(self, info):
        self._actions = select_candidates(info["legal_actions"], self.K, self._rng)
        rows, g = encode_state(self.env, self._actions, self.config.lookahead)
        flat = np.zeros(self.K * CANDIDATE_DIM + GLOBAL_DIM, dtype=np.float32)
        for k, r in enumerate(rows):
            flat[k * CANDIDATE_DIM:(k + 1) * CANDIDATE_DIM] = r
        flat[-GLOBAL_DIM:] = g
        return flat

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        circuit = self.pool[self._cursor % len(self.pool)]
        self._cursor += 1
        self.env = make_env(circuit, self.config)
        _, info = self.env.reset()
        return self._observe(info), {}

    def step(self, action):
        k = int(action)
        a = self._actions[k] if 0 <= k < len(self._actions) else self._actions[0]  # masked-out slot -> NOOP
        _, r, term, trunc, info = self.env.step(a)
        return self._observe(info), float(r), bool(term), bool(trunc), {}

    def action_masks(self):
        m = np.zeros(self.K, dtype=bool)
        m[: len(self._actions)] = True
        return m


def train_sb3(config: RLConfig, out_dir: str | Path | None = None, *, log=print):
    """Train ``MaskablePPO`` on generated circuits; returns the model (saved to ``<out>/sb3_model.zip``)."""
    _require()
    from sb3_contrib.common.wrappers import ActionMasker  # noqa: F401  (documented extension point)

    from qco.modules.rl_scheduler.training import build_pools

    config.validate()
    train_pool, _ = build_pools(config)
    env = SlotEnv(train_pool, config)
    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=config.lr, n_steps=config.n_steps, batch_size=config.minibatch,
        n_epochs=config.epochs, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm, seed=config.seed, verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=config.total_timesteps)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out / "sb3_model"))
        config.to_yaml(out / "config.yaml")
    if log:
        log(f"[sb3] trained {config.total_timesteps} timesteps")
    return model


def run_sb3_policy(model, circuit, config: RLConfig):
    """Greedy (deterministic, masked) rollout of a trained ``MaskablePPO``.

    Returns the best ``(depth, gate_count)`` circuit seen, like ``ppo.run_policy``.
    """
    _require()
    env = SlotEnv([circuit], config)
    obs, _ = env.reset()
    best = env.env.circuit
    best_key = (best.depth(), best.gate_count())
    while True:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, term, trunc, _ = env.step(int(action))
        c = env.env.circuit
        key = (c.depth(), c.gate_count())
        if key < best_key:
            best, best_key = c, key
        if term or trunc:
            return best


def load_sb3(path: str | Path):
    """Load a model saved by :func:`train_sb3` (``<dir>/sb3_model.zip``)."""
    _require()
    return MaskablePPO.load(str(path))
