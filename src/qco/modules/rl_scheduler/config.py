"""Module A — ``RLConfig``: every knob of the PPO agent and training loop.

Lives in its own module so ``agent.py`` (which imports the trainer) and the
trainer itself can both import it without a cycle; ``agent.RLConfig`` re-exports
it, so ``from qco.modules.rl_scheduler.agent import RLConfig`` keeps working.

The first six fields are the Phase 2 interface and keep their names and defaults.
Everything below them is Sprint 2. Configs round-trip through plain dicts / YAML
(``to_dict`` / ``from_dict`` / ``from_yaml`` / ``to_yaml``) so a training run is
fully described by one small file (see ``configs/*.yaml``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from qco.modules.rl_scheduler.environment import RewardWeights

ALGOS = ("ppo", "ddqn")
BACKENDS = ("builtin", "sb3")


@dataclass(slots=True)
class RLConfig:
    # ---- Phase 2 interface (unchanged) ------------------------------------
    algo: str = "ppo"            # "ppo" | "ddqn" (ddqn is not implemented yet)
    total_timesteps: int = 200_000
    train_qubits: tuple[int, ...] = (5,)
    eval_qubits: tuple[int, ...] = (10, 20, 40)
    weights: RewardWeights = field(default_factory=RewardWeights)
    deterministic_reduction: bool = True   # the CR hybrid pass

    # ---- Sprint 2: which learner ------------------------------------------
    backend: str = "builtin"     # "builtin" (pure-Python PPO) | "sb3" (MaskablePPO, needs the `rl` extra)
    seed: int = 0
    numeric_backend: str = "auto"  # builtin PPO maths: "auto" | "python" | "numpy"

    # ---- environment -------------------------------------------------------
    commutation: str = "full"    # "disjoint" (Sprint 1) | "full" (provably-commuting gates may be reordered)
    max_episode_steps: int = 40
    calibration: str = "default"
    cr_window: int | None = None
    lookahead: bool = True       # give the policy the one-step depth delta of each candidate move

    # ---- training circuits (generated on the fly; disjoint from benchmarks/) ---
    circuit_families: tuple[str, ...] = ("clifford_t", "cnot_pauli", "qaoa", "qft")
    circuit_layers: int = 12
    n_train_circuits: int = 24
    n_eval_circuits: int = 12

    # ---- PPO ---------------------------------------------------------------
    hidden: int = 16
    n_steps: int = 1024          # environment steps collected per PPO iteration
    epochs: int = 4
    minibatch: int = 64
    lr: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    max_candidates: int | None = 48   # cap on legal actions scored per step *during training*

    # ---- logging / checkpoints ---------------------------------------------
    eval_every: int = 5          # PPO iterations between held-out evaluations (0 = never)
    checkpoint_every: int = 5    # PPO iterations between ``latest.json`` writes (0 = only at the end)

    # ------------------------------------------------------------------
    def validate(self) -> "RLConfig":
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, got {self.algo!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.commutation not in ("disjoint", "full"):
            raise ValueError(f"commutation must be 'disjoint' or 'full', got {self.commutation!r}")
        if self.total_timesteps <= 0 or self.n_steps <= 0 or self.minibatch <= 0 or self.epochs <= 0:
            raise ValueError("total_timesteps, n_steps, minibatch and epochs must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not self.train_qubits or any(q < 2 for q in self.train_qubits):
            raise ValueError("train_qubits must be a non-empty tuple of ints >= 2")
        if self.max_candidates is not None and self.max_candidates < 2:
            raise ValueError("max_candidates must be None or >= 2 (NOOP plus at least one move)")
        return self

    # -- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "RLConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown RLConfig field(s): {', '.join(unknown)}")
        kwargs = dict(data)
        if "weights" in kwargs and not isinstance(kwargs["weights"], RewardWeights):
            kwargs["weights"] = RewardWeights(**kwargs["weights"])
        for name in ("train_qubits", "eval_qubits", "circuit_families"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLConfig":
        import yaml  # core dependency of the project

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
