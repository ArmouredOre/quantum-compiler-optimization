"""Module A — PPO/DDQN agent wrapper (Member 2, Phase 3).

Design (from the literature survey): pair the learned policy with a deterministic
Commutation-and-Reduction pass so the agent spends capacity on non-trivial
rewrites and generalizes from small training circuits (Tao et al. 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler.environment import RewardWeights, SchedulerEnv


@dataclass(slots=True)
class RLConfig:
    algo: str = "ppo"            # "ppo" | "ddqn"
    total_timesteps: int = 200_000
    train_qubits: tuple[int, ...] = (5,)
    eval_qubits: tuple[int, ...] = (10, 20, 40)
    weights: RewardWeights = field(default_factory=RewardWeights)
    deterministic_reduction: bool = True   # the CR hybrid pass


class RLScheduler:
    def __init__(self, config: RLConfig | None = None):
        self.config = config or RLConfig()
        self._model = None

    def train(self, circuits: list[IntermediateRepresentation]) -> None:  # pragma: no cover - Phase 3
        raise NotImplementedError("RLScheduler.train lands in Phase 3 (Member 2)")

    def load(self, path: str) -> "RLScheduler":  # pragma: no cover - Phase 3
        raise NotImplementedError

    def schedule(self, circuit: IntermediateRepresentation) -> IntermediateRepresentation:
        """Return a reordered/commuted circuit.

        Phase 2 stand-in: bounded left-to-right passes applying an adjacent
        commute only when it *strictly* reduces circuit depth, so the pipeline
        produces a correct, terminating result before the agent is trained.
        """
        current = circuit.copy()
        for _ in range(current.num_qubits + 1):          # bounded number of sweeps
            changed = False
            i = 0
            while i < len(current.gates) - 1:
                a, b = current.gates[i], current.gates[i + 1]
                if set(a.qubits).isdisjoint(b.qubits):
                    trial = current.copy()
                    trial.gates[i], trial.gates[i + 1] = trial.gates[i + 1], trial.gates[i]
                    if trial.depth() < current.depth():
                        current = trial
                        changed = True
                i += 1
            if not changed:
                break
        return current
