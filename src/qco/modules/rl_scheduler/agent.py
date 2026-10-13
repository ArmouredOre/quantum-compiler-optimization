"""Module A — PPO agent wrapper (Maanas Nair).

Design (from the literature survey): pair the learned policy with a deterministic
Commutation-and-Reduction pass so the agent spends capacity on non-trivial
rewrites and generalizes from small training circuits (Tao et al. 2026).

Sprint 2: ``RLConfig`` moved to :mod:`qco.modules.rl_scheduler.config` (re-exported
here, so old imports keep working); ``RLScheduler.train`` / ``save`` / ``load``
are real; ``schedule`` uses the trained policy when there is one and otherwise
keeps the Phase 2 greedy pass (pinned by the pipeline tests).
"""

from __future__ import annotations

from pathlib import Path

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.cr_pass import reduce_circuit

__all__ = ["RLConfig", "RLScheduler"]


class RLScheduler:
    def __init__(self, config: RLConfig | None = None):
        self.config = config or RLConfig()
        self._model = None

    def reduce(self, circuit: IntermediateRepresentation) -> IntermediateRepresentation:
        """The deterministic CR pass alone (no learning)."""
        return reduce_circuit(circuit)

    def train(self, circuits: list[IntermediateRepresentation] | None = None, out_dir: str | Path | None = None, log=print):
        """Train the PPO policy (``config.backend`` = "builtin" or "sb3").

        ``circuits`` is accepted for interface compatibility; training circuits are
        generated from ``config`` (see ``circuits.py``) so runs are reproducible.
        Returns the per-iteration history (built-in backend).
        """
        if self.config.algo != "ppo":
            raise NotImplementedError("algo='ddqn' is not implemented; use algo='ppo'")
        if self.config.backend == "sb3":
            from qco.modules.rl_scheduler.sb3_backend import train_sb3

            self._model = train_sb3(self.config, out_dir, log=log)
            return []
        from qco.modules.rl_scheduler.training import train

        self._model, history = train(self.config, out_dir, log=log)
        return history

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("nothing to save: train() or load() first")
        from qco.modules.rl_scheduler.ppo import save_policy

        save_policy(self._model, path, {"config": self.config.to_dict()})

    def load(self, path: str | Path) -> "RLScheduler":
        from qco.modules.rl_scheduler.ppo import load_policy

        self._model, payload = load_policy(path, self.config.numeric_backend)
        if "config" in payload:
            self.config = RLConfig.from_dict(payload["config"])
        return self

    def schedule(self, circuit: IntermediateRepresentation) -> IntermediateRepresentation:
        """Return a reordered/commuted circuit.

        With a trained policy (``train()`` / ``load()``): CR pass, then the greedy
        policy rollout (best circuit seen, never worse than the CR output).

        Untrained Phase 2 stand-in: bounded left-to-right passes applying an adjacent
        commute only when it *strictly* reduces circuit depth, so the pipeline
        produces a correct, terminating result before the agent is trained.
        """
        if self._model is not None:
            from qco.modules.rl_scheduler.ppo import run_policy

            out, _ = run_policy(self._model, circuit, self.config, greedy=True)
            return out
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
