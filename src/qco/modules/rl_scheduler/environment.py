"""Module A — RL environment over the gate DAG.

Interfaces are frozen for Phase 2; the transition/reward bodies are Phase 3
(Maanas Nair). The environment follows the Gymnasium API so Stable-Baselines3 can
drive it directly.

State   : gate-DAG features (per-node gate type, qubit ids, depth position,
          commutation flags) + global (gate count, depth, est. fidelity).
Action  : (kind, i, j) where kind in {COMMUTE, MOVE_EARLIER, MOVE_LATER,
          MERGE_ROTATION, NOOP} over gate indices i, j.
Reward  : w_d * dDepth + w_g * dGateCount + w_f * dFidelity  (deltas vs. previous
          step); terminal bonus from the Stage 6 evaluation engine (closed loop).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from qco.graphs.dag import GateDAG, build_dag
from qco.ir.intermediate_representation import IntermediateRepresentation


class ActionKind(enum.IntEnum):
    NOOP = 0
    COMMUTE = 1
    MOVE_EARLIER = 2
    MOVE_LATER = 3
    MERGE_ROTATION = 4


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    i: int
    j: int = -1


@dataclass(slots=True)
class RewardWeights:
    depth: float = 1.0
    gate_count: float = 1.0
    fidelity: float = 0.5


class SchedulerEnv:
    """Gymnasium-style environment. ``reset`` / ``step`` land in Phase 3."""

    metadata = {"render_modes": []}

    def __init__(self, circuit: IntermediateRepresentation, weights: RewardWeights | None = None):
        self.initial = circuit.copy()
        self.circuit = circuit.copy()
        self.dag: GateDAG = build_dag(self.circuit)
        self.weights = weights or RewardWeights()
        self._steps = 0

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed: int | None = None):  # pragma: no cover - Phase 3
        raise NotImplementedError("SchedulerEnv.reset lands in Phase 3 (Maanas Nair)")

    def step(self, action: Action):  # pragma: no cover - Phase 3
        raise NotImplementedError("SchedulerEnv.step lands in Phase 3 (Maanas Nair)")

    # -- helpers already usable ------------------------------------------------
    def legal_actions(self) -> list[Action]:
        """Commuting pairs = adjacent gates on disjoint qubits (safe reorderings)."""
        acts: list[Action] = [Action(ActionKind.NOOP, -1)]
        gates = self.circuit.gates
        for i in range(len(gates) - 1):
            a, b = gates[i], gates[i + 1]
            if set(a.qubits).isdisjoint(b.qubits):
                acts.append(Action(ActionKind.COMMUTE, i, i + 1))
            if a.name == b.name and a.qubits == b.qubits and a.name in {"rx", "ry", "rz", "p"}:
                acts.append(Action(ActionKind.MERGE_ROTATION, i, i + 1))
        return acts

    def observation(self) -> dict:
        return {
            "gate_count": self.circuit.gate_count(),
            "depth": self.circuit.depth(),
            "two_qubit": self.circuit.two_qubit_gate_count(),
        }
