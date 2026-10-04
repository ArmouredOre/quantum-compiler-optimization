"""Module A — RL environment over the gate DAG.

Interfaces are frozen for Phase 2; the transition/reward bodies are Phase 3.
The environment follows the Gymnasium API so Stable-Baselines3 can
drive it directly.

State   : gate-DAG features (per-node gate type, qubit ids, depth position,
          commutation flags) + global (gate count, depth, est. fidelity).
Action  : (kind, i, j) where kind in {COMMUTE, MOVE_EARLIER, MOVE_LATER,
          MERGE_ROTATION, NOOP} over gate indices i, j.
Reward  : w_d * dDepth + w_g * dGateCount + w_f * dFidelity  (deltas vs. previous
          step); terminal bonus from the Stage 6 evaluation engine (closed loop).

Implementation notes for Phase 3
-------------------------------
* Actual gate-list rewrites live in :mod:`qco.modules.rl_scheduler.rewrites`,
  kept separate so each rewrite (commute / move / merge-rotation) is
  independently unit-testable without spinning up a Gym episode.
* Reward uses **per-step deltas** in gate count / depth (matching the
  docstring above) plus a fidelity term, scaled by ``RewardWeights``; the
  *episode-terminal* bonus additionally folds in
  ``qco.evaluation.metrics.scalarized_reward(initial, final)`` — the "Stage 6
  evaluation engine ... closed loop" this docstring already promises — so an
  episode that nets out worse than it started (possible via COMMUTE-only
  reordering followed by a bad MERGE) is penalized once, at the end, on the
  same five-parameter scale the rest of the pipeline reports against.
* Observation gains a fixed-shape ``gate_features`` / ``gate_mask`` pair (see
  ``qco.modules.rl_scheduler.spaces.make_observation_space``) alongside the
  original scalar dict, so both a quick/no-dependency inspection
  (``observation()["gate_count"]`` etc., unchanged) and an SB3 policy
  (``observation()["gate_features"]``) are served by the same call.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from qco.evaluation.metrics import estimated_fidelity, scalarized_reward
from qco.graphs.dag import GateDAG, build_dag
from qco.ir.intermediate_representation import ARITY, PARAMETRIC, IntermediateRepresentation
from qco.modules.rl_scheduler.rewrites import IllegalActionError, apply_action


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
    # Weight applied to the episode-terminal ``scalarized_reward`` bonus (see
    # module docstring). Zero disables the terminal bonus entirely, which is
    # useful for unit tests that only want to check the per-step shaping term.
    terminal: float = 1.0


# Fixed gate-name -> small-int vocabulary for the ``gate_features`` observation
# array. Built once from ``ARITY`` (the IR's own gate registry) so this file
# never drifts out of sync with which gates the IR actually supports; index 0
# is reserved for "unknown / padding" rather than aliasing a real gate.
_GATE_VOCAB: dict[str, int] = {name: idx + 1 for idx, name in enumerate(sorted(ARITY))}


class MaxStepsExceeded(RuntimeError):
    """Raised if ``step`` is called after an episode has already terminated."""


class SchedulerEnv:
    """Gymnasium-style environment. ``reset`` / ``step`` land in Phase 3."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        circuit: IntermediateRepresentation,
        weights: RewardWeights | None = None,
        *,
        max_episode_steps: int = 200,
        calibration: str = "default",
    ):
        self.initial = circuit.copy()
        self.circuit = circuit.copy()
        self.dag: GateDAG = build_dag(self.circuit)
        self.weights = weights or RewardWeights()
        self.max_episode_steps = max_episode_steps
        self.calibration = calibration
        self._steps = 0
        self._done = False
        # Deltas are computed against theprevious step's circuit, not the
        # episode's initial one, so reward reflects the marginal effect of the
        # action just taken rather than re-scoring the whole episode each step.
        self._prev_gate_count = self.circuit.gate_count()
        self._prev_depth = self.circuit.depth()
        self._prev_fidelity = estimated_fidelity(self.circuit, self.calibration).value

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed: int | None = None):
        """Restore the circuit to its initial state and return ``(obs, info)``.

        ``seed`` currently only affects downstream random *policies* that read
        it themselves (e.g. a random-action baseline seeding its own RNG from
        this call) — the environment's own transition/reward functions are
        deterministic given a fixed circuit and action sequence, so there is
        no internal RNG state to seed yet. Kept as a no-op-but-accepted
        parameter to match the Gymnasium ``reset(seed=...)`` signature SB3
        relies on, rather than dropping the keyword and breaking that contract
        (see docs/gaps_and_notes_module_a.md).
        """
        self.circuit = self.initial.copy()
        self.dag = build_dag(self.circuit)
        self._steps = 0
        self._done = False
        self._prev_gate_count = self.circuit.gate_count()
        self._prev_depth = self.circuit.depth()
        self._prev_fidelity = estimated_fidelity(self.circuit, self.calibration).value
        info: dict = {"legal_actions": self.legal_actions()}
        return self.observation(), info

    def step(self, action: Action):
        """Apply ``action``, return ``(obs, reward, terminated, truncated, info)``.

        * ``action`` need not be pre-validated by the caller: an
          out-of-range/no-longer-legal action is treated as a ``NOOP`` (reward
          0 for that step, episode continues) rather than raising, matching
          Gymnasium's convention that a policy exploring randomly shouldn't
          crash the episode — see ``IllegalActionError`` handling below and
          the equivalent fallback in ``spaces.decode_action`` for the
          fixed-size SB3 adapter.
        * ``terminated`` is only ``True`` when the circuit reaches a fixed
          point (no legal action left besides NOOP) — a genuine "nothing more
          to optimize" end state. ``truncated`` is ``True`` once
          ``max_episode_steps`` is hit — a training-budget cutoff, not a
          claim the circuit is optimal (the Gymnasium-standard distinction
          SB3's ``VecEnv`` machinery expects).
        """
        if self._done:
            raise MaxStepsExceeded(
                "step() called after episode termination; call reset() first"
            )

        before = self.circuit
        try:
            after = apply_action(before, action.kind.name, action.i, action.j)
        except IllegalActionError:
            after = before.copy()

        self.circuit = after
        self.dag = build_dag(self.circuit)
        self._steps += 1

        new_gate_count = self.circuit.gate_count()
        new_depth = self.circuit.depth()
        new_fidelity = estimated_fidelity(self.circuit, self.calibration).value

        d_gate_count = self._prev_gate_count - new_gate_count
        d_depth = self._prev_depth - new_depth
        d_fidelity = new_fidelity - self._prev_fidelity

        reward = (
            self.weights.gate_count * d_gate_count
            + self.weights.depth * d_depth
            + self.weights.fidelity * d_fidelity
        )

        self._prev_gate_count = new_gate_count
        self._prev_depth = new_depth
        self._prev_fidelity = new_fidelity

        terminated = self._is_fixed_point()
        truncated = self._steps >= self.max_episode_steps
        if terminated or truncated:
            self._done = True
            reward += self.weights.terminal * scalarized_reward(
                self.initial, self.circuit, calibration=self.calibration
            )

        info = {
            "legal_actions": self.legal_actions(),
            "d_gate_count": d_gate_count,
            "d_depth": d_depth,
            "d_fidelity": d_fidelity,
        }
        return self.observation(), reward, terminated, truncated, info

    def _is_fixed_point(self) -> bool:
        """True once ``legal_actions()`` only contains NOOP: nothing left to try."""
        return len(self.legal_actions()) <= 1

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
        """Scalar summary (unchanged Phase-2 keys) plus fixed-shape NN features.

        ``gate_features[k] = [gate_type_id, qubit_a, qubit_b, is_two_qubit,
        is_parametric]`` for gate ``k`` (``qubit_b = -1`` for single-qubit
        gates); rows past the current gate count are left as ``0`` and masked
        out via ``gate_mask``. Padded to ``max_gates`` so callers get a
        constant-shape array regardless of circuit size — required for a
        batched SB3 policy, harmless for anything reading the scalar keys only.
        """
        gates = self.circuit.gates
        max_gates = max(len(gates), 1)
        gate_features = [[0.0, -1.0, -1.0, 0.0, 0.0] for _ in range(max_gates)]
        gate_mask = [0.0] * max_gates
        for idx, g in enumerate(gates):
            qa = g.qubits[0] if len(g.qubits) > 0 else -1
            qb = g.qubits[1] if len(g.qubits) > 1 else -1
            gate_features[idx] = [
                float(_GATE_VOCAB.get(g.name, 0)),
                float(qa),
                float(qb),
                1.0 if g.is_two_qubit else 0.0,
                1.0 if g.name in PARAMETRIC else 0.0,
            ]
            gate_mask[idx] = 1.0

        return {
            "gate_count": self.circuit.gate_count(),
            "depth": self.circuit.depth(),
            "two_qubit": self.circuit.two_qubit_gate_count(),
            "estimated_fidelity": self._prev_fidelity,
            "gate_features": gate_features,
            "gate_mask": gate_mask,
        }