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

from qco.evaluation.hardware import HardwareProfile, get_hardware, hardware_execution_time, hardware_fidelity
from qco.evaluation.metrics import estimated_fidelity, execution_time, scalarized_reward
from qco.graphs.dag import GateDAG, build_dag
from qco.ir.intermediate_representation import ARITY, PARAMETRIC, IntermediateRepresentation
from qco.modules.rl_scheduler.commutation import commutes, validate_mode
from qco.modules.rl_scheduler.cr_pass import reduce_with_stats
from qco.modules.rl_scheduler.rewrites import IllegalActionError, apply_action, flippable_edges


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
    # Sprint 3 (#29): weight on the per-step drop in *calibrated* execution time
    # (critical-path ns under ``calibration``), expressed in "average layers" so it
    # is on the same scale as ``depth``. 0.0 (default) keeps the old reward exactly.
    # This is what lets policies trained under different device profiles differ:
    # depth and gate-count rewards are backend-independent, execution time is not.
    time: float = 0.0


# Fixed gate-name -> small-int vocabulary for the ``gate_features`` observation
# array. Built once from ``ARITY`` (the IR's own gate registry) so this file
# never drifts out of sync with which gates the IR actually supports; index 0
# is reserved for "unknown / padding" rather than aliasing a real gate.
_GATE_VOCAB: dict[str, int] = {name: idx + 1 for idx, name in enumerate(sorted(ARITY))}


class MaxStepsExceeded(RuntimeError):
    """Raised if ``step`` is called after an episode has already terminated."""


class SchedulerEnv:
    """Gymnasium-style environment over the gate list.

    Sprint 2 adds three **opt-in** keyword arguments (all default to the Sprint 1
    behaviour, so existing callers and tests are unaffected):

    ``commutation``
        ``"disjoint"`` (default) lets ``COMMUTE`` swap only adjacent gates on
        disjoint qubits, which never changes the dependency DAG. ``"full"`` also
        allows adjacent gates that share qubits but provably commute (see
        :mod:`qco.modules.rl_scheduler.commutation`) — those swaps change the
        DAG, so they are what can actually reduce depth.
    ``reduction``
        Run the deterministic Commutation-and-Reduction pass
        (:mod:`qco.modules.rl_scheduler.cr_pass`) on the starting circuit and
        after every action, so the agent only ever sees circuits with no
        trivial cancellations left. The terminal bonus is then measured against
        the *CR-reduced* start, i.e. the agent is credited only for what it adds
        on top of CR.
    ``stop_on_noop``
        Treat a ``NOOP`` action as "I'm done" and terminate the episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        circuit: IntermediateRepresentation,
        weights: RewardWeights | None = None,
        *,
        max_episode_steps: int = 200,
        calibration: str = "default",
        hardware: "str | HardwareProfile | None" = None,
        commutation: str = "disjoint",
        reduction: bool = False,
        cr_window: int | None = None,
        stop_on_noop: bool = False,
    ):
        self.initial = circuit.copy()
        self.weights = weights or RewardWeights()
        self.max_episode_steps = max_episode_steps
        # Opt-in (Sprint 3 follow-up): a HardwareProfile adds topology / SWAP-routing /
        # native-gate costs to the time and fidelity terms. None = old Calibration-only behaviour.
        self.hardware = get_hardware(hardware) if hardware is not None else None
        self.calibration = self.hardware.calibration if self.hardware else calibration
        self.commutation = validate_mode(commutation)
        self.reduction = reduction
        self.cr_window = cr_window
        self.stop_on_noop = stop_on_noop
        self._start_episode()

    def _start_episode(self) -> None:
        """(Re)initialize all per-episode state from ``self.initial``."""
        start = self.initial.copy()
        self.cr_initial_removed = 0
        if self.reduction:
            start, stats = reduce_with_stats(start, window=self.cr_window)
            self.cr_initial_removed = stats.gates_removed
        self.circuit = start
        # Baseline the terminal bonus is measured against: the original circuit,
        # or (with ``reduction``) the CR-reduced start.
        self.reference = start.copy()
        self.dag: GateDAG = build_dag(self.circuit)
        self._steps = 0
        self._done = False
        # Deltas are computed against the previous step's circuit, not the
        # episode's initial one, so reward reflects the marginal effect of the
        # action just taken rather than re-scoring the whole episode each step.
        self._prev_gate_count = self.circuit.gate_count()
        self._prev_depth = self.circuit.depth()
        self._prev_fidelity = self.fidelity_of(self.circuit)
        self._prev_time = self.time_of(self.circuit)
        # one "layer-equivalent" in ns: average critical-path time per layer of the start
        self.time_unit = max(self._prev_time / max(self.circuit.depth(), 1), 1e-9)

    # -- backend-aware metrics (hardware profile if set, else the Calibration) ---
    def time_of(self, circuit: IntermediateRepresentation) -> float:
        if self.hardware is not None:
            return hardware_execution_time(circuit, self.hardware)
        return execution_time(circuit, self.calibration).value

    def fidelity_of(self, circuit: IntermediateRepresentation) -> float:
        if self.hardware is not None:
            return hardware_fidelity(circuit, self.hardware)
        return estimated_fidelity(circuit, self.calibration).value

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
        self._start_episode()
        info: dict = {
            "legal_actions": self.legal_actions(),
            "cr_removed": self.cr_initial_removed,
        }
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
            after = apply_action(before, action.kind.name, action.i, action.j, self.commutation)
        except IllegalActionError:
            after = before.copy()

        cr_removed = 0
        if self.reduction:
            after, cr_stats = reduce_with_stats(after, window=self.cr_window)
            cr_removed = cr_stats.gates_removed

        self.circuit = after
        self.dag = build_dag(self.circuit)
        self._steps += 1

        new_gate_count = self.circuit.gate_count()
        new_depth = self.circuit.depth()
        new_fidelity = self.fidelity_of(self.circuit)

        d_gate_count = self._prev_gate_count - new_gate_count
        d_depth = self._prev_depth - new_depth
        d_fidelity = new_fidelity - self._prev_fidelity
        new_time = self.time_of(self.circuit) if self.weights.time else self._prev_time
        d_time = (self._prev_time - new_time) / self.time_unit

        reward = (
            self.weights.gate_count * d_gate_count
            + self.weights.depth * d_depth
            + self.weights.fidelity * d_fidelity
            + self.weights.time * d_time
        )

        self._prev_gate_count = new_gate_count
        self._prev_depth = new_depth
        self._prev_fidelity = new_fidelity
        self._prev_time = new_time

        terminated = (self.stop_on_noop and action.kind == ActionKind.NOOP) or self._is_fixed_point()
        truncated = self._steps >= self.max_episode_steps
        if terminated or truncated:
            self._done = True
            reward += self.weights.terminal * scalarized_reward(
                self.reference, self.circuit, calibration=self.calibration
            )

        info = {
            "legal_actions": self.legal_actions(),
            "d_gate_count": d_gate_count,
            "d_depth": d_depth,
            "d_fidelity": d_fidelity,
            "d_time": d_time,
            "cr_removed": cr_removed,
        }
        return self.observation(), reward, terminated, truncated, info

    def _is_fixed_point(self) -> bool:
        """True once ``legal_actions()`` only contains NOOP: nothing left to try."""
        return len(self.legal_actions()) <= 1

    # -- helpers already usable ------------------------------------------------
    def legal_actions(self) -> list[Action]:
        """Legal reorderings plus adjacent mergeable rotations (and ``NOOP``).

        ``commutation="disjoint"`` (default, Sprint 1): adjacent gates on disjoint
        qubits — swapping them never changes the dependency DAG.

        ``commutation="full"``: every *flippable dependency edge* — two gates
        consecutive on a shared qubit that provably commute and can be reordered
        by hoisting/sinking past commuting gates. Each offered ``COMMUTE(i, j)``
        therefore changes the DAG (and so possibly the depth). Swapping two
        *identical* gates is a no-op and is never offered.
        """
        acts: list[Action] = [Action(ActionKind.NOOP, -1)]
        gates = self.circuit.gates
        if self.commutation == "full":
            # DAG-edge flips: every offered COMMUTE changes the dependency DAG.
            for i, j in flippable_edges(gates, "full"):
                acts.append(Action(ActionKind.COMMUTE, i, j))
        else:
            for i in range(len(gates) - 1):
                a, b = gates[i], gates[i + 1]
                if a != b and commutes(a, b, "disjoint"):
                    acts.append(Action(ActionKind.COMMUTE, i, i + 1))
        for i in range(len(gates) - 1):
            a, b = gates[i], gates[i + 1]
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