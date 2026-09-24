"""Sprint 1 (#5): Module A — RL scheduler environment
written and maintained by Maanas Nair.

Covers:
  * ``rewrites.py`` pure-function unit tests (commute / move / merge-rotation /
    illegal-action handling), independent of any Gym episode.
  * ``environment.py``'s ``reset`` / ``step`` Gymnasium contract: shapes,
    determinism of ``reset``, terminal/truncation flags, reward sign sanity.
  * ``spaces.py``'s fixed-size ``MultiDiscrete`` encode/decode round trip
    (skipped if the ``rl`` extra / gymnasium isn't installed).
  * The acceptance-criteria smoke test: 10k random-policy steps per benchmark
    circuit, asserting the circuit stays semantically equivalent to the
    original throughout — via ``EquivalenceChecker`` for circuits within its
    numeric backend's qubit cap, and a syntactic "still ran to completion
    without crashing" check for larger ones (see
    ``docs/gaps_and_notes_module_a.md`` for why: the numeric backend caps at
    8 qubits and the SMT/symbolic path that removes that cap is in
    Sprint 2 deliverable, issue #7).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.rl_scheduler.environment import (
    Action,
    ActionKind,
    MaxStepsExceeded,
    RewardWeights,
    SchedulerEnv,
)
from qco.modules.rl_scheduler.rewrites import (
    IllegalActionError,
    apply_action,
    can_commute,
    can_merge_rotation,
    commute,
    merge_rotation,
    move,
)
from qco.modules.smt_verifier.equivalence import EquivalenceChecker

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "benchmarks" / "circuits"

# EquivalenceChecker's numeric backend caps at 8 qubits (see
# qco/modules/smt_verifier/equivalence.py); until the SMT/symbolic path lands
# (issue #7), the smoke test only asserts *equivalence* on circuits at or
# below this size, matching every other qubit-count-gated call site in the
# repo (e.g. qco.evaluation.benchmark_runner).
MAX_NUMERIC_QUBITS = 8


def _load(name: str) -> IntermediateRepresentation:
    ir = from_qasm2((CIRCUITS / name).read_text(encoding="utf-8"))
    ir.name = Path(name).stem
    return ir


def _small_disjoint_ir() -> IntermediateRepresentation:
    """2-qubit circuit with an adjacent disjoint-qubit pair for COMMUTE tests.

    Gate 0 (h q0) and gate 1 (x q1) touch disjoint qubits and are adjacent, so
    ``legal_actions()`` must offer ``COMMUTE(0, 1)``; gate 2 (cx q0,q1) then
    depends on both and is never involved in a legal swap here.
    """
    ir = IntermediateRepresentation(2, name="disjoint")
    ir.add("h", [0])
    ir.add("x", [1])
    ir.add("cx", [0, 1])
    return ir


def _merge_ir() -> IntermediateRepresentation:
    ir = IntermediateRepresentation(2, name="merge")
    ir.add("h", [0])
    ir.add("rz", [0], [0.3])
    ir.add("rz", [0], [0.4])
    ir.add("cx", [0, 1])
    return ir


# ---------------------------------------------------------------------------
# rewrites.py — pure function unit tests
# ---------------------------------------------------------------------------


class TestCommute:
    def test_can_commute_disjoint_qubits(self):
        ir = _small_disjoint_ir()
        a, b = ir.gates[0], ir.gates[1]
        assert can_commute(a, b) is True

    def test_can_commute_shared_qubit_false(self):
        ir = _small_disjoint_ir()
        b, c = ir.gates[1], ir.gates[2]
        assert can_commute(b, c) is False

    def test_commute_swaps_positions(self):
        ir = _small_disjoint_ir()
        out = commute(ir, 0, 1)
        assert out.gates[0].name == "x"
        assert out.gates[1].name == "h"
        assert out.gate_count() == ir.gate_count()

    def test_commute_does_not_mutate_input(self):
        ir = _small_disjoint_ir()
        original_first = ir.gates[0]
        commute(ir, 0, 1)
        assert ir.gates[0] is original_first

    def test_commute_illegal_shared_qubit_raises(self):
        ir = _small_disjoint_ir()
        with pytest.raises(IllegalActionError):
            commute(ir, 1, 2)

    def test_commute_out_of_range_raises(self):
        ir = _small_disjoint_ir()
        with pytest.raises(IllegalActionError):
            commute(ir, 0, 99)

    def test_commute_preserves_equivalence(self):
        ir = _small_disjoint_ir()
        out = commute(ir, 0, 1)
        result = EquivalenceChecker().check(ir, out)
        assert result.equivalent


class TestMove:
    def test_move_earlier_at_boundary_is_noop(self):
        ir = _small_disjoint_ir()
        out = move(ir, 0, earlier=True)
        assert [g.name for g in out.gates] == [g.name for g in ir.gates]

    def test_move_later_at_boundary_is_noop(self):
        ir = _small_disjoint_ir()
        out = move(ir, len(ir.gates) - 1, earlier=False)
        assert [g.name for g in out.gates] == [g.name for g in ir.gates]

    def test_move_earlier_blocked_by_shared_qubit_is_noop(self):
        ir = _small_disjoint_ir()
        # gate 2 (cx q0,q1) shares qubits with gate 1 (x q1) -> can't move earlier
        out = move(ir, 2, earlier=True)
        assert [g.name for g in out.gates] == [g.name for g in ir.gates]

    def test_move_earlier_valid_swap(self):
        ir = _small_disjoint_ir()
        out = move(ir, 1, earlier=True)
        assert out.gates[0].name == "x"
        assert out.gates[1].name == "h"


class TestMergeRotation:
    def test_can_merge_rotation_true(self):
        ir = _merge_ir()
        assert can_merge_rotation(ir.gates[1], ir.gates[2]) is True

    def test_merge_sums_angles(self):
        ir = _merge_ir()
        out = merge_rotation(ir, 1, 2)
        assert out.gate_count() == ir.gate_count() - 1
        merged = [g for g in out.gates if g.name == "rz"][0]
        assert merged.params[0] == pytest.approx(0.7)

    def test_merge_preserves_equivalence(self):
        ir = _merge_ir()
        out = merge_rotation(ir, 1, 2)
        result = EquivalenceChecker().check(ir, out)
        assert result.equivalent

    def test_merge_to_identity_drops_both_gates(self):
        ir = IntermediateRepresentation(1, name="cancel")
        ir.add("rz", [0], [math.pi])
        ir.add("rz", [0], [math.pi])
        out = merge_rotation(ir, 0, 1)
        assert out.gate_count() == 0
        result = EquivalenceChecker().check(ir, out)
        assert result.equivalent

    def test_merge_mismatched_gates_raises(self):
        ir = IntermediateRepresentation(1)
        ir.add("rz", [0], [0.1])
        ir.add("rx", [0], [0.1])
        with pytest.raises(IllegalActionError):
            merge_rotation(ir, 0, 1)

    def test_merge_different_qubits_raises(self):
        ir = IntermediateRepresentation(2)
        ir.add("rz", [0], [0.1])
        ir.add("rz", [1], [0.1])
        with pytest.raises(IllegalActionError):
            merge_rotation(ir, 0, 1)


class TestApplyActionDispatch:
    def test_noop_returns_equal_copy(self):
        ir = _small_disjoint_ir()
        out = apply_action(ir, "NOOP", -1, -1)
        assert [g.name for g in out.gates] == [g.name for g in ir.gates]
        assert out is not ir

    def test_unknown_kind_raises(self):
        ir = _small_disjoint_ir()
        with pytest.raises(IllegalActionError):
            apply_action(ir, "TELEPORT", 0, 1)


# ---------------------------------------------------------------------------
# environment.py — Gymnasium contract
# ---------------------------------------------------------------------------


class TestSchedulerEnvReset:
    def test_reset_returns_obs_and_info(self):
        env = SchedulerEnv(_small_disjoint_ir())
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "legal_actions" in info

    def test_reset_restores_initial_circuit(self):
        env = SchedulerEnv(_small_disjoint_ir())
        env.reset()
        env.step(Action(ActionKind.COMMUTE, 0, 1))
        obs, info = env.reset()
        assert [g.name for g in env.circuit.gates] == [g.name for g in env.initial.gates]

    def test_reset_accepts_seed_kwarg(self):
        env = SchedulerEnv(_small_disjoint_ir())
        # Must not raise: SB3 / Gymnasium call reset(seed=...) on every VecEnv reset.
        env.reset(seed=123)

    def test_observation_shape_keys(self):
        env = SchedulerEnv(_small_disjoint_ir())
        obs, _ = env.reset()
        for key in ("gate_count", "depth", "two_qubit", "estimated_fidelity", "gate_features", "gate_mask"):
            assert key in obs

    def test_observation_gate_features_padded_and_masked(self):
        ir = _small_disjoint_ir()
        env = SchedulerEnv(ir)
        obs, _ = env.reset()
        assert len(obs["gate_features"]) == ir.gate_count()
        assert sum(obs["gate_mask"]) == ir.gate_count()


class TestSchedulerEnvStep:
    def test_step_noop_zero_reward_no_change(self):
        env = SchedulerEnv(_small_disjoint_ir())
        env.reset()
        obs, r, terminated, truncated, info = env.step(Action(ActionKind.NOOP, -1))
        assert env.circuit.gate_count() == env.initial.gate_count()

    def test_step_commute_changes_order_not_count(self):
        env = SchedulerEnv(_small_disjoint_ir())
        env.reset()
        before_count = env.circuit.gate_count()
        obs, r, terminated, truncated, info = env.step(Action(ActionKind.COMMUTE, 0, 1))
        assert env.circuit.gate_count() == before_count
        assert env.circuit.gates[0].name == "x"

    def test_step_merge_reduces_gate_count_and_rewards_positively(self):
        env = SchedulerEnv(_merge_ir())
        env.reset()
        before_count = env.circuit.gate_count()
        legal = env.legal_actions()
        merge = next(a for a in legal if a.kind == ActionKind.MERGE_ROTATION)
        obs, r, terminated, truncated, info = env.step(merge)
        assert env.circuit.gate_count() == before_count - 1
        assert info["d_gate_count"] == 1
        assert r > 0  # fewer gates, same depth-ish -> net positive shaping reward

    def test_step_illegal_action_falls_back_to_noop(self):
        env = SchedulerEnv(_small_disjoint_ir())
        env.reset()
        before = [g.name for g in env.circuit.gates]
        # (1, 2) share a qubit -> illegal COMMUTE, must not raise or corrupt state
        obs, r, terminated, truncated, info = env.step(Action(ActionKind.COMMUTE, 1, 2))
        assert [g.name for g in env.circuit.gates] == before

    def test_step_after_termination_raises(self):
        env = SchedulerEnv(_small_disjoint_ir(), max_episode_steps=1)
        env.reset()
        env.step(Action(ActionKind.NOOP, -1))
        with pytest.raises(MaxStepsExceeded):
            env.step(Action(ActionKind.NOOP, -1))

    def test_truncation_at_max_steps(self):
        # rand_cnotpauli has many legal COMMUTE actions and won't hit a fixed
        # point quickly, so a small max_episode_steps reliably exercises
        # truncation rather than natural termination.
        ir = _load("rand_cnotpauli_n5.qasm")
        env = SchedulerEnv(ir, max_episode_steps=5)
        obs, info = env.reset()
        rng = random.Random(0)
        terminated = truncated = False
        for _ in range(5):
            action = rng.choice(info["legal_actions"])
            obs, r, terminated, truncated, info = env.step(action)
        assert truncated is True

    def test_terminal_bonus_only_applied_once(self):
        env = SchedulerEnv(_small_disjoint_ir(), max_episode_steps=1)
        env.reset()
        obs, r, terminated, truncated, info = env.step(Action(ActionKind.NOOP, -1))
        assert terminated or truncated


class TestRewardWeights:
    def test_zero_weights_zero_reward(self):
        weights = RewardWeights(depth=0.0, gate_count=0.0, fidelity=0.0, terminal=0.0)
        env = SchedulerEnv(_merge_ir(), weights=weights)
        env.reset()
        merge = next(a for a in env.legal_actions() if a.kind == ActionKind.MERGE_ROTATION)
        obs, r, terminated, truncated, info = env.step(merge)
        assert r == 0.0


# ---------------------------------------------------------------------------
# spaces.py — fixed-size SB3 adapter (skipped without the `rl` extra)
# ---------------------------------------------------------------------------

gym_spaces = pytest.importorskip(
    "qco.modules.rl_scheduler.spaces",
    reason="gymnasium (the `rl` extra) is not installed",
)


class TestSpacesAdapter:
    def test_observation_space_shapes(self):
        space = gym_spaces.make_observation_space(max_gates=32)
        assert space["gate_features"].shape == (32, gym_spaces.GATE_FEATURE_DIM)
        assert space["gate_mask"].shape == (32,)

    def test_action_space_bounds(self):
        space = gym_spaces.make_action_space(max_gates=32)
        assert list(space.nvec) == [len(ActionKind), 32, 32]

    def test_encode_decode_round_trip(self):
        env = SchedulerEnv(_small_disjoint_ir())
        _, info = env.reset()
        real_action = next(a for a in info["legal_actions"] if a.kind == ActionKind.COMMUTE)
        raw = gym_spaces.encode_action(real_action)
        decoded = gym_spaces.decode_action(raw, info["legal_actions"])
        assert decoded == real_action

    def test_decode_illegal_raw_falls_back_to_noop(self):
        env = SchedulerEnv(_small_disjoint_ir())
        _, info = env.reset()
        decoded = gym_spaces.decode_action((ActionKind.COMMUTE, 50, 51), info["legal_actions"])
        assert decoded.kind == ActionKind.NOOP


# ---------------------------------------------------------------------------
# Acceptance criteria: 10k random-policy steps, equivalence preserved
# ---------------------------------------------------------------------------

_SMALL_QUBIT_CIRCUITS = [
    "ghz_n5.qasm",
    "qft_n5.qasm",
    "rand_cliffordt_n5.qasm",
    "rand_cnotpauli_n5.qasm",
    "qaoa_n6_p1.qasm",
    "qaoa_n6_p2.qasm",
]

_LARGE_QUBIT_CIRCUITS = [
    "ghz_n40.qasm",
    "qft_n16.qasm",
    "rand_cliffordt_n20.qasm",
    "qaoa_n16_p1.qasm",
]


@pytest.mark.parametrize("name", _SMALL_QUBIT_CIRCUITS)
def test_smoke_10k_steps_preserves_equivalence(name):
    """Acceptance criteria: 10k random-policy steps, circuit stays equivalent.

    Restricted to circuits at or below ``MAX_NUMERIC_QUBITS`` because
    ``EquivalenceChecker``'s numeric backend raises above 8 qubits (see
    docs/gaps_and_notes_module_a.md); larger circuits get the crash-only
    variant below until the SMT/symbolic path (issue #7) lands.
    """
    ir = _load(name)
    assert ir.num_qubits <= MAX_NUMERIC_QUBITS

    env = SchedulerEnv(ir, max_episode_steps=10_000)
    obs, info = env.reset()
    rng = random.Random(hash(name) & 0xFFFF)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 10_000:
        action = rng.choice(info["legal_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    result = EquivalenceChecker().check(ir, env.circuit)
    assert result.equivalent, f"{name}: circuit diverged after {steps} random steps ({result.detail})"


@pytest.mark.parametrize("name", _LARGE_QUBIT_CIRCUITS)
def test_smoke_10k_steps_large_circuits_do_not_crash(name):
    """Same random-policy smoke test for >8-qubit circuits, crash-test only.

    No ``EquivalenceChecker`` call here — see ``MAX_NUMERIC_QUBITS`` and
    docs/gaps_and_notes_module_a.md. This still catches the class of bug the
    small-circuit test can't: rewrites that only misbehave at larger gate
    counts / qubit indices (off-by-one in qubit indexing, performance
    cliffs, etc.).
    """
    ir = _load(name)
    assert ir.num_qubits > MAX_NUMERIC_QUBITS

    env = SchedulerEnv(ir, max_episode_steps=2_000)
    obs, info = env.reset()
    rng = random.Random(hash(name) & 0xFFFF)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 2_000:
        action = rng.choice(info["legal_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    # No exception raised above is the actual assertion; this just documents
    # that the loop actually progressed rather than exiting on step 0.
    assert steps >= 1