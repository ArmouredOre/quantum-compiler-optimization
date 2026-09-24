"""Module A — Gymnasium space definitions and the fixed-size action adapter.

``SchedulerEnv`` itself is driven by ``(kind, i, j)`` :class:`~qco.modules.rl_scheduler.environment.Action`
objects and a dynamic ``legal_actions()`` list — the natural representation for
the rewrite logic in ``rewrites.py`` and for any non-NN controller (random
baseline, search, human-in-the-loop debugging). Stable-Baselines3 policies,
however, need a **fixed-size** space declared once at construction time. This
module is the thin translation layer between the two, kept separate so:

* ``SchedulerEnv`` has no hard dependency on ``gymnasium`` at import time
  (only ``spaces.py`` does, and only when actually used) — matching the
  "usable without extras installed" pattern used elsewhere in this repo
  (see ``qco.modules.gnn_cancellation.model``'s optional-torch import).
* the SB3 adapter can be revisited/extended in Sprint 2 (e.g. action masking
  via ``sb3-contrib``'s ``MaskablePPO``) without touching ``environment.py``.

Import ``gymnasium`` lazily so ``environment.py`` and ``rewrites.py`` keep
working in a base install; only code that actually builds spaces needs the
``rl`` extra (``pip install -e '.[rl]'``).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - CI installs only ``.[test]``
    np = None  # type: ignore[assignment]
    gym = None  # type: ignore[assignment]
    gym_spaces = None  # type: ignore[assignment]

from qco.modules.rl_scheduler.environment import Action, ActionKind

_RL_EXTRA = "qco.modules.rl_scheduler.spaces requires the `rl` extra: pip install -e '.[rl]'"

# Per-gate feature width produced by ``environment.SchedulerEnv.observation()``'s
# ``gate_features`` array: [gate_type_id (float), qubit_a, qubit_b (-1 if n/a),
# is_two_qubit, is_parametric]. Kept as a module constant so the SB3 wrapper and
# the environment agree on shape without importing each other's internals.
GATE_FEATURE_DIM = 5

# Number of distinct action *kinds* the fixed action space enumerates
# (matches ``ActionKind``); NOOP has no operands so it occupies one slot.
_NUM_KINDS = len(ActionKind)


def _require_gym() -> None:
    if gym_spaces is None:
        raise ImportError(_RL_EXTRA)


@dataclass(frozen=True, slots=True)
class ActionSpaceConfig:
    """Bounds for the fixed ``MultiDiscrete`` action space.

    ``max_gates`` should be set to (at least) the largest circuit's gate count
    the policy will ever see; indices beyond the *current* circuit's length are
    legal to *emit* but decoded to ``NOOP`` by :func:`decode_action` (see
    module docstring on masking below), so an under-sized ``max_gates`` merely
    clips reachable gates rather than erroring.
    """

    max_gates: int = 512


def make_observation_space(max_gates: int = 512):
    """``Dict`` observation space matching ``SchedulerEnv.observation()``.

    ``gate_features``: ``(max_gates, GATE_FEATURE_DIM)`` float32, zero-padded
    past the circuit's current gate count. ``gate_mask``: ``(max_gates,)``
    float32 in {0, 1} marking padding so a policy/value network can ignore it
    (standard padding-mask pattern for variable-length sequences in SB3).
    ``globals``: ``(3,)`` float32 = ``[gate_count, depth, two_qubit_count]``,
    unnormalized (an SB3 ``VecNormalize`` wrapper is the conventional place to
    rescale these, rather than baking a normalization scheme in here that
    would need to match whatever the reward-scale ends up being — left as a
    Sprint 2 training-script concern).
    """
    _require_gym()
    return gym_spaces.Dict(
        {
            "gate_features": gym_spaces.Box(low=-1.0, high=float("inf"), shape=(max_gates, GATE_FEATURE_DIM), dtype=np.float32),
            "gate_mask": gym_spaces.Box(low=0.0, high=1.0, shape=(max_gates,), dtype=np.float32),
            "globals": gym_spaces.Box(low=0.0, high=float("inf"), shape=(3,), dtype=np.float32),
        }
    )


def make_action_space(max_gates: int = 512):
    """Fixed ``MultiDiscrete([_NUM_KINDS, max_gates, max_gates])`` action space.

    Encodes ``Action(kind, i, j)`` as three independent discrete choices so a
    standard SB3 ``MultiDiscrete``-compatible policy (e.g. PPO with a
    ``MultiCategorical`` head) can drive the environment without any
    dynamic-sized output layer. Decoding back to an ``Action`` — including
    rejecting/no-opping combinations that aren't currently legal — is
    :func:`decode_action`'s job, not the space's; the space only declares
    bounds.
    """
    _require_gym()
    return gym_spaces.MultiDiscrete([_NUM_KINDS, max_gates, max_gates])


def decode_action(raw, legal_actions: list[Action]) -> Action:
    """Map a raw ``MultiDiscrete`` sample (or any 3-tuple) to a legal ``Action``.

    Sprint-1 policy (intentionally simple, revisit for Sprint 2's masked
    policy): if the decoded ``(kind, i, j)`` matches one of ``legal_actions``
    exactly, return it; otherwise fall back to ``Action(ActionKind.NOOP, -1)``.
    This keeps ``SchedulerEnv.step`` always safe to call from an untrained /
    randomly-initialized policy network (which will emit mostly-illegal raw
    actions early in training) without the environment needing to raise,
    matching Gymnasium convention of tolerating "wasted" steps rather than
    crashing an episode. A full action-masking wrapper (``sb3-contrib``'s
    ``MaskablePPO`` + ``action_masks()``) is the Sprint-2 follow-up noted in
    ``docs/gaps_and_notes_module_a.md`` to stop wasting training signal on
    illegal picks.
    """
    kind_id, i, j = int(raw[0]), int(raw[1]), int(raw[2])
    try:
        kind = ActionKind(kind_id)
    except ValueError:
        return Action(ActionKind.NOOP, -1)

    for act in legal_actions:
        if act.kind == kind and act.i == i and act.j == j:
            return act
        if kind == ActionKind.NOOP and act.kind == ActionKind.NOOP:
            return act
    return Action(ActionKind.NOOP, -1)


def encode_action(action: Action) -> tuple[int, int, int]:
    """Inverse of the decoding used above: ``Action`` -> raw ``MultiDiscrete`` tuple.

    Mainly useful for tests and for logging what a random/scripted policy
    picked in the same integer encoding SB3 will see.
    """
    return (int(action.kind), max(action.i, 0), max(action.j, 0))