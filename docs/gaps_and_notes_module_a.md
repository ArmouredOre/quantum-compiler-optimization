# Module A (RL Scheduler) — Sprint 1 gaps, decisions, and handoff notes

This file records the design decisions made while implementing
`qco.modules.rl_scheduler.environment.SchedulerEnv.reset` / `.step`, the gaps
found in adjacent modules while doing so, and what Sprint 2 should pick up.
Nothing here changes any other contributor's frozen interfaces.

## 1. Gap: numeric `EquivalenceChecker` caps at 8 qubits

`qco/modules/smt_verifier/equivalence.py` raises `ValueError` above 8 qubits
in its numeric (dense unitary) backend:

```python
if n > 8:
    raise ValueError("numeric unitary limited to 8 qubits; use the SMT path")
```

but `benchmarks/circuits/` includes circuits up to 40 qubits (`ghz_n40`,
`rand_cliffordt_n20`, `qft_n16`, `qaoa_n16_p*`, etc.), and issue #5's
acceptance criteria explicitly asks for a smoke test that checks equivalence
"on the benchmark suite" without qualifying qubit count.

**This is not new** — `qco/evaluation/benchmark_runner.py` already gates its
own equivalence calls on qubit count for the same reason so it's a known,
already-worked-around limitation of Phase 2, not something introduced here.

**What was done:** the smoke test
(`tests/test_rl_scheduler.py`) is split in two:

* `test_smoke_10k_steps_preserves_equivalence` — runs the full 10k-step
  random-policy loop **and** asserts `EquivalenceChecker` reports equivalence,
  restricted to the six benchmark circuits at or below 8 qubits
  (`ghz_n5`, `qft_n5`, `rand_cliffordt_n5`, `rand_cnotpauli_n5`,
  `qaoa_n6_p1`, `qaoa_n6_p2`).
* `test_smoke_10k_steps_large_circuits_do_not_crash` — runs the same
  random-policy loop on four larger circuits (`ghz_n40`, `qft_n16`,
  `rand_cliffordt_n20`, `qaoa_n16_p1`) for correctness/crash-testing only, with
  **no** equivalence assertion.

**Sprint 2 follow-up:** issue #7  is described in
`docs/sprint_plan.md` as adding the symbolic/SMT path that removes this qubit
cap. Once that lands, the two test functions above should be merged back into
one parametrized test covering the whole suite which was kept
`MAX_NUMERIC_QUBITS` as a single named constant in the test file specifically
so that merge is a one-line change (delete the split, use one list, drop the
constant) rather than a rewrite.

## 2. Design decision: action space

The frozen `Action(kind, i, j)` / `legal_actions()` contract from Phase 2 is
unchanged — `SchedulerEnv` itself still speaks in variable-length legal-action
lists, which is the easiest representation to keep provably safe (every
`Action` the env receives from `legal_actions()` is legal by construction).

A **separate, optional adapter** (`qco/modules/rl_scheduler/spaces.py`) adds:

* `make_observation_space(max_gates)` — a fixed-shape Gymnasium `Dict` space
  (`gate_features`, `gate_mask`, `globals`) suitable for an SB3 policy.
* `make_action_space(max_gates)` — a fixed `MultiDiscrete([5, max_gates,
  max_gates])` space.
* `decode_action(raw, legal_actions)` / `encode_action(action)` — translate
  between the two representations. An illegal or nonsensical raw action
  decodes to `NOOP` rather than raising, so an untrained policy network can't
  crash an episode by sampling garbage early in training.

This file imports `gymnasium`/`numpy` lazily and degrades gracefully (raises a
clear `ImportError` naming the `rl` extra) if they aren't installed, so
`environment.py` and `rewrites.py` have **no hard dependency** on the `rl`
extra — only code that actually wants the SB3-shaped spaces needs
`pip install -e '.[rl]'`.

**Sprint 2 follow-up (flagged, not built yet):** `decode_action`'s current
"illegal -> NOOP" fallback wastes training signal — a policy that samples an
illegal action gets a `0` reward step instead of useful gradient. The standard
fix is action masking (`sb3-contrib`'s `MaskablePPO` + an `action_masks()`
method built from `legal_actions()`). I've deliberately left this out of
Sprint 1 scope since it depends on which SB3/sb3-contrib version Sprint 2
standardizes on, and touches training-script code that doesn't exist yet.

## 3. Design decision: reward shaping

The issue's frozen docstring specifies:

```
Reward  : w_d * dDepth + w_g * dGateCount + w_f * dFidelity  (deltas vs. previous
          step); terminal bonus from the Stage 6 evaluation engine (closed loop).
```

Implemented literally:

* Every `step` computes `d_gate_count`, `d_depth`, `d_fidelity` as
  **this-step-vs-previous-step** deltas (not vs. the episode's initial
  circuit), matching "deltas vs. previous step" precisely, and combines them
  via the existing `RewardWeights` (`depth`, `gate_count`, `fidelity`).
* `estimated_fidelity` is `qco.evaluation.metrics.estimated_fidelity` — the
  same Stage 6 fidelity proxy the rest of the pipeline reports against, not a
  reimplementation.
* On the step that ends an episode (`terminated` or `truncated`), the reward
  additionally adds `weights.terminal * scalarized_reward(initial, final)` —
  `scalarized_reward` is exactly the function
  `qco/evaluation/metrics.py` already documents as *"Single scalar fed back to
  Module A as the RL reward (closed loop)"*, so this wires up a hook that was
  already sitting there waiting for Module A, rather than inventing a new one.
  added a `terminal: float = 1.0` field to `RewardWeights` to make this
  bonus's weight independently tunable (and zero-able, for unit tests that
  only want to check per-step shaping) without touching the other three
  weights' meaning.

**Something to revisit:** the three per-step weights and the new
`terminal` weight are on different natural scales (`d_gate_count`/`d_depth`
are small integers; `scalarized_reward` is roughly in `[-1, 1]`). not
attempted to auto-normalize these, that's a training-stability tuning
question best answered empirically once Sprint 2 actually trains something,
not guessed at now.

## 4. Design decision: episode termination

* `terminated=True` when `legal_actions()` reduces to `[NOOP]` only — a
  genuine fixed point, nothing left to try.
* `truncated=True` at `max_episode_steps` (default 200; the smoke test
  overrides this to 10,000 / 2,000 as needed) <- a training-budget cutoff.

This is the standard Gymnasium `terminated` vs. `truncated` distinction (the
post-0.26 replacement for a single `done` flag), used because SB3's `VecEnv`
machinery expects it and because it keeps "ran out of budget" separable from
"provably nothing left to optimize" in logged episode data.

## 5. What was not touched

* `agent.py` (the actual RL policy / training loop) — that's `agent.py`'s
  `# Phase 3` stub, listed as a Sprint 2 deliverable in `docs/sprint_plan.md`,
  not part of issue #5.
* Any file outside `qco.modules.rl_scheduler` and its tests — no changes to
  `qco/ir`, `qco/graphs`, `qco/evaluation`, or any other contributor's module.
* `RewardWeights`'s three original fields keep their original defaults
  (`depth=1.0, gate_count=1.0, fidelity=0.5`) — only the new `terminal` field
  was added, as an appended dataclass field with a default, so no existing
  call site that constructs `RewardWeights()` (there are none yet, since this
  was a stub) is affected either way.

## 6. Files added / changed

| File | Status | Purpose |
|---|---|---|
| `src/qco/modules/rl_scheduler/environment.py` | modified | `reset`/`step` implemented; observation extended with `gate_features`/`gate_mask`/`estimated_fidelity` |
| `src/qco/modules/rl_scheduler/rewrites.py` | **new** | Pure IR-rewrite functions (commute/move/merge-rotation), independently unit-tested |
| `src/qco/modules/rl_scheduler/spaces.py` | **new** | Optional Gymnasium space + fixed-size SB3 action adapter |
| `tests/test_rl_scheduler.py` | **new** | 46 tests: rewrite unit tests, env contract tests, spaces adapter tests, 10k-step smoke tests |
| `docs/gaps_and_notes_module_a.md` | **new** | this file |