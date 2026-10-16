# Module A: Complexity and Scaling

Time and memory costs of the RL scheduler (`src/qco/modules/rl_scheduler/`), derived from the code and then checked against measurements (section 9). Mirrors `docs/module_b_complexity_and_scaling.md` and `docs/module_c_complexity_and_scaling.md`.

## Notation

G gates, Q qubits, D circuit depth, E dependency-DAG edges, A legal actions offered in a state, T episode steps (`max_episode_steps`, 40 by default), W the CR scan window (`cr_window`, `None` = unbounded), R rewrites performed by the CR pass (R ≤ G, each removes at least one gate), h hidden width (16), d = `CANDIDATE_DIM` = 28, g = `GLOBAL_DIM` = 7, B gates per block in the pipeline (`max_block_gates`, 64), K = `n_steps` per PPO iteration (1024), C = `max_candidates` (48, training only). Gate arity is at most 3, so every per-gate set or table operation (qubit overlap, `commutes`, `reduce_pair`, `same_operands`) is O(1).

## 1. Circuit primitives

`IR.depth()`, `execution_time`, `estimated_fidelity`, `scalarized_reward`, `build_dag`, `asap_and_tail`, `dag_edges` and `observation()` are each a single pass over the gates: O(G + Q). `IR.copy()` is O(G). E ≤ 3G (each gate adds at most one edge per qubit it touches), so A ≤ E + G ≤ 4G.

## 2. CR pass (`cr_pass.reduce_with_stats`)

1. Identity removal: one pass, O(G).
2. `_find_partner(i)` scans forward from gate i, skipping disjoint gates, until it finds a partner or the first non-commuting gate on a shared qubit. At most min(W, G − i) gates are examined, O(1) each, so O(min(W, G)) per call.
3. One sweep calls `_find_partner` for every i: O(G · min(W, G)) worst case. After each rewrite the pass restarts at i = 0 (removing a gate can unblock an earlier scan), and each rewrite also costs O(G) for the `del` on a Python list.

Worst case O((R + 1) · G · min(W, G)) = O(G³) with `cr_window=None`; O(R · G · W) with a window. The worst case needs long runs of mutually commuting gates (diagonal-heavy circuits such as QAOA/QFT); the scan stops at the first non-commuting gate on a shared qubit, so on typical circuits the mean scan length w̄ is small and the pass is about O((R + 1) · G · w̄).

Memory: O(G) for the gate list.

## 3. Action enumeration (`SchedulerEnv.legal_actions`)

- `commutation="disjoint"`: one pass over adjacent pairs, O(G).
- `commutation="full"` (the default config): `dag_edges` is O(G). `flippable_edges` runs `flip_strategy` for each edge (i, j), which scans the gates strictly between i and j. For a fixed qubit wire the gaps between consecutive gates sum to at most G, and every edge lies on some wire, so the total scan length is at most Q · G. Hence O(Q · G) worst case, O(G) when gaps are short (the usual case).
- `MERGE_ROTATION` candidates: one pass, O(G).

Output size A ≤ 4G (usually far fewer; most dependency edges are not flippable).

## 4. Features and the one-step lookahead (`features.encode_state`)

`asap_and_tail` and the global features are O(G + Q). Each candidate row is O(1) plus the lookahead, which applies the move (`apply_action`: `dag_edges` + `flip_strategy` + list slicing + `copy`, all O(G)) and recomputes depth (O(G)). So the lookahead costs O(G) per action and **O(A · G) per step = O(G²) in the worst case**; with training's candidate cap it is O(C · G). With `lookahead=False` the rows cost O(A) after the O(G + Q) preprocessing. With `lookahead_metric="time"` the same O(G) per action applies (`execution_time`).

This lookahead dominates inference time on large circuits (section 9).

## 5. Policy (`ppo.PPOPolicy`)

Actor: one hidden layer, `tanh(W1 x + b1)` then a dot product. Scoring A rows costs O(A · d · h) ≈ 450 · A multiply-adds at d = 28, h = 16; the critic evaluates one g-dimensional row. Parameter count is fixed: actor 28·16 + 16 + 16 + 1 = 481, critic 7·16 + 16 + 16 + 1 = 145, so 626 floats (14 KB as JSON) **independent of G and Q**. This is why a policy trained on 5 qubits can be run unchanged on 40.

Greedy selection is an O(A) argmax; sampling is O(A).

## 6. One environment step, one episode, `schedule()`

`SchedulerEnv.step` does: `apply_action` O(G); CR pass (section 2); `build_dag` O(G); metrics O(G); then `legal_actions()` **twice** (once for `info`, once in `_is_fixed_point`), each O(Q · G); `observation()` O(G).

- One step: O(Q · G + CR) plus, for the policy, O(A · G) from the lookahead.
- One episode (`run_policy`, greedy, no candidate cap): **O(T · (A · G + CR)) = O(T · G²)**. T is bounded by `max_episode_steps` and the episode ends early at a fixed point or on NOOP.
- `RLScheduler.schedule` = one `run_policy` call. The untrained stand-in is O(Q · G²) (at most Q + 1 sweeps, each trial swap recomputes `depth()`).

Memory per episode: O(G) per circuit copy and O(A) candidate rows of d floats, i.e. O(G) overall.

### In the pipeline

`HybridOptimizer` calls `schedule` per **block**, not on the whole circuit. With B-gate blocks, Module A costs about (G / B) · T · B² = **O(T · B · G)**, linear in G for fixed B and T. Running the agent on the whole circuit (as the 40-qubit generalization study does) is the O(T · G²) case. Blocks are independent, so `parallel_blocks` can spread them over processes (`docs/parallel_blocks.md`); that pays only when other per-block work (resynthesis) dominates.

## 7. Training (`ppo.PPOTrainer`, `training.train`)

Per iteration:

- **Collection:** K environment steps on circuits with G_train gates, each O(Q · G_train + C · G_train) (candidate cap C applies to scoring and lookahead, not to `legal_actions`).
- **Update:** `epochs` passes over K transitions in minibatches; each transition re-scores at most C candidate rows: O(epochs · K · C · d · h) for the forward pass and the same order for the backward pass.
- Total iterations: `total_timesteps / K`.

Training cost therefore depends on the **training** circuit size (5 qubits, 12 layers, a few tens of gates), not on the size of circuits the policy is later used on. Peak memory is dominated by stored transitions: K · C · d floats ≈ 1024 · 48 · 28 ≈ 1.4 M floats, tens of MB as Python lists.

`reward_source="pipeline"` (#35) adds one pipeline run per episode end (two with the reference, cached by an LRU of 8192): about a millisecond per 5-qubit circuit, so training time barely changes (4.9 s vs 4.2 s smoke run).

### Per-backend policies (#29)

One independent training run per device profile: cost × number of profiles. Inference cost is unchanged. Model memory is 626 floats per policy, so P backends cost 626 · P floats.

## 8. Summary

| Quantity | Time | Memory |
|---|---|---|
| depth / metrics / DAG | O(G + Q) | O(G) |
| CR pass | O((R+1) · G · min(W, G)); typically ≈ O((R+1) · G · w̄) | O(G) |
| `legal_actions` (full) | O(Q · G) worst, O(G) typical | O(A) |
| `encode_state` with lookahead | O(A · G) = O(G²) | O(A · d) |
| `encode_state` without lookahead | O(G + Q + A) | O(A · d) |
| policy scoring | O(A · d · h) | 626 parameters |
| one step | O(Q · G + CR + A · G) | O(G) |
| one episode / `schedule` (whole circuit) | O(T · G²) | O(G) |
| `schedule` inside the pipeline (blocks of B) | O(T · B · G) | O(B) per block |
| PPO iteration | O(K · (Q·G_train + C·G_train) + epochs · K · C · d · h) | O(K · C · d) |

## 9. Measured scaling

### Gate-count sweep (Q = 20)

| layers | G | legal actions | depth (ms) | cr_pass (ms) | legal_actions (ms) | encode_state (ms) | env_step (ms) | episode (ms) |
|---|---|---|---|---|---|---|---|---|
| L=10 | 185 | 55 | 0.0 | 9.0 | 0.9 | 7.6 | 2.6 | 468.4 |
| L=20 | 370 | 113 | 0.1 | 29.2 | 2.5 | 30.6 | 4.4 | 1584.3 |
| L=40 | 714 | 228 | 0.2 | 109.0 | 3.6 | 103.8 | 14.4 | - |
| L=80 | 1400 | 481 | 0.4 | 507.7 | 7.4 | 525.6 | 24.0 | - |
| L=160 | 2761 | 946 | 0.7 | 1766.0 | 14.9 | 2369.5 | 49.7 | - |

Fitted log-log slope of time vs G (1 = linear, 2 = quadratic): depth 1.06, cr_pass 1.99, legal_actions 0.99, encode_state 2.13, env_step 1.12, episode 1.76

### Qubit sweep (20 layers)

| qubits | G | legal actions | depth (ms) | cr_pass (ms) | legal_actions (ms) | encode_state (ms) | env_step (ms) | episode (ms) |
|---|---|---|---|---|---|---|---|---|
| Q=5 | 82 | 26 | 0.0 | 1.5 | 0.3 | 1.7 | 1.0 | 122.2 |
| Q=10 | 182 | 71 | 0.0 | 7.0 | 1.0 | 11.3 | 3.1 | 568.9 |
| Q=20 | 370 | 113 | 0.1 | 32.3 | 2.0 | 41.1 | 6.1 | 1577.0 |
| Q=40 | 702 | 239 | 0.2 | 123.9 | 4.3 | 120.8 | 13.7 | - |

Reproduce: `python benchmarks/bench_module_a_scaling.py` (random Clifford+T circuits, best of 3, trained default policy, 2-core shared sandbox, so read the absolute times as indicative). The "episode" column stops at 700 gates because a whole-circuit rollout becomes minutes.

What the numbers say:

- **Predicted and measured agree on the quadratic terms.** `encode_state` (the lookahead) has slope 2.13 and the standalone CR pass 1.99 against the predicted 2. `legal_actions` (0.99), `depth` (1.06) and one `env_step` (1.12) are about linear.
- **The CR cubic worst case does not show up** on these circuits (scan length stops at the first non-commuting gate). The CR pass is expensive when run on a raw circuit with many rewrites (1.8 s at 2761 gates) and cheap inside an episode afterwards, because `env_step` re-runs it on an already reduced circuit (one rewrite-free sweep).
- **Whole-circuit episodes are the problem at 40 qubits.** Measured greedy `agent_cr` rollouts (3 seeds, `docs/rl_training/large/`): qaoa_n40_p1 (260 gates) 1.1 s, qaoa_n40_p2 (480) 3.8 s, rand_cnotpauli_n40 (1264) 19 s, rand_cliffordt_n40 (1383) 26 s, qft_n40 (840) 58 s. QFT is slowest because its many commuting diagonal gates give the most legal actions A.
- **Per-block use is much cheaper.** The full `HybridOptimizer` (agent per 64-gate block, plus verification) takes 1, 1, 4, 5 and 2 s on the same five circuits (`docs/rl_training/large/e2e_40q.md`), consistent with O(T · B · G) versus O(T · G²).
- **A bigger episode budget costs linearly and buys nothing here.** `agent_cr_x` (8x the steps at 40 qubits) took 8x as long (e.g. 189 s vs 26 s on rand_cliffordt_n40) and gave identical circuits, so the greedy policy has converged within the default 40 steps.


## 10. Bottlenecks and cheap improvements (not implemented)

1. **Lookahead dominates inference.** Rebuilding the whole circuit per candidate is O(G); computing the depth change of a flip from the cached ASAP layers and tails would make it O(gap) per action.
2. **`legal_actions()` is called twice per step** (`info` and `_is_fixed_point`); caching the list in `step` halves that cost.
3. **CR restarts at i = 0 after every rewrite.** Resuming a little before the rewrite point, or using the per-qubit last-gate index instead of a forward scan, would remove the R · G factor.
4. **`_flip` rebuilds `dag_edges` to validate** an action that `legal_actions` already produced; validating from the offered edge set would save O(G) per applied move.
5. **Large circuits:** run the agent per block (the pipeline already does) or with `cr_window` set; whole-circuit rollouts are quadratic.
