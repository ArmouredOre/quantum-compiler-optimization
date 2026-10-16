# Module A: RL Adaptive Scheduler, Report Section

Final draft for #41, handed to M5 for assembly (`docs/final_report.md` §5.2). Every number is copied from a committed result file: `docs/rl_training/generalization.md`, `docs/rl_training/large/`, `docs/rl_training/ablations/ablations.md`, `docs/rl_training/closed_loop/closed_loop.md`, `docs/rl_training/backend_benchmarks/backend_benchmarks.md`, `docs/module_a_complexity_and_scaling.md`, `docs/partitioner_tuning.md` and `docs/parallel_blocks.md`. The figures are drawn from those files by `scripts/plot_module_a_figures.py`.

## 1. Method

Module A reorders and merges gates so that later stages see a shallower circuit. It is a learned policy wrapped around a deterministic reduction pass.

- **Environment.** `SchedulerEnv` holds a circuit and offers three kinds of move: `NOOP`; `COMMUTE(i, j)`, which flips a dependency edge between two gates that provably commute (the reorder is realised by hoisting or sinking one gate past commuting gates, so the unitary is unchanged by construction); and `MERGE_ROTATION(i, i+1)` for back-to-back rotations of the same axis. Commutation uses a wire-type rule (two gates commute if, on every shared qubit, both have the same Z, X or Y wire type) plus the shared Module B table; both answer `True` only when the unitaries commute exactly.
- **CR pass.** Before each episode and after every step, a deterministic Commutation-and-Reduction pass removes identities, cancels inverse pairs and merges rotations, sliding a gate only past gates it commutes with. It is sound, monotone, terminating, idempotent and deterministic (all tested).
- **Policy.** Each legal move is described by a 28-dimensional feature vector (move kind, gate classes, qubit overlap, ASAP layer and remaining path of both gates, critical-path flags, position, and the one-step change in depth if the move were applied). A one-hidden-layer actor (16 tanh units) scores every move and a 7-feature critic estimates the state value: 626 parameters in total, independent of circuit size, so one policy serves any number of gates or qubits.
- **Training.** PPO (clip 0.2, GAE 0.95, discount 0.99) on generated 5-qubit circuits from four families (Clifford+T, CNOT+Pauli, QAOA, QFT; 12 layers), disjoint from `benchmarks/`. A pure-Python implementation runs in CI; stable-baselines3 `MaskablePPO` is an optional backend.
- **Reward.** Per step: 1.0 × change in depth + 1.0 × change in gate count + 0.5 × change in estimated fidelity, all against the previous step. At the end of an episode: the Stage 6 `scalarized_reward` of the result against the CR-reduced start, so the agent is credited only for what it adds beyond CR.
- **Inference.** The greedy policy is rolled out for at most 40 steps and the best circuit seen (lowest depth, then gate count) is returned, so the output is never worse than the CR result.
- **Place in the pipeline.** `HybridOptimizer` applies Module A to each block (at most 64 gates and 8 qubits) before Module B and Module C. The trained policy `models/rl_ppo_5q.json` is the default; the untrained greedy stand-in is used only if no policy is available.
- **Closed loop (#35).** `RewardLog` consumes the pipeline's per-block and per-iteration reward events. Training can use the real Stage 6 reward of the downstream pipeline instead of the local proxy (`reward_source="pipeline"`).
- **Per backend (#29).** A time-aware reward and lookahead let one policy be trained per device profile (superconducting-like, trapped-ion-like; optional hardware profiles with topology and SWAP cost).
- **Parallel blocks (#46).** `parallel_blocks` runs the per-block work in forked processes; output equals the serial run when the solver budget is not binding.

## 2. Design choices

- **A deterministic CR pass plus a learned policy.** CR does the textbook cancellations so the agent spends its capacity on reorderings. The ablation in Section 3.3 shows why: all gate-count reduction comes from CR, and the agent adds depth on top.
- **A per-action scoring policy, not a fixed action head.** The same weights work on any circuit size, which is what makes the 5-to-40-qubit generalization study possible without retraining.
- **Reward against the CR-reduced reference.** Otherwise the agent would be paid for what CR already does.
- **Train small, test large.** 5-qubit episodes cost milliseconds (Section 3.7), and zero-shot transfer is tested on 6 to 40 qubits.
- **Soundness first.** Every move is a sequence of swaps of commuting gates or an exact rotation merge, so no check is needed to trust it; the pipeline still verifies each block.
- **Honest comparison.** Methods are compared against the untrained stub, CR alone, a greedy one-step depth baseline with CR, and the no-CR agent, all on the same circuits.

## 3. Results

Training uses generated 5-qubit circuits. "Benchmarks" are the circuits in `benchmarks/circuits` (up to 20 qubits) and `benchmarks/circuits_40q` plus `ghz_n40` (40 qubits). Std is over training seeds only.

### 3.1 Zero-shot generalization (Figure 1, `fig_generalization.png`)

Depth reduction %, mean ± std over 3 seeds (100k-step policies, benchmark circuits):

| Method | 5-6 qubits (n=6) | 8-12 qubits (n=7) | 16-20 qubits (n=6) | 40 qubits (n=6) |
|---|---|---|---|---|
| untrained stub | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| CR only | 4.6 ± 0.0 | 2.4 ± 0.0 | 3.1 ± 0.0 | 1.0 ± 0.0 |
| greedy depth + CR | 11.5 ± 0.0 | 8.0 ± 0.4 | 7.4 ± 0.5 | 7.5 ± 0.4 |
| **PPO + CR** | 12.4 ± 0.7 | 9.2 ± 0.6 | 7.9 ± 0.2 | 9.3 ± 0.5 |
| PPO, no CR | 7.9 ± 1.0 | 7.3 ± 0.8 | 9.2 ± 2.3 | 8.5 ± 0.2 |

Gate-count reduction is 8.7 / 7.2 / 6.5 / 6.7 % for CR only, greedy + CR and PPO + CR in the four buckets, and 0 for the stub and the no-CR agent. The policy trained on 5 qubits beats the greedy baseline in every bucket except 16-20 qubits, where the margin (7.9 vs 7.4) is inside the seed spread and the no-CR agent is nominally higher with a large spread (±2.3). Per circuit (Figure 2): QAOA and random circuits improve; GHZ and QFT get nothing at any size (no depth-reducing reordering was found for them).

### 3.2 Closed loop: real reward vs proxy (Figure 4, `fig_closed_loop.png`)

End-to-end Stage 6 reward of the full pipeline with Module A in the loop (2 seeds, 40k steps):

| Method | held-out 5q (n=40) | 5-6q | 8-12q | 16-20q |
|---|---|---|---|---|
| untrained stub | 0.0455 | 0.0436 | 0.0191 | 0.0198 |
| PPO, proxy reward | 0.0815 | 0.0811 | 0.0397 | 0.0380 |
| PPO, real Stage 6 reward | 0.0821 | 0.0794 | 0.0372 | 0.0391 |

Training on the real reward is **not measurably better** than the proxy (differences are within the seed spread; gate reduction is identical). Both roughly double the stub's reward. The real reward costs almost nothing (about a millisecond per 5-qubit circuit), so it is available for future work, but it is not what produces the gain.

### 3.3 Ablations (Figure 3, `fig_ablations.png`)

2 seeds, 40k steps.

- **CR on/off.** Without CR the gate-count reduction is exactly 0 (with CR: 8.7 / 7.2 / 6.5 / 6.7 %). Depth reduction is 13.1 / 9.7 / 12.2 / 10.6 % with CR against 7.5 / 6.9 / 5.5 / 7.7 % without. A CR-trained agent run with CR switched off lands in between, so the agent does not replace CR.
- **Reward weights.** The gate-count weight is inert: "depth only" and "no fidelity term" give identical policies, because with CR on no agent move changes the gate count. Removing the terminal bonus is the only change with a consistent effect (lowest Stage 6 reward in three of four buckets; 8.0 % vs 12.2 % depth at 16-20 qubits). The other variants are within 1-3 points of the default and inside the 2-seed spread, so no ranking among them is claimed.

### 3.4 40 qubits through the full pipeline

With the default trained agent, Module B rules and Module C verification per block (no SMT re-synthesis, no NSGA-II), all six 40-qubit circuits are verified with 0 undecided blocks. Mean Stage 6 reward is 0.0382 against 0.0130 for the stub. The agent is not better everywhere: on `rand_cnotpauli_n40` it scores 0.0451 against the stub's 0.0508. (`docs/rl_training/large/e2e_40q.md`.)

### 3.5 Per-backend policies on the benchmarks (Figure 5, `fig_backend_policies.png`)

Execution-time reduction %, test backend superconducting / trapped ion, policy trained on superconducting versus trapped ion (3 seeds):

| Qubits | trained on superconducting | trained on trapped ion | greedy time heuristic |
|---|---|---|---|
| 5-6 | 5.60 / 4.15 | 3.45 / 2.68 | 4.14 / 3.34 |
| 8-12 | 5.34 / 4.96 | 2.29 / 1.35 | 4.10 / 3.73 |
| 16-20 | 4.36 / 2.90 | 2.12 / 1.01 | 5.41 / 5.12 |
| 40 | 6.16 / 5.30 | 3.26 / 3.18 | 6.04 / 5.80 |

The superconducting-trained policy is better on **both** backends in every bucket; the trapped-ion policy does not win even on its own backend, and the one-step greedy time heuristic beats both learned policies at 16-20 qubits. Schedules do differ between backends (36-53 % of cases), but the difference is not an improvement. The claim "a per-backend policy beats a shared one" is therefore **not supported** on benchmark circuits, consistent with #29's matrix on generated circuits (trapped-ion own-backend wins in 2 of 7 seeds).

### 3.6 Partitioning and parallel blocks

- Block size for the greedy partitioner is capped by the 8-qubit width limit, so `max_block_gates` above about 32 changes nothing (60 blocks, cut 255 on `rand_cliffordt_n20`). The fallback partitioner cuts 2-3x more wires but allows larger blocks, and the gate reduction grows with block size (0.3 % at 16 gates to 14-15 % at 192). Defaults were kept (`docs/partitioner_tuning.md`).
- Parallel and serial runs give identical circuits when the solver budget is not binding. Without re-synthesis the parallel run is slower (0.55-0.62x; blocks take about 25 ms each). With re-synthesis and a binding 20 s budget it is 2.0-2.3x faster, but partly because each block gets a smaller budget share, so less solver work is done (`docs/parallel_blocks.md`).

### 3.7 Complexity (Figure 6, `fig_scaling.png`)

Whole-circuit rollout is O(T · G²) for T steps and G gates: the one-step lookahead applies each of the A ≤ 4G candidate moves to a copy of the circuit (O(G) each). Measured slopes against G: lookahead features 2.13, CR pass on a raw circuit 1.99, `legal_actions` 0.99, one environment step 1.12. Inside the pipeline the agent runs per block of B gates, giving O(T · B · G), linear in G. At 40 qubits a whole-circuit rollout takes 1-58 s (260 to 1383 gates), the pipeline with the agent per block 1-5 s. Training cost depends on the training circuits (5 qubits), not on test size. The policy has 626 parameters. Details in `docs/module_a_complexity_and_scaling.md`.

### 3.8 Correctness

About 950 random circuits (1 to 5 qubits, every gate in the IR, angles including 0, 2π and multiples of π/4) were run through the CR pass, the agent with and without CR, the stand-in and trained schedulers, random-action environment walks in both commutation modes, and the full pipeline. Every output was checked for unitary equivalence (up to global phase) with the exact numeric checker, and gate count and depth never increased under CR or the trained agent. No failure was found. The fuzz harness is not part of the test suite (it is slow); the committed tests cover the same properties on fixed cases.

## 4. Limitations

1. **Small training and small test sets.** Ablations use 2 seeds x 40k steps; each qubit bucket has 6-7 circuits, of which several (GHZ, QFT) cannot be improved. Differences of 1-3 points are not distinguishable from noise.
2. **The agent adds modestly over a greedy baseline.** Most of the gain over the stub is CR; PPO adds about 0.5-2 depth points over greedy + CR, and at 16-20 qubits the no-CR agent is nominally ahead.
3. **Reordering cannot expose new cancellations.** A gate that does not commute with a candidate stays between the pair in every valid order, so the agent changes depth, not gate count.
4. **Whole circuit vs per block.** Blocks (at most 8 qubits, 64 gates) fragment wide circuits and limit what the agent can see; on `rand_cliffordt_n20` the whole-circuit agent reaches 515 gates against 579 per block (`final_report.md` §8).
5. **Per-backend specialization is not demonstrated**, and the calibration profiles are illustrative, not live device data.
6. **No equivalence check of the agent alone beyond 8 qubits.** The numeric checker is capped at 8 qubits; the pipeline's block-wise verification covers wider circuits.
7. **Time.** A whole-circuit rollout is quadratic in the gate count (seconds to a minute at 40 qubits), so the agent should run per block.
8. **Policy location.** `models/rl_ppo_5q.json` is found relative to the repository; a non-editable install silently falls back to the stand-in unless `rl_policy_path` is given.

## 5. Figures

Generated by `python scripts/plot_module_a_figures.py` into `docs/rl_training/figures/` (PNG at 200 dpi and SVG):

- `fig_generalization`: depth and gate-count reduction by qubit bucket, 5 to 40 qubits (Figure 1; the main generalization plot).
- `fig_generalization_per_circuit`: per-circuit depth reduction, PPO + CR against the greedy baseline.
- `fig_ablations`: CR on/off and the reward-weight sweep.
- `fig_closed_loop`: proxy against real reward, with held-out depth during training.
- `fig_backend_policies`: per-backend policies on the benchmarks.
- `fig_scaling`: measured time against circuit size with slope-1 and slope-2 guides.

Error bars are seed standard deviations only.

## 6. Hand-off to assembly (M5)

| Report part | Take from |
|---|---|
| §5.2 Module A (method, design choices) | Sections 1 and 2 above (can be pasted as is) |
| §7.3 Ablations (Module A rows) | Section 3.3 and `fig_ablations` |
| Generalization figure and table | Section 3.1, `fig_generalization` (and `fig_generalization_per_circuit`) |
| §7.7 Runtime complexity (Module A) | Section 3.7, `fig_scaling`, full derivation in `module_a_complexity_and_scaling.md` |
| §8 Discussion and limitations (Module A) | Section 4 |
| §7.2 qco vs baselines | the `make eval` run; it now loads the same default agent as the pipeline (fixed in #41) |

Open dependency: the final `make eval` (Swaraj, #36/#40) produces the end-to-end qco numbers; the Module A figures here do not depend on it.
