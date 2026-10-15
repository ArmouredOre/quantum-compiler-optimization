# Sprint 3 / N4 — Module A report (issues #26 and #29)

## Verdict
| Issue | Status |
|---|---|
| **#26** train on 5q, zero-shot 10/20q, with/without CR, loadable checkpoint | **Done.** Both acceptance criteria met (below). Real `Block`s (#45) not needed; whole circuits are used as the issue allows. |
| **#29** `train`/`load`/`schedule()` with the trained policy | **Done** (builtin since #10; SB3 `MaskablePPO` model now also works inside `RLScheduler`: train, save/load `.zip`, `schedule`). |
| **#29** per-backend policies + cross-backend matrix | **Built and measured; acceptance NOT met.** Matrix committed, but the policies do not reliably win on their own backend (details below). |

## #26 — generalization (`docs/rl_training/generalization.md`)
Trained on generated 5-qubit circuits (disjoint from `benchmarks/`), 100k steps, 3 seeds; evaluated zero-shot on the benchmark circuits (ghz, qft, qaoa, random Clifford+T, random CNOT+Pauli; up to 20 qubits). Mean depth reduction vs the raw circuit:

| Method | 5-6 q | 8-12 q | 16-20 q |
|---|---|---|---|
| greedy `schedule()` stub | 0.0 | 0.0 | 0.0 |
| CR only | 4.6 | 2.4 | 3.1 |
| greedy-depth + CR | 11.5 | 8.0 | 7.4 |
| **PPO + CR** | **12.4 ± 0.7** | **9.2 ± 0.6** | **7.9 ± 0.2** |
| PPO, no CR (trained & run) | 7.9 | 7.3 | 9.2 |
| CR-trained PPO, CR off at test | 7.2 | 7.0 | 4.9 |

* Criterion 1 (beats the greedy stub on 5q depth): **met**, but note the stub only swaps adjacent *disjoint* gates, which can never change depth, so it is a 0 % baseline. The fairer baseline is greedy-depth + CR: PPO leads by ~1 point at 5-12q and ~0.5 at 16-20q (within about one std over seeds at 16-20q).
* Criterion 2 (5 → 10 → 20 table committed): **met** (`generalization.md/json`, per-circuit rows included).
* Gate-count reduction comes entirely from CR (8.7 / 7.2 / 6.5 %). Without CR the agent cuts 0 gates.
* "No CR" wins on depth at 16-20q (9.2 vs 7.9), driven by `qaoa_n16_*`; with CR the policy is credited only for what it adds on top of CR. Only the CR variant reduces gate count, so it is the shipped one.
* qft/ghz circuits get 0 % depth reduction from every method (nothing commutes usefully).
* Raising the step budget to `40·n/5` changes nothing: the best circuit is reached within the first 40 steps.
* Equivalence is checked numerically only for ≤ 8 qubits (147 checks, 0 failures); 10/20-qubit outputs are unverified (SMT follow-up).
* Checkpoint: `models/rl_ppo_5q.json` → `RLScheduler().load("models/rl_ppo_5q.json").schedule(circuit)`.

## #29 — per-backend policies (`docs/rl_training/backend_matrix.md`)
**Finding that changes the plan:** the issue says no environment change is needed because the reward already takes `calibration`. It does, but only through estimated fidelity, which depends on gate *counts*; COMMUTE moves change no counts, so the learned reward (and policy) is backend-independent. I therefore added two opt-in pieces (defaults keep old behaviour): `RewardWeights.time` (per-step drop in calibrated critical-path time, in average-layer units) and `RLConfig.lookahead_metric="time"`.

Result (60 held-out 5q circuits, **7 seeds**, 60k steps; mean ± std of execution-time reduction % under the test backend; `backend_matrix.md`):

| test \ train | ibm_like | trapped_ion_like | greedy time heuristic |
|---|---|---|---|
| ibm_like | **8.01 ± 0.35** | 7.48 ± 0.43 | 7.62 |
| trapped_ion_like | 7.51 ± 0.42 | **7.14 ± 0.36** | 7.33 |

* Schedules differ between the two backends' policies in 21 % of cases: measurably different.
* The ibm-like policy wins on its own backend robustly (6/7 seeds, +0.5 points, and beats the greedy time heuristic). The ion-like policy does **not**: it loses on its own backend (7.14 vs 7.51; own policy better in 2/7 seeds). In fact it is the weaker policy in *both* columns, i.e. a training-quality difference rather than specialization. **Acceptance criterion 2 is therefore not met.**
* Why it is hard: a policy trained on A and run on B gets B's calibrated lookahead as an input, so it adapts to B through its features; the remaining backend-specific behaviour is small. I also tried hiding the lookahead (`--no-lookahead`, 100k steps, 3 seeds, `backend_matrix_nolookahead.md`): policies then only reach ~2-3 % (vs ~7.5-8 %) and still do not specialize (ion-like again loses its own column).
* What would likely change it: profiles that differ in structure (connectivity / SWAP cost, native gate sets) rather than scale, larger circuits, or multi-step credit assignment. Recommendation: relax the criterion to "measurably different schedules + matrix reported", or scope a follow-up with structurally different profiles; I did not tune the result to force a win.

* **Follow-up:** the structural-profile experiment (topology, SWAP cost, native gates) is in `docs/hardware_profiles.md` and sub-issue draft `docs/issues/sub_issue_hardware_profiles.md`; it shows structure does matter (routing-constrained policies beat all-to-all ones by ~2.2 points on the grid) but still no clean per-backend win.

## How to run / test
```bash
make eval-generalization    # ~25 min on 2 cores (6 trainings); writes docs/rl_training/generalization.*
make eval-backends          # ~15 min; writes docs/rl_training/backend_matrix.*
pip install -e ".[test]" && python -m pytest -q tests/test_rl_sprint3.py   # ~1 min, pure Python
pip install -e ".[test,rl]" && python -m pytest -q tests/test_rl_sprint3.py -k SB3
```
New code: `generalization.py`, `backend_matrix.py`, `scripts/eval_generalization.py`, `scripts/eval_backend_matrix.py`; changed: `environment.py` (time term), `features.py` (time lookahead), `config.py`, `ppo.py`, `sb3_backend.py` (`run_sb3_policy`, `load_sb3`), `agent.py` (SB3 save/load/schedule), `Makefile`. Restored from #10 because they were missing on main: `configs/*.yaml`, `docs/sprint2_report_module_a.md`.

Writing tests for this code: build tiny configs (`total_timesteps=100`), assert structure/invariants (matrix keys, bounds, equivalence of outputs, `stub` depth 0), never assert that a trained policy beats a baseline in unit tests; that belongs to the committed study outputs.
