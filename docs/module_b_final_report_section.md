# Module B: GNN Cancellation, Report Section

Draft for #42. Every number comes from `docs/gnn_eval/window_ranking.md`, `docs/gnn_eval/pr_table*.md`, `results_new/sweep/*` (summaries and per-circuit tables), the profile runs under `results/profiles/*/gnn_noleak`, or `docs/module_b_complexity_and_scaling.md`.

## 1. Method

Module B predicts where a circuit can be shortened. It scores two kinds of candidates: overlapping gate pairs (cancellation candidates) and short gate windows (resynthesis candidates).

- **Encoder.** The circuit is a gate DAG. Each gate is a node with a 35-dimensional feature vector (a one-hot gate type plus scalars). Edges are added in both directions with self-loops. A 3-layer GraphSAGE encoder (GAT is an option) with hidden width 128 gives each gate an embedding. The default model has 174,978 parameters.
- **Pair head.** An MLP scores each overlapping gate pair from the two embeddings, their absolute difference and product, and four pair features. A decision threshold is calibrated on validation circuits.
- **Window head.** Each window's embeddings are mean- and max-pooled and joined with six hand features: window size, active qubits, two-qubit fraction, and three rule-evidence features (log count of rule candidates inside the window, a flag for any, a flag for a non-inverse-pair candidate). An MLP produces a logit, and temperature scaling turns it into a probability.
- **Rule candidates.** A deterministic generator proposes inverse-pair and merge candidates. It is the evidence source for the rule features and the baseline the GNN is compared with.
- **Verification.** Rewrites proven by rules are applied directly. Extra rewrites proposed only by the GNN are checked by an equivalence checker (numeric when the lowest GNN score is above the confidence threshold, the full checker otherwise). A rewrite is kept only if equivalence is proven.

## 2. Design choices

- **Rules rank, the GNN breaks ties.** Windows are ordered by their rule-candidate count, and the calibrated GNN probability orders windows with equal counts. This follows from the out-of-fold results in Section 3.
- **Leakage-free evaluation.** Windows are scored out of fold: whole circuit families are held out together, and the fold models are trained inside the evaluation script, not taken from the saved checkpoint.
- **Labels.** A window is "reducible" if the solver found a reduction within the gate set, angle pool and time budget. Timeouts count as not-reducible. "Not reducible" therefore means no reduction was found, not that none exists.
- **Calibration.** Temperature scaling, cross-fitted so each fold's temperature is fitted on the other folds' logits.

## 3. Results

### 3.1 Acceptance

With the leakage-free GNN, rules removed 304 gates and the GNN removed 306 (calibration `default`, solver budget 10.0s). Only `grover_w5_i1` differs (4 vs 6).

### 3.2 Labels

74 circuits in 8 families give 2388 windows: 84 reducible, 2304 not-reducible, 0 unknown. Rule candidates cover 51 of the 84 positives, so 33 are found only by the solver. The positives come from 38 circuits in 5 families (grover and the four `rand_*`). ghz, qaoa and qft have none, and `rand_cnotpauli` holds 39 of the 84.

### 3.3 Window ranking (leave-family-out, 3 seeds)

| System | P@1 | P@3 | P@5 | AP | AUROC |
|---|---|---|---|---|---|
| Random | 0.046 | 0.045 | 0.045 | 0.038 | 0.500 |
| Rule-based (candidates in window) | 0.294 | 0.181 | 0.130 | 0.257 | 0.781 |
| Window length only | 0.080 | 0.083 | 0.076 | 0.086 | 0.759 |
| GNN window head | 0.225 | 0.140 | 0.108 | 0.153 | 0.621 |
| Rule + GNN tie-break | 0.315 | 0.187 | 0.131 | 0.283 | 0.744 |
| LR (hand features) | 0.345 | 0.206 | 0.145 | 0.348 | 0.820 |
| Rule + LR tie-break | 0.318 | 0.215 | 0.153 | 0.341 | 0.838 |
| GNN (no rule features) | 0.090 | 0.077 | 0.068 | 0.038 | 0.464 |
| LR (no rule features) | 0.199 | 0.129 | 0.097 | 0.205 | 0.779 |

Paired bootstrap over circuits (1000 resamples, 95% CI). Differences whose CI spans 0 are not distinguishable from noise at this circuit count.

| A vs B | ΔAP | ΔAUROC | ΔP@1 |
|---|---|---|---|
| GNN head vs rule | −0.072 [−0.197, +0.051] | −0.132 [−0.206, −0.061] | −0.030 [−0.135, +0.081] |
| Rule + GNN vs rule | +0.026 [−0.014, +0.068] | −0.042 [−0.080, −0.003] | +0.039 [−0.027, +0.108] |
| Rule + GNN vs GNN head | +0.098 [+0.002, +0.203] | +0.090 [+0.036, +0.158] | +0.069 [−0.014, +0.149] |
| GNN head vs LR (hand features) | −0.162 [−0.288, −0.031] | −0.171 [−0.244, −0.108] | −0.075 [−0.176, +0.014] |
| Rule + LR vs rule | +0.082 [+0.032, +0.142] | +0.058 [+0.012, +0.105] | +0.018 [−0.027, +0.068] |
| Rule + GNN vs Rule + LR | −0.056 [−0.117, +0.000] | −0.100 [−0.153, −0.049] | +0.021 [−0.041, +0.081] |
| GNN head vs GNN (no rule features) | +0.152 [+0.081, +0.234] | +0.166 [+0.089, +0.246] | +0.176 [+0.081, +0.284] |
| GNN vs LR (both no rule features) | −0.166 [−0.275, −0.070] | −0.294 [−0.397, −0.193] | −0.106 [−0.203, −0.014] |

What the table supports:

- The GNN head alone ranks worse than the rule baseline on AUROC, and the other two metrics are not distinguishable. Adding the GNN as a tie-break to the rules does not clearly improve AP or P@1 and is slightly worse on AUROC.
- The GNN's ranking skill depends on the rule features. Without them its AUROC is 0.464, below the length-only baseline (0.759).
- A logistic regression on the same six hand features beats the GNN head on AP and AUROC. The LR baseline sees exactly the extras the GNN head receives, built from whole-circuit rule candidates.
- The per-family AUROC of the GNN head is 0.737 (grover, 12 positives), 0.521 (`rand_cliffordrz`, 10), 0.674 (`rand_cliffordt`, 16), 0.692 (`rand_cnotpauli`, 39) and 0.683 (`rand_cxrz`, 7). `rand_cliffordrz` ranges from 0.46 to 0.56 across seeds, so it is indistinguishable from chance. The report has no per-family rule AUROC, so no per-family GNN-vs-rule claim is made.

### 3.4 Calibration

Pooled out-of-fold, mean temperature 2.044 ± 0.122. Negative log-likelihood improves from 0.2443 to 0.1837. Brier score goes from 0.0343 to 0.0373 and ECE from 0.0274 to 0.0348, both slightly worse; the constant base-rate Brier is 0.0339. Scaling is monotone, so ranking metrics are unchanged. With 84 positives most ECE bins are nearly empty, so ECE is only indicative.

### 3.5 Where the rules are silent

159 windows carry rule evidence (51 positives) and 2229 do not (33 positives).

| System | AP rule-flagged | AP unflagged |
|---|---|---|
| GNN window head | 0.392 | 0.018 |
| GNN (no rule features) | 0.312 | 0.019 |
| LR (hand features) | 0.481 | 0.147 |
| LR (no rule features) | 0.336 | 0.242 |
| Window length only | 0.246 | 0.045 |

On unflagged windows the GNN's AP of 0.018 is close to the base rate (33/2229 ≈ 0.015). The logistic regressions do better there, so the hand features carry signal that the GNN does not use.

### 3.6 Robustness across calibration profiles

| Profile | Rules removed | GNN removed | Differing rows |
|---|---|---|---|
| default | 304 | 306 | `grover_w5_i1` (4 vs 6) |
| superconducting_ibm_like | 304 | 306 | `grover_w5_i1` (4 vs 6) |
| trapped_ion_like | 304 | 306 | `grover_w5_i1` (4 vs 6) |

Gate removal on these circuits does not vary across the three profiles. Cost, fidelity and timing columns were not compared, so nothing is claimed about them.

### 3.7 Solver-budget sweep (6 circuits, at most 5 qubits)

Gate reduction %, with CX-cost reduction in brackets (2.6 for gnn, rule and random; 3.3 for exhaustive at every budget):

| Budget (s/circuit) | gnn | rule | random | exhaustive |
|---|---|---|---|---|
| 0.1 | 13.2 | 13.2 | 14.0 | 16.2 |
| 0.25 | 13.2 | 13.2 | 14.0 | 16.1 |
| 0.5 | 13.6 | 13.2 | 14.0 | 16.2 |
| 1.0 | 13.6 | 13.2 | 14.0 | 16.2 |

Optimization time per circuit (s): gnn 0.25 / 0.40 / 0.56 / 1.17, rule 0.24 / 0.37 / 0.55 / 1.13, random 0.23 / 0.37 / 0.55 / 1.14, exhaustive 140.74 / 144.45 / 142.94 / 139.89.

Read this as a cost-quality trade-off, not a ranking. The per-circuit tables explain the means (read at 1.0s for all three optimizers, and at 0.1s for GNN only).

- **GNN vs rule.** At 1.0s they are identical on 5 of 6 circuits. The only difference is `rand_cnotpauli_n5` (143 gates), where GNN reaches 109 gates (23.8%) and rule 113 (21.0%), 4 gates ahead with the same depth (57→47). At 0.1s GNN reaches 113 on that circuit, and the means at 0.1s and 0.25s are equal, so the 0.4-point mean gap appears from 0.5s.
- **Random.** At 1.0s random matches rule on 5 of 6 circuits and is ahead on one, `grover_w3_i2`, where it reaches 29 gates against 31 for rule and GNN.
- **What that means.** GNN's lead over rule and random's lead over GNN are each one circuit and a few gates. With 6 circuits and one run per setting, this is not evidence of an ordering between the three.
- **Exhaustive** is about 3 points ahead of all of them at roughly 120 to 560 times the optimization time (140 s against 0.25 to 1.2 s per circuit).
- **Caveat.** The sweep's GNN column was not re-run with the leakage-free checkpoint.

Solver time is within display rounding (0.05 s) of 6 × the budget at every budget (0.6, 1.5, 3.0, 6.0 s), so the budget is consumed in full on these circuits, including those with no reduction (ghz, qft). The pipeline documents the field as time actually spent; this report does not claim that individual solver calls time out.

### 3.8 Pair-level precision and recall

`scripts/eval_gnn.py`, seeds 0 to 2, 50 epochs, 11 circuits split 7/2/2 (train/val/test) by circuit. The model is trained inside the script on the train circuits only. Gold is the set of rule candidates that pass Module C's check (68 of 68 verified, all inverse pairs). This table therefore measures how well the pair head reproduces verified rule candidates. It does not measure discovery of reductions the rules miss, and the rule generator scores 1.000 by construction.

| Seed | Gold pairs in test | GNN precision | GNN recall | TP / FP / FN |
|---|---|---|---|---|
| 0 | 40 | 0.000 | 0.000 | 0 / 31 / 40 |
| 1 | 0 | undefined (printed as 1.000) | undefined (printed as 1.000) | 0 / 0 / 0 |
| 2 | 16 | 0.171 | 0.375 | 6 / 29 / 10 |

Seed 1's test circuits contain no gold pairs, so the 1.000 values in that run are 0/0 artifacts and carry no information. The adjacent-inverse baseline has recall 0 on seeds 0 and 2. The "rule labels" and "verified labels" GNN rows are identical in every seed because every candidate verified. The calibrated threshold was 0.95 in all three seeds. `calibrate_threshold` maximises F1 on the validation circuits and resolves ties to the higher threshold, so a flat F1 on 2 validation circuits ends at the highest threshold in the sweep, which is consistent with 0.95 every time; it does not show the threshold was tuned. At 0.95 the head still flagged 31 and 29 pairs on seeds 0 and 2, and none on seed 1.

With 11 circuits, a 2-circuit test set, a split by circuit (sibling grouping not confirmed) and one configuration, this shows that the pair head does not reliably recover the rule candidates. It does not establish how much worse it is. Files: `docs/gnn_eval/pr_table.md` (seed 0), `pr_table_seed1.md`, `pr_table_seed2.md`.

### 3.9 Scaling

Costs are read from the code, none measured (`docs/module_b_complexity_and_scaling.md`). The encoder is linear in the gate count G. The pair path enumerates all overlapping gate pairs in Θ(G²) before any model call, so it is the bottleneck on large circuits. The window path has no pair enumeration, and its cost depends on the rule generator (Θ(G²) to O(G³) worst case).

## 4. Limitations

1. **Few positives.** 84 positives from 38 circuits, at most 5 informative leave-family-out folds, and 39 of 84 from one family. Differences of a few points are noise.
2. **The GNN window head does not beat simple baselines.** A logistic regression on the same hand features is better, and without rule features the GNN is at chance. This head and training setup does not extract useful signal beyond what the hand features give. Why is not established here.
3. **No skill where rules are silent.** The 33 solver-only positives are exactly where the GNN adds nothing.
4. **Calibration.** Temperature scaling improves NLL but not Brier or ECE.
5. **Labels are not proofs of optimality.** Not-reducible means no reduction was found in the budget.
6. **Sweep scope.** 6 circuits of at most 5 qubits, one run per setting. GNN's lead over rule (one circuit, 4 gates) and random's lead over GNN (one circuit, 2 gates) are each a single-circuit effect, and the GNN column predates the leakage-free checkpoint.
7. **Verification.** The numeric checker used for confident GNN extras cannot prove equivalence past 8 qubits, so those extras are rejected. This loses gain and never accepts a wrong rewrite.
8. **Robustness.** Only gate removal was compared across calibration profiles.
9. **Pair head.** On 11 circuits with a 2-circuit test set, the pair head reached precision 0.000 and 0.171 on the two seeds that have gold pairs, and threshold selection was degenerate: ties resolve to the highest threshold, and 0.95 was chosen in every seed (Section 3.8).

## 5. Figures

- `docs/gnn_eval/fig_window_pk.png`: precision@k for all systems in 3.3.
- `docs/gnn_eval/fig_family_auroc.png`: per-family GNN AUROC.
- `docs/gnn_eval/window_ranking.png`: written by `eval_windows.py` through `plot_reliability`, so it is a reliability (calibration) plot, not a ranking plot.
- No pair-level precision/recall plot: only two seeds have gold pairs, so Section 3.8 gives a table instead.

Both `fig_*` files are generated by `scripts/plot_report_figures.py` from `window_ranking.md`. Error bars are seed standard deviation only.
