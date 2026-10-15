## Window ranking + calibration (8-fold leave-family-out CV, 3 seed(s) from 0)

- Labels: 84 reducible / 2304 not-reducible; timeouts counted as not-reducible. 'Not reducible' means no reduction found within the gate set, angle pool and time budget, not provably optimal.
- Positives come from 38 of 74 circuits; rule candidates cover 51/84 positives.
- Folds: whole families are held out together.
- ± is the std over CV seeds / tie-breaks only; it does not capture the small number of positive circuits, so treat differences of a few points as noise.
- P@k pools hits over circuits (each circuit contributes min(k, #windows) slots). Rule-based scores are heavily tied, so its P@1 is an expectation over random tie-breaks.
- LR baselines are ridge logistic regressions (numpy IRLS, standardised features) on the same per-window extras the GNN head receives, trained on the same folds.
- Calibration (pooled out-of-fold; T cross-fitted, each fold's T fitted on the other folds' logits): mean T = 2.044 ± 0.122; NLL 0.2443 -> 0.1837; Brier 0.0343 -> 0.0373 (constant base-rate Brier = 0.0339); ECE 0.0274 -> 0.0348. Temperature scaling is monotone, so it does not change ranking metrics.
- ECE uses 10 equal-width bins and only 84 positives, so most bins are nearly empty; treat it as indicative, not conclusive.


Base rate of reducible windows: 84/2388

### Ranking (all circuits; AP/AUROC pooled over windows)

| System | P@1 | P@3 | P@5 | AP | AUROC |
|---|---|---|---|---|---|
| Random | 0.046 ± 0.004 | 0.045 ± 0.002 | 0.045 ± 0.001 | 0.038 ± 0.001 | 0.500 ± 0.000 |
| Rule-based (candidates in window) | 0.294 ± 0.001 | 0.181 ± 0.002 | 0.130 ± 0.001 | 0.257 ± 0.001 | 0.781 ± 0.000 |
| Window length only | 0.080 ± 0.005 | 0.083 ± 0.004 | 0.076 ± 0.000 | 0.086 ± 0.001 | 0.759 ± 0.000 |
| GNN window head | 0.225 ± 0.006 | 0.140 ± 0.007 | 0.108 ± 0.005 | 0.153 ± 0.045 | 0.621 ± 0.045 |
| Rule + GNN tie-break | 0.315 ± 0.006 | 0.187 ± 0.006 | 0.131 ± 0.001 | 0.283 ± 0.011 | 0.744 ± 0.015 |
| LR (hand features) | 0.345 ± 0.002 | 0.206 ± 0.000 | 0.145 ± 0.000 | 0.348 ± 0.000 | 0.820 ± 0.000 |
| Rule + LR tie-break | 0.318 ± 0.002 | 0.215 ± 0.000 | 0.153 ± 0.000 | 0.341 ± 0.000 | 0.838 ± 0.000 |
| GNN (no rule features) | 0.090 ± 0.013 | 0.077 ± 0.004 | 0.068 ± 0.003 | 0.038 ± 0.003 | 0.464 ± 0.028 |
| LR (no rule features) | 0.199 ± 0.001 | 0.129 ± 0.001 | 0.097 ± 0.000 | 0.205 ± 0.000 | 0.779 ± 0.000 |

### P@k restricted to circuits with at least one positive window

| System | P@1* | P@3* | P@5* |
|---|---|---|---|
| Random | 0.084 ± 0.002 | 0.089 ± 0.003 | 0.089 ± 0.002 |
| Rule-based (candidates in window) | 0.586 ± 0.008 | 0.352 ± 0.002 | 0.252 ± 0.000 |
| Window length only | 0.158 ± 0.004 | 0.157 ± 0.003 | 0.144 ± 0.002 |
| GNN window head | 0.439 ± 0.012 | 0.272 ± 0.014 | 0.207 ± 0.009 |
| Rule + GNN tie-break | 0.614 ± 0.012 | 0.363 ± 0.011 | 0.251 ± 0.002 |
| LR (hand features) | 0.671 ± 0.003 | 0.399 ± 0.001 | 0.279 ± 0.000 |
| Rule + LR tie-break | 0.618 ± 0.003 | 0.417 ± 0.001 | 0.295 ± 0.000 |
| GNN (no rule features) | 0.175 ± 0.025 | 0.149 ± 0.007 | 0.130 ± 0.007 |
| LR (no rule features) | 0.391 ± 0.002 | 0.248 ± 0.002 | 0.186 ± 0.001 |

### Top-1 hits (expected number of circuits whose top window is reducible)

Denominators: 74 circuits (all), 38 circuits with a positive (*).

| System | hits@1 (all) | hits@1* (positive circuits) |
|---|---|---|
| Random | 3.417 ± 0.312 | 3.183 ± 0.062 |
| Rule-based (candidates in window) | 21.733 ± 0.062 | 22.283 ± 0.306 |
| Window length only | 5.883 ± 0.342 | 6.017 ± 0.165 |
| GNN window head | 16.667 ± 0.471 | 16.667 ± 0.471 |
| Rule + GNN tie-break | 23.333 ± 0.471 | 23.333 ± 0.471 |
| LR (hand features) | 25.517 ± 0.131 | 25.483 ± 0.125 |
| Rule + LR tie-break | 23.517 ± 0.131 | 23.483 ± 0.125 |
| GNN (no rule features) | 6.667 ± 0.943 | 6.667 ± 0.943 |
| LR (no rule features) | 14.700 ± 0.082 | 14.850 ± 0.071 |

### Per-family GNN AUROC (families with positives)

| Family | Positives | GNN AUROC (out-of-fold) |
|---|---|---|
| grover | 12 | 0.737 ± 0.068 |
| rand_cliffordrz | 10 | 0.521 ± 0.044 |
| rand_cliffordt | 16 | 0.674 ± 0.027 |
| rand_cnotpauli | 39 | 0.692 ± 0.025 |
| rand_cxrz | 7 | 0.683 ± 0.022 |

### Stratified AP (rule-flagged: 159 windows, 51 positives; unflagged: 2229 windows, 33 positives)

Unflagged windows have all-zero rule evidence, so skill there is not explained by the rules. Few positives: report ± as seed noise only, no significance claim.

| System | AP rule-flagged | AP unflagged |
|---|---|---|
| GNN window head | 0.392 ± 0.034 | 0.018 ± 0.007 |
| GNN (no rule features) | 0.312 ± 0.016 | 0.019 ± 0.005 |
| LR (hand features) | 0.481 ± 0.000 | 0.147 ± 0.000 |
| LR (no rule features) | 0.336 ± 0.000 | 0.242 ± 0.000 |
| Window length only | 0.246 ± 0.002 | 0.045 ± 0.001 |

### Paired bootstrap over circuits (seed-averaged scores, 1000 resamples, 95% CI)

| A vs B | ΔAP | ΔAUROC | ΔP@1 |
|---|---|---|---|
| GNN window head vs Rule-based (candidates in window) | -0.072 [-0.197, +0.051] | -0.132 [-0.206, -0.061] | -0.030 [-0.135, +0.081] |
| Rule + GNN tie-break vs Rule-based (candidates in window) | +0.026 [-0.014, +0.068] | -0.042 [-0.080, -0.003] | +0.039 [-0.027, +0.108] |
| Rule + GNN tie-break vs GNN window head | +0.098 [+0.002, +0.203] | +0.090 [+0.036, +0.158] | +0.069 [-0.014, +0.149] |
| GNN window head vs LR (hand features) | -0.162 [-0.288, -0.031] | -0.171 [-0.244, -0.108] | -0.075 [-0.176, +0.014] |
| Rule + GNN tie-break vs Rule + LR tie-break | -0.056 [-0.117, +0.000] | -0.100 [-0.153, -0.049] | +0.021 [-0.041, +0.081] |
| Rule + LR tie-break vs Rule-based (candidates in window) | +0.082 [+0.032, +0.142] | +0.058 [+0.012, +0.105] | +0.018 [-0.027, +0.068] |
| GNN window head vs GNN (no rule features) | +0.152 [+0.081, +0.234] | +0.166 [+0.089, +0.246] | +0.176 [+0.081, +0.284] |
| GNN (no rule features) vs LR (no rule features) | -0.166 [-0.275, -0.070] | -0.294 [-0.397, -0.193] | -0.106 [-0.203, -0.014] |

A CI that spans 0 means the difference is not distinguishable from noise at this circuit count.
