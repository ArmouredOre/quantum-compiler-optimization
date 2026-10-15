# Closed-loop reward: proxy vs real Stage 6 reward (issue #35)

End-to-end `HybridOptimizer` runs (Module A in the loop, rule-based Module B, no SMT re-synthesis), calibration `default`, seeds [0, 1]. Reward = Stage 6 `scalarized_reward(original, pipeline output)` (mean over circuits; mean ± std over seeds). Policies are the final checkpoints of 5-qubit training.

| Method | heldout_5q (n=40) | bench_5-6 (n=6) | bench_8-12 (n=7) | bench_16-20 (n=6) |
|---|---|---|---|---|
| untrained greedy stand-in | 0.0455 ± 0.0000 | 0.0436 ± 0.0000 | 0.0191 ± 0.0000 | 0.0198 ± 0.0000 |
| PPO, proxy reward (as #26) | 0.0815 ± 0.0006 | 0.0811 ± 0.0019 | 0.0397 ± 0.0019 | 0.0380 ± 0.0044 |
| **PPO, real Stage 6 reward** | 0.0821 ± 0.0000 | 0.0794 ± 0.0002 | 0.0372 ± 0.0006 | 0.0391 ± 0.0000 |

## Depth reduction %

| Method | heldout_5q | bench_5-6 | bench_8-12 | bench_16-20 |
|---|---|---|---|---|
| untrained greedy stand-in | 2.25 ± 0.00 | 4.04 ± 0.00 | 0.86 ± 0.00 | 1.66 ± 0.00 |
| PPO, proxy reward (as #26) | 8.88 ± 0.14 | 11.92 ± 0.46 | 5.52 ± 0.47 | 5.60 ± 1.11 |
| **PPO, real Stage 6 reward** | 9.02 ± 0.00 | 11.50 ± 0.05 | 4.90 ± 0.15 | 5.88 ± 0.00 |

## Gate-count reduction %

| Method | heldout_5q | bench_5-6 | bench_8-12 | bench_16-20 |
|---|---|---|---|---|
| untrained greedy stand-in | 8.90 ± 0.00 | 6.35 ± 0.00 | 3.65 ± 0.00 | 3.22 ± 0.00 |
| PPO, proxy reward (as #26) | 11.16 ± 0.00 | 7.68 ± 0.00 | 4.12 ± 0.00 | 3.80 ± 0.00 |
| **PPO, real Stage 6 reward** | 11.16 ± 0.00 | 7.68 ± 0.00 | 4.12 ± 0.00 | 3.80 ± 0.00 |
