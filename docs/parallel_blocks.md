# Parallel block processing (#46)

`PipelineConfig.parallel_blocks` (default 1 = serial) runs the per-block Stage 4 work
(Module A schedule, Module B, Module C verification, N1 window resynthesis) in forked worker
processes. Entry point for #36 (Swaraj): `HybridOptimizer.optimize_blocks(blocks, workers=None)`
returns the verified variants of every block in block order; `run()` already calls it.

## Design
- `fork` workers inherit the optimizer (trained policy, GNN, checkers) copy-on-write; only a chunk
  of `Block`s goes in and variants plus audit deltas (revert events, unverified set, verification
  counters, solver time) come back and are merged in block order. Threads would not help (GIL).
- No `fork` (Windows): `RuntimeWarning` and serial run. One block or `workers<=1`: serial.
- `PipelineResult.parallel_workers` and `block_stage_s` report what was used and the block-stage time.

## Identity with serial
Tests (`tests/test_parallel_blocks.py`) check identical Pareto circuits and identical audit counters
(blocks, reverts, unverified, verification calls/skips) on rand_cliffordt_n20 without resynthesis,
and with exhaustive-ranking resynthesis on a small circuit. The solver budget is wall-clock based, so
identity is guaranteed only when it is not binding. With a binding budget each parallel block gets
`budget / n_blocks`, while serial is first-come, so outputs can differ (seen once below).

## Measured speedup (2 cores, block stage, median of 3)
| circuit | mode | workers | block stage s | total s | speedup (block stage) | identical to serial |
|---|---|---|---|---|---|---|
| rand_cliffordt_n20 | no resynthesis | 1 | 1.229 | 1.241 | 1.00x | yes |
| rand_cliffordt_n20 | no resynthesis | 2 | 2.217 | 2.229 | 0.55x | yes |
| rand_cliffordt_n20 | resynthesis, 20 s budget | 1 | 21.776 | 21.790 | 1.00x | yes |
| rand_cliffordt_n20 | resynthesis, 20 s budget | 2 | 9.409 | 9.421 | 2.31x | yes |
| rand_cnotpauli_n20 | no resynthesis | 1 | 1.906 | 1.917 | 1.00x | yes |
| rand_cnotpauli_n20 | no resynthesis | 2 | 3.060 | 3.071 | 0.62x | yes |
| rand_cnotpauli_n20 | resynthesis, 20 s budget | 1 | 21.802 | 21.814 | 1.00x | yes |
| rand_cnotpauli_n20 | resynthesis, 20 s budget | 2 | 10.702 | 10.714 | 2.04x | NO |

How to read it:
- **No resynthesis: parallel is slower (0.55-0.62x).** Blocks cost ~25 ms each, so fork, pickling
  and merging cost more than they save. Keep `parallel_blocks=1` here.
- **With resynthesis (20 s budget): 2.0-2.3x, but this is not a clean speedup.** Serial spends the
  whole 20 s; parallel blocks each get a smaller share and finish earlier, i.e. less solver work is
  done. The rand_cnotpauli_n20 run differs from serial for exactly that reason (flagged `NO`).
  A like-for-like speedup needs a non-binding budget, which is too slow to run here.
- Conclusion: parallelism is worth it only when resynthesis dominates; default stays serial.

Reproduce: `python benchmarks/bench_parallel.py --workers 1 2 --repeat 3`.
