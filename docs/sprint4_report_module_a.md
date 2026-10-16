# Sprint 4 report - Module A (#35 closed-loop reward, #46 parallel blocks and partitioner tuning)

## 1. What was done

### #35 closed-loop reward
- `qco/modules/rl_scheduler/closed_loop.py`: `RewardLog` (consumes the pipeline `RewardEvent` hook per
  block and per iteration, CSV export), `PipelineReward` (the Stage 6 reward of running the downstream
  pipeline on the agent's scheduled circuit, cached), `evaluate_end_to_end`, `write_reward_curves`.
- `RLConfig.reward_source = "proxy" | "pipeline"` (plus `pipeline_resynthesis`, `eval_pipeline_reward`).
  With `"pipeline"` the terminal bonus is the Stage 6 reward of the agent's circuit minus that of the
  CR-only reference, so the agent is credited only for what it adds beyond CR.
- `HybridOptimizer` now loads `models/rl_ppo_5q.json` by default (`rl_policy_path="auto"`; `None` forces
  the untrained stand-in; an explicit path is loaded and errors are raised).
- `scripts/eval_closed_loop.py` trains proxy and pipeline runs per seed and evaluates them end to end.

### #46 parallel blocks and tuning
- `PipelineConfig.parallel_blocks`, `HybridOptimizer.optimize_blocks` (hand-off for #36), see `docs/parallel_blocks.md`.
- `benchmarks/tune_partition.py` and `docs/partitioner_tuning.md`; `benchmarks/bench_parallel.py`.

## 2. Results (honest reading)
- **#35:** real-reward vs proxy-reward PPO (2 seeds, 40k steps, end-to-end Stage 6 reward):
  heldout_5q 0.0821 vs 0.0815, bench_5-6 0.0794 vs 0.0811, bench_8-12 0.0372 vs 0.0397,
  bench_16-20 0.0391 vs 0.0380. Gate reduction is identical. The differences are inside seed noise, so
  **the real reward is not measurably better than the proxy** here; both beat the untrained stand-in
  (e.g. 0.0815 vs 0.0455). Likely reason: with rule-based Module B and no resynthesis, the proxy already
  tracks the downstream reward. The committed default checkpoint was therefore left unchanged.
  Files: `docs/rl_training/closed_loop/{closed_loop.md,closed_loop.json,reward_curves.svg}`, per-run `metrics.csv`/`curves.svg`.
  Caveat: only 2 seeds x 40k steps (the 3 x 100k plan did not fit the 2-core sandbox). Run `make eval-closed-loop`
  with more seeds/steps for a firmer answer.
- **Side finding:** the trained agent includes CR, so Module B's rule-proven cancellations are already
  gone; two existing tests were pinned to `rl_policy_path=None` for that reason.
- **Behaviour change to know about:** the pipeline default now uses the trained agent, so default
  pipeline outputs differ from the stand-in.
- **#46 tuning:** see `docs/partitioner_tuning.md` (greedy ignores `max_block_gates` >= 32; fallback
  has larger cuts but more gate reduction at large blocks; recommendation: keep defaults).
- **#46 parallel:** identical circuits to serial when the solver budget is not binding; slower without
  resynthesis (0.55-0.62x), 2.0-2.3x with a binding 20 s budget but that is partly less work done.

## 3. How to test and deploy
1. `git checkout -b sprint4/module-a-closed-loop origin/main` then `git apply sprint4.patch` (or untar over the repo).
2. `pip install -e '.[test]'` then `python -m pytest -q` (all green here; numpy/torch/z3 optional tests skip).
3. New tests: `tests/test_rl_closed_loop.py` (16) and `tests/test_parallel_blocks.py` (7).
4. Try it: `python -c "from qco.pipeline import optimize"`; `PipelineConfig(parallel_blocks=2)`;
   `make tune-partition`, `make bench-parallel`, `make eval-closed-loop` (slow).
5. Open the PR in two parts if preferred: #35 (rl_scheduler, pipeline default, tests, docs) and #46
   (pipeline parallel code, benchmarks, docs).

## 4. How to write further unit tests
- Pure Python only; CI installs `.[test]` (pyyaml, pytest). For optional deps use an `*_AVAILABLE` flag with
  a class- or function-scoped `skipif`, never a module-level skip.
- Keep configs tiny (see `TINY` in `tests/test_rl_closed_loop.py`): few steps, `numeric_backend="python"`.
- Compare circuits through `ir.to_qasm2()`, and pin `rl_policy_path=None` when a test needs the stand-in.
- Parallel tests: skip without `fork`; use no resynthesis or exhaustive ranking so the budget is not binding;
  simulate missing fork by monkeypatching `multiprocessing.get_context`.
- Reward hooks: pass `reward_hook=RewardLog()` in `PipelineConfig` and assert on `summary()`.

## 5. Open items
- Hand-off for #36: call `optimize_blocks` and keep `parallel_blocks=1` unless resynthesis dominates.
- METIS/KaHyPar rows are missing from the tuning table (not installable here).
- Parallel speedup under a non-binding budget is unmeasured (too slow here).
- `benchmark_runner.DEFAULT_RL_POLICY` pointed at `docs/rl_training/ppo_5q/best.json`; fixed in #41 (it now matches the pipeline default).
