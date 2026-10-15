.PHONY: help setup benchmarks check test eval clean train-rl train-rl-smoke eval-generalization eval-backends

help:
	@echo "setup       - create venv and install dev extras"
	@echo "benchmarks  - (re)generate the OpenQASM benchmark suite"
	@echo "check       - report which optional research stacks are importable"
	@echo "test        - run the Phase 2 smoke tests"
	@echo "eval        - baselines + ablations + window-selection tables into results/"
	@echo "train-rl   - full Module A PPO training (configs/rl_ppo_5q.yaml)"
	@echo "eval-generalization - train 5q, zero-shot 5/10/20q table (#26)"
	@echo "eval-backends - per-device-profile policies + cross-backend matrix (#29)"
	@echo "train-rl-smoke - tiny Module A training smoke run"
	@echo "clean       - remove caches and build artifacts"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

benchmarks:
	python3 benchmarks/generate_circuits.py

check:
	python3 scripts/check_environment.py

test:
	python3 -m pytest

eval:
	PYTHONPATH=src python3 -m qco.evaluation.benchmark_runner --out results

train-rl:
	python3 scripts/train_rl.py --config configs/rl_ppo_5q.yaml --out runs/rl_5q --compare

eval-generalization:
	python3 scripts/eval_generalization.py --seeds 0 1 2 --timesteps 100000 --out docs/rl_training --save-checkpoint models/rl_ppo_5q.json

eval-backends:
	python3 scripts/eval_backend_matrix.py --seeds 0 1 2 --timesteps 60000 --out docs/rl_training --models models/backend

train-rl-smoke:
	python3 scripts/train_rl.py --config configs/rl_smoke.yaml --out runs/rl_smoke

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +

eval-closed-loop:  ## #35: proxy vs real-reward training and end-to-end eval (slow)
	python scripts/eval_closed_loop.py --stage all --seeds 0 1 --timesteps 40000 --out docs/rl_training/closed_loop

tune-partition:  ## #46: partitioner tuning table
	python benchmarks/tune_partition.py --downstream --out docs/partitioner_tuning.md

bench-parallel:  ## #46: serial vs parallel block stage
	python benchmarks/bench_parallel.py --workers 1 2
