.PHONY: help setup benchmarks check test eval clean train-rl train-rl-smoke

help:
	@echo "setup       - create venv and install dev extras"
	@echo "benchmarks  - (re)generate the OpenQASM benchmark suite"
	@echo "check       - report which optional research stacks are importable"
	@echo "test        - run the Phase 2 smoke tests"
	@echo "eval        - baselines + ablations + window-selection tables into results/"
	@echo "train-rl   - full Module A PPO training (configs/rl_ppo_5q.yaml)"
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

train-rl-smoke:
	python3 scripts/train_rl.py --config configs/rl_smoke.yaml --out runs/rl_smoke

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
