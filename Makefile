.PHONY: help setup benchmarks check test clean

help:
	@echo "setup       - create venv and install dev extras"
	@echo "benchmarks  - (re)generate the OpenQASM benchmark suite"
	@echo "check       - report which optional research stacks are importable"
	@echo "test        - run the Phase 2 smoke tests"
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

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
