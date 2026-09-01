# Quantum Compiler Optimization — Gate Scheduling & Gate Cancellation

[![CI](https://github.com/ArmouredOre/quantum-compiler-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/ArmouredOre/quantum-compiler-optimization/actions/workflows/ci.yml)

**Team 2** · Review 1 project

A hybrid quantum‑circuit optimizer that combines **reinforcement learning**
(adaptive gate scheduling), a **graph neural network** (gate‑cancellation
prediction), **SAT/SMT** solving (equivalence verification + bounded exact
re‑synthesis), and an **NSGA‑II evolutionary search** (multi‑objective
refinement) into a single closed‑loop pipeline that aims to beat Qiskit's fixed
O0–O3 transpiler passes on gate count, depth, execution time, fidelity, and
optimizer runtime.

---

## Status

| Phase | Scope | State |
|-------|-------|-------|
| **Phase 1** | Problem study & literature survey | ✅ complete — see [`docs/phase1_literature_survey.md`](docs/phase1_literature_survey.md), [`docs/research_gap.md`](docs/research_gap.md) |
| **Phase 2** | Architecture design & environment setup | ✅ complete — see [`docs/architecture.md`](docs/architecture.md), [`docs/phase2_environment_setup.md`](docs/phase2_environment_setup.md), [`benchmarks/`](benchmarks/) |
| Phase 3 | Module‑wise prototyping | ⏳ next (Review 2) — module skeletons in `src/qco/` |
| Phase 4 | Integration | not started |
| Phase 5 | Evaluation & benchmarking | not started |
| Phase 6 | Final report & demo | not started |

This repository currently delivers **Phase 1 + Phase 2**: the consolidated
literature survey, the finalized six‑stage architecture (with a clean diagram),
an installable Python package skeleton mapping 1‑to‑1 onto the architecture, and
a generated benchmark circuit suite.

---

## Repository layout

```
docs/                     Review-1 documentation (Phase 1 + Phase 2 deliverables)
  architecture.md          <- finalized architecture + clean Mermaid diagram
  architecture-diagram.svg <- presentation-ready vector diagram
  phase1_literature_survey.md
  research_gap.md
  evaluation_parameters.md
  work_distribution.md
  project_timeline.md
  phase2_environment_setup.md
src/qco/                  Installable package, one sub-package per architecture stage
benchmarks/               Selected benchmark suite + a dependency-free QASM generator
  suite.yaml
  generate_circuits.py
  circuits/                generated OpenQASM 2.0 files
tests/                    Smoke tests for the Phase 2 scaffold
scripts/check_environment.py
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # full research stack: see pyproject.toml
python benchmarks/generate_circuits.py   # (re)generate the benchmark suite
python scripts/check_environment.py       # report which optional deps are present
pytest -q
```

The benchmark generator and the smoke tests have **no third‑party
dependencies** so the scaffold verifies on a bare Python 3.10+ install; the RL /
GNN / SMT / EA stacks are only needed from Phase 3 onward.

## Documentation index

* [Architecture](docs/architecture.md) — the six‑stage closed‑loop pipeline
* [Literature survey](docs/phase1_literature_survey.md) — 4 sub‑topics + baseline
* [Research gap](docs/research_gap.md)
* [Evaluation parameters](docs/evaluation_parameters.md) — the 5 measured metrics
* [Work distribution](docs/work_distribution.md) — 5 members, one module each
* [Project timeline](docs/project_timeline.md) — 6 phases / 3 reviews
* [Phase 2 — environment setup](docs/phase2_environment_setup.md)
