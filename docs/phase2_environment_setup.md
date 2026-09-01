# Phase 2 — Architecture Design & Environment Setup

**Checkpoint:** Review 1.
**Deliverables:** (a) finalized architecture diagram, (b) tool/library setup,
(c) benchmark circuit suite selected.

---

## (a) Finalized architecture

See [`architecture.md`](architecture.md) for the full six‑stage closed‑loop
description, data contracts, and code mapping, and
[`architecture-diagram.svg`](architecture-diagram.svg) for the presentation‑ready
vector diagram.

Summary of what was locked for Phase 2:

* **Six stages**: Front End → Graph Construction → Partitioning & Block
  Scheduling → Cooperating Optimization Modules (A/B/C) → Evolutionary
  Multi‑Objective Search (D) → Evaluation & Benchmarking Engine.
* **Closed loop**: the evaluation engine returns a scalarized 5‑metric score as
  the RL reward (edge `EV → Module A`).
* **Verification invariant**: no rewrite from Module A or B is accepted until
  Module C confirms unitary equivalence.
* **Output is a Pareto front**, not a single circuit.

## (b) Tool / library setup

| Layer | Tools / libraries | Declared in |
|-------|-------------------|-------------|
| Circuit representation & baseline compiler | Qiskit (Terra/transpiler), Cirq, pytket | `pyproject.toml` `[project.optional-dependencies] compiler` |
| Graph / hypergraph processing | NetworkX (DAG), KaHyPar / METIS (multilevel hypergraph partitioning) | `graphs` extra |
| Reinforcement learning | PyTorch, Stable‑Baselines3 / custom PPO, Gymnasium‑style env | `rl` extra |
| Graph neural networks | PyTorch Geometric (PyG) or DGL, GraphSAGE / GAT | `gnn` extra |
| Formal verification / SAT‑SMT | Z3 or CVC5, Qiskit `Operator` equivalence checking for validation | `smt` extra |
| Evolutionary multi‑objective search | pymoo or DEAP (NSGA‑II) | `ea` extra |
| Simulation & fidelity estimation | Qiskit Aer noisy simulator, custom noise models | `compiler` extra |
| Benchmark circuits | MQT Bench, QASMBench, standard algorithms (QFT, GHZ, Grover, QAOA, random) | `benchmarks/` |
| Project management / VCS | Git/GitHub, Jupyter, shared docs | repo |

Install everything for development:

```bash
pip install -e ".[dev]"     # dev = compiler + graphs + rl + gnn + smt + ea + test tooling
```

Or install per‑module while prototyping (Phase 3):

```bash
pip install -e ".[rl]"      # Member 2
pip install -e ".[gnn]"     # Member 3
pip install -e ".[smt]"     # Member 4
pip install -e ".[ea]"      # Member 5
```

`python scripts/check_environment.py` prints a table of which optional stacks are
importable in the current environment — the Phase 2 "environment set up" check.

### Package skeleton (maps 1‑to‑1 onto the architecture)

```
src/qco/
├── ir/                     Stage 1
│   ├── parser.py           OpenQASM 2.0 / Qiskit / Cirq  -> IR
│   └── intermediate_representation.py   Gate, IntermediateRepresentation
├── graphs/                 Stage 2 + 3
│   ├── dag.py              gate-dependency DAG
│   ├── hypergraph.py       qubit-interaction hypergraph
│   └── partitioning.py     multilevel partitioning -> blocks
├── modules/
│   ├── rl_scheduler/       Stage 4 Module A  (environment.py, agent.py)
│   ├── gnn_cancellation/   Stage 4 Module B  (features.py, model.py)
│   ├── smt_verifier/       Stage 4 Module C  (equivalence.py, bounded_optimizer.py)
│   └── evolutionary/       Stage 5 Module D  (nsga2.py)
├── evaluation/             Stage 6           (metrics.py, benchmark_runner.py)
└── pipeline.py             wires Stage 1..6 + closed-loop feedback
```

Every module is a runnable skeleton: dataclasses and function signatures are
final, bodies raise `NotImplementedError` with a `# Phase 3:` note so ownership
and interfaces are frozen before prototyping starts. The IR, the DAG builder, and
the five evaluation metrics are **fully implemented** in Phase 2 because every
downstream module depends on them.

## (c) Benchmark circuit suite

Defined in [`../benchmarks/suite.yaml`](../benchmarks/suite.yaml). Selection
rationale:

| Family | Sizes (qubits) | Why it is in the suite |
|--------|----------------|------------------------|
| **GHZ** | 5, 10, 20, 40 | Minimal entangler; trivial ground truth for the SMT equivalence check |
| **QFT** | 5, 8, 12, 16 | Dense rotations + long commutation chains — stresses the RL scheduler |
| **Grover** (1–2 iterations) | 5, 8, 12 | Repeated oracle/diffuser structure — many exact cancellations |
| **QAOA** (p = 1, 2; random 3‑regular) | 6, 10, 16 | Parameterized, hardware‑relevant; tests fidelity surrogate in Module D |
| **Random Clifford+T** | 5, 10, 20 | Matches the training distribution reported by Tao et al. 2026 |
| **Random CNOT+Pauli** | 5, 10, 20 | Second gate set from Tao et al. 2026; isolates two‑qubit‑gate reduction |
| *(external)* **MQT Bench / QASMBench** | as published | Cross‑checks results against a community standard once Qiskit is installed |

`benchmarks/generate_circuits.py` writes OpenQASM 2.0 files for the six
locally‑generated families into `benchmarks/circuits/` with **no third‑party
dependencies**, so the suite is reproducible on a bare Python install. MQT
Bench / QASMBench circuits are pulled in Phase 3 once the `compiler` extra is
installed.

```bash
python benchmarks/generate_circuits.py            # default suite.yaml
python benchmarks/generate_circuits.py --list     # show what would be generated
```
