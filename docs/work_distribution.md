# Team & Work Distribution

**Project:** Quantum Compiler Optimization — Gate Scheduling & Gate Cancellation
**Team 2 · Review 1**

Each member owns one or more architecture modules end‑to‑end (literature deep‑dive,
design, implementation, evaluation). The integration owner additionally wires the
modules into one pipeline and runs benchmarking.

## Roster

| Member | Name | Registration No. | Role | Owned module / stage | Package(s) |
|:------:|------|------------------|------|----------------------|-----------|
| 2 | **Maanas Nair** | 24BCT0157 | RL Engineer | Module A — RL‑Based Adaptive Gate Scheduler (Stage 4·A) | `qco.modules.rl_scheduler` |
| 3 | **Prisha** | 24BDS0146 | ML / Graph‑Learning Engineer | Module B — GNN‑Based Gate Cancellation Predictor (Stage 4·B) | `qco.modules.gnn_cancellation` |
| 4 | **Sahib Singh** | 24BCI0292 | Formal‑Methods Engineer | Module C — SAT/SMT‑Based Global Optimizer & Verifier (Stage 4·C) | `qco.modules.smt_verifier` |
| 5 | **Swaraj Rane** | 24BCT0086 | Integration & Evaluation Lead · Front‑End & Graph Infrastructure · Repository / CI / DevOps Owner *(team coordination)* | Circuit IR, DAG/Hypergraph, Partitioning & block scheduling (Stages 1–3) **+** Module D — Evolutionary Multi‑Objective Compiler (Stage 5) **+** Evaluation & Benchmarking Engine (Stage 6) **+** end‑to‑end integration | `qco.ir`, `qco.graphs`, `qco.modules.evolutionary`, `qco.evaluation`, `qco.pipeline` |

## Responsibilities in detail

| Member | Key responsibilities |
|--------|----------------------|
| **Maanas Nair** (2) | Design the RL environment (state, action, reward) over the circuit DAG; implement the PPO/DDQN agent with the deterministic Commutation‑and‑Reduction hybrid pass; train on small circuits and evaluate generalization to larger ones. |
| **Prisha** (3) | Design the gate/node feature encoding; implement the GraphSAGE/GAT model; generate labeled training data (valid & profitable cancellations); evaluate prediction precision/recall against the rule‑based baseline. |
| **Sahib Singh** (4) | Formulate the SMT equivalence‑checking encoding for proposed rewrites; implement bounded‑window exact re‑optimization; integrate Z3/CVC5 solver calls into the pipeline. |
| **Swaraj Rane** (5) | Build the OpenQASM/Qiskit/Cirq parser and internal IR; implement the gate‑dependency DAG builder and the qubit‑interaction hypergraph builder; implement and benchmark the multilevel graph‑partitioning block scheduler; implement the NSGA‑II multi‑objective search (pymoo/DEAP) over candidate circuits; build the Evaluation & Benchmarking Engine for the five parameters; integrate all modules into one closed‑loop pipeline; own the Git repository, branch protection and CI; prepare the benchmark suite, comparison plots and the final report/presentation. |

## Roles at a glance

- **Maanas Nair — RL Engineer:** the adaptive scheduler; the only module with a
  training loop driven by the Stage 6 reward (closed loop).
- **Prisha — ML / Graph‑Learning Engineer:** the learned cancellation predictor;
  data‑science branch (BDS) aligns with the graph‑ML workload.
- **Sahib Singh — Formal‑Methods Engineer:** the correctness gate — no rewrite is
  accepted until his equivalence check passes.
- **Swaraj Rane — Integration & Evaluation Lead:** owns everything upstream of
  the optimization modules (the IR every other member codes against, the DAG /
  hypergraph and the partitioner — taken over from the former M1 slot),
  Module D, the benchmarking engine, the pipeline that stitches Stages 1–6 together, and the project
  infrastructure (repo, `main` branch protection, GitHub Actions CI). Acts as
  team coordinator for integration and reviews.

## Roster change

Ajay M (24BCE2493, former M1) is no longer a member of the team. The Stages 1–3
scope (`qco.ir`, `qco.graphs`) is reassigned to **Swaraj Rane**. Member IDs
M2–M5 are kept unchanged so existing commit, issue and PR references stay valid.
