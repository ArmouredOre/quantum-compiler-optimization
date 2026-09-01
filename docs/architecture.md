# Proposed System Architecture

**Project:** Quantum Compiler Optimization — Gate Scheduling & Gate Cancellation
**Team 2 · Review 1 · Phase 2 deliverable (Architecture Design)**

---

## 1. Overview

The optimizer is a **six‑stage closed‑loop pipeline**. An input circuit is first
lifted into an internal representation, expressed in two complementary graph
forms (a **DAG** for gate‑dependency scheduling and a **hypergraph** for
qubit‑interaction partitioning), broken into near‑independent blocks, and then
processed by three cooperating optimization modules (RL scheduler, GNN
cancellation predictor, SAT/SMT verifier). The surviving verified candidates are
refined by an evolutionary multi‑objective search into a **Pareto front**, and an
evaluation engine scores every candidate against the Qiskit O0–O3 baselines. The
evaluation result is fed **back** into the RL reward, closing the loop.

Design principles:

| Principle | Consequence in the architecture |
|-----------|---------------------------------|
| No single technique is sufficient | RL + GNN + SAT/SMT + EA are combined, not compared in isolation |
| Learned heuristics need a safety net | *Every* rewrite passes an SMT equivalence check before it is accepted |
| Exact methods do not scale | SAT/SMT runs only on bounded local windows, not the whole circuit |
| Objectives conflict | Output is a Pareto front (gate count, depth, fidelity), not one circuit |
| Fixed pipelines are sub‑optimal | Evaluation feedback tunes the RL policy per circuit / backend |

---

## 2. Architecture Diagram (clean)

```mermaid
flowchart TD
    IN["Input circuit&#10;OpenQASM 2.0 / Qiskit / Cirq"]

    subgraph S1["Stage 1 — Front End"]
        P["Parser &amp; IR Builder&#10;(Member 1)"]
    end

    subgraph S2["Stage 2 — Graph Construction"]
        DAG["Gate-dependency DAG&#10;nodes = gates, edges = qubit-wire deps"]
        HG["Qubit-interaction hypergraph&#10;vertices = qubits, hyperedges = multi-qubit gates"]
    end

    subgraph S3["Stage 3 — Partitioning &amp; Block Scheduling"]
        PART["Multilevel partitioning&#10;KaHyPar / METIS -> near-independent blocks"]
    end

    subgraph S4["Stage 4 — Cooperating Optimization Modules  (per block)"]
        direction LR
        MA["Module A&#10;RL Adaptive Scheduler&#10;PPO / DDQN&#10;(Member 2)"]
        MB["Module B&#10;GNN Cancellation Predictor&#10;GraphSAGE / GAT&#10;(Member 3)"]
        MC["Module C&#10;SAT/SMT Verifier &amp; Bounded Optimizer&#10;Z3 / CVC5&#10;(Member 4)"]
        MA -->|reordered / commuted gates| MB
        MB -->|ranked cancellation candidates| MC
        MC -.->|rejected rewrite: revert| MA
    end

    subgraph S5["Stage 5 — Module D"]
        EA["Evolutionary Multi-Objective Search&#10;NSGA-II (pymoo / DEAP)&#10;(Member 5)"]
        PF["Pareto front of circuit variants"]
        EA --> PF
    end

    subgraph S6["Stage 6 — Evaluation &amp; Benchmarking Engine  (Member 5)"]
        EV["5 metrics vs Qiskit O0-O3&#10;gate count | depth | exec time | fidelity | runtime cost"]
    end

    OUT["Output&#10;Pareto-optimal circuit variants + benchmark report"]

    IN --> P --> DAG
    P --> HG
    DAG --> PART
    HG --> PART
    PART -->|blocks| MA
    MC -->|verified candidate circuits| EA
    PF --> EV
    EV --> OUT
    EV -->|reward signal (closed loop)| MA

    classDef stage fill:#eef4ff,stroke:#3366cc,stroke-width:1px;
    classDef io fill:#f5f5f5,stroke:#888,stroke-width:1px;
    class IN,OUT io;
```

> A static, presentation‑ready vector version of the same diagram is in
> [`architecture-diagram.svg`](architecture-diagram.svg).

---

## 3. Stage descriptions

### Stage 1 — Front End: Parser & IR Builder
Parses the input circuit (OpenQASM 2.0, Qiskit `QuantumCircuit`, or Cirq) into a
compact internal **intermediate representation (IR)**: an ordered gate list with
qubit operands, parameters, and metadata slots for scheduling/verification
annotations. The IR is the single source of truth that every later stage reads
and rewrites.

### Stage 2 — Graph Construction
Two graphs are built from the IR:

* **Gate‑dependency DAG** — one node per gate; a directed edge `g1 -> g2` when
  `g2` uses a qubit last touched by `g1`. Drives scheduling, commutation, and
  critical‑path analysis.
* **Qubit‑interaction hypergraph** — qubits are vertices; each multi‑qubit gate
  (or a time window of them) is a hyperedge. Drives structural partitioning.

### Stage 3 — Partitioning & Block Scheduling
Multilevel hypergraph/graph partitioning (KaHyPar / METIS) cuts a large circuit
into smaller, weakly‑coupled **blocks**. This bounds the search space handed to
each optimization module and lets large circuits be processed in parallel. A
block scheduler orders blocks and resolves cross‑block dependencies.

### Stage 4 — Cooperating Optimization Modules (per block)
Runs on each block; iterates until no module proposes an accepted change.

* **Module A — RL Adaptive Gate Scheduler.** A PPO/DDQN agent observes the block
  DAG and applies reorder / commute / local‑rewrite actions to shorten the
  critical path and expose cancellation opportunities. Reward = weighted
  (depth ↓, gate count ↓, estimated fidelity ↑).
* **Module B — GNN Cancellation Predictor.** A GraphSAGE/GAT network embeds each
  gate node from local structural + commutation‑relation features and scores
  candidate gate pairs/chains for cancellation, replacing exhaustive pairwise
  search. Emits a ranked candidate list.
* **Module C — SAT/SMT Verifier & Bounded Optimizer.** Encodes unitary
  equivalence in SMT (Z3 / CVC5). *Every* rewrite proposed by A or B is checked;
  rejected rewrites are reverted. Where a window is small enough, C also performs
  exact minimal re‑synthesis.

### Stage 5 — Module D: Evolutionary Multi‑Objective Search
NSGA‑II operates over the **population of verified candidate circuits** produced
by Stage 4 (across blocks and iterations). Crossover/mutation act on gate
sequences and scheduling choices; fitness uses structure‑aware fidelity
surrogates instead of full unitary simulation, so it scales past the ~12‑qubit
exact‑evaluation ceiling. Output: a **Pareto front** trading off gate count,
depth, and fidelity.

### Stage 6 — Evaluation & Benchmarking Engine
Computes the five agreed evaluation parameters for every Pareto point and for the
Qiskit O0–O3 baselines on the benchmark suite. Produces comparison tables/plots
**and** returns a scalarized score as the RL reward signal — this is the
closed‑loop edge `EV --> MA`.

---

## 4. Data contracts between stages

| Edge | Payload |
|------|---------|
| `IN -> P` | Circuit source (`.qasm` text / `QuantumCircuit` / `cirq.Circuit`) |
| `P -> DAG, HG` | Internal IR (`qco.ir.IntermediateRepresentation`) |
| `DAG, HG -> PART` | `networkx.DiGraph` + hypergraph adjacency |
| `PART -> MA` | List of `Block` objects (sub‑IR + boundary qubit map) |
| `MA -> MB` | Rewritten block IR + list of applied commutations |
| `MB -> MC` | Ranked `CancellationCandidate` list |
| `MC -> EA` | Verified candidate circuits (IR) + equivalence proof status |
| `EA -> EV` | Pareto front: `list[CandidateCircuit]` with objective vector |
| `EV -> OUT` | Pareto variants + `BenchmarkReport` |
| `EV -> MA` | `float` reward (scalarized 5‑metric score) |

---

## 5. Module ownership

| Module / Stage | Owner |
|----------------|-------|
| Stage 1–3: IR, DAG/Hypergraph, Partitioning & block scheduling | Member 1 |
| Stage 4 · Module A: RL adaptive gate scheduler | Member 2 |
| Stage 4 · Module B: GNN cancellation predictor | Member 3 |
| Stage 4 · Module C: SAT/SMT verifier & bounded optimizer | Member 4 |
| Stage 5 · Module D: NSGA‑II search **+** Stage 6 evaluation engine **+** integration | Member 5 |

---

## 6. Mapping to the codebase

```
src/qco/
├── ir/          -> Stage 1  (parser.py, intermediate_representation.py)
├── graphs/      -> Stage 2 + 3  (dag.py, hypergraph.py, partitioning.py)
├── modules/
│   ├── rl_scheduler/     -> Stage 4 Module A
│   ├── gnn_cancellation/ -> Stage 4 Module B
│   ├── smt_verifier/     -> Stage 4 Module C
│   └── evolutionary/     -> Stage 5 Module D
├── evaluation/  -> Stage 6  (metrics.py, benchmark_runner.py)
└── pipeline.py  -> wires Stage 1..6 + closed-loop feedback
```
