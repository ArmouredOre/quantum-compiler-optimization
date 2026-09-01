# Phase 1 — Literature Survey

**Team 2:** Ajay M (24BCE2493) · Maanas Nair (24BCT0157) · Prisha (24BDS0146) · Sahib Singh (24BCI0292) · Swaraj Rane (24BCT0086)
Lit‑review compilation coordinated by Ajay M; each member surveyed their own module (see [`work_distribution.md`](work_distribution.md)).

Consolidated from the Review 1 report. Organized around the four project
sub‑topics plus the classical rule‑based baseline the system is benchmarked
against.

---

## 1. Introduction

Quantum circuits produced by naive compilation of high‑level algorithms carry
excess gates, redundant adjacent/commuting operations, and unnecessarily long
critical paths. On NISQ hardware every extra gate and every extra layer of depth
increases exposure to decoherence and gate error, so compiler‑level optimization
is essential for usable fidelity.

This project targets two linked stages of the compilation pipeline:

* **Gate scheduling** — ordering/placing gates, commuting them to shorten the
  critical path and expose parallelism.
* **Gate cancellation** — identifying pairs/chains of gates that combine to the
  identity or to a simpler equivalent, and removing them.

The proposed approach is a hybrid pipeline combining learning‑based heuristics
(RL, GNN), formal methods (SAT/SMT), and evolutionary multi‑objective search, to
outperform fixed rule‑based passes such as Qiskit's transpiler.

## 2. Motivation

* NISQ devices are error‑ and decoherence‑limited; shorter, shallower circuits
  raise success probability directly.
* Rule‑based passes (Qiskit `CommutativeCancellation`, `InverseCancellation`) are
  fast but fixed; benchmarking shows the best pass combination differs by circuit
  and backend, so a single fixed pipeline is provably sub‑optimal.
* Learning methods (RL, GNN) adapt to circuit structure but are usually evaluated
  without formal correctness guarantees.
* Exact methods (SAT/SMT) guarantee optimality/correctness but scale poorly
  (double‑exponential in qubit count for full synthesis) — practical only on
  bounded local windows.
* Objectives (gate count, depth, fidelity, optimizer runtime) conflict,
  motivating a Pareto/multi‑objective view.

---

## 3. Literature review

### 3.1 Reinforcement Learning‑based adaptive gate scheduler

RL treats scheduling/optimization as a sequential decision process: an agent
observes the circuit (often as a graph) and takes reorder/commute/rewrite
actions, rewarded on depth, gate count, or fidelity. Recurring finding: pairing
RL with a deterministic reduction rule set improves performance and
generalization from small training circuits to much larger evaluation circuits.

| Study | Method | Key result | Limitation / gap |
|-------|--------|-----------|------------------|
| Tao et al., 2026 (arXiv:2608.19103) | RL + deterministic Commutation‑and‑Reduction (CR) hybrid; PPO‑style agent | Generalizes 5‑qubit → 20‑qubit; removes ~2× more gates than plain RL on Clifford+T and CNOT+Pauli circuits | Still needs the deterministic pass to be effective; scalability beyond 20 qubits unproven |
| Riu et al., 2025 (Quantum) | PPO agent with GNN policy/value network over ZX‑calculus diagrams | Trained on 5 qubits, generalizes to 80 qubits / ~2100 gates; ~10% extra gate reduction over best hand‑coded ZX rules | Relies on ZX rewrite rule set; per‑hardware reward tuning |
| Rieckmann, Scheel & Plato, 2025 (arXiv:2511.08096) | RL for entangling‑gate sequence optimization in parameterized circuits | Lower CNOT count under connectivity constraints; higher state‑prep fidelity at equal CNOT budget | Focused on state preparation, not general scheduling |
| Baseline cited in Anon., 2025 (arXiv:2508.21253) | Deep RL optimizer trained on 12‑qubit random entangled circuits | Avg. 27% depth reduction, 15% CNOT reduction | Entanglement not explicitly optimized during training |

**Takeaway:** hybrid RL + deterministic rules generalizes far better than RL alone.

### 3.2 Graph Neural Network for gate‑cancellation prediction

A circuit is naturally a graph/DAG of gates connected by qubit‑wire
dependencies, so GNNs fit without a fixed‑size input (unlike CNNs). Existing work
mostly uses GNNs to predict downstream circuit properties; this project adapts
the same graph‑embedding approach to score cancellation opportunities directly.

| Study | Method | Key result | Limitation / gap |
|-------|--------|-----------|------------------|
| Wang et al., 2025 (arXiv:2504.00464) | GNN over circuit graph to predict output/expectation values | Adapts to variable depth/qubit count; high R² even under device noise | Predicts outputs, not cancellation decisions |
| Tudisco et al., 2025 (arXiv:2507.19093) | GNN on DAG to predict best‑fit hardware target | 94.4% accuracy / 85.5% F1 across 498 circuits on 4 real devices (MQT Bench) | Predicts device choice, not intra‑circuit cancellation |
| Tudisco et al., 2026 (RC, LNCS) | GNN on DAG to predict probability of successful trial (PST) | Avoids brute‑force compile‑and‑execute search | Dataset limited by exponential simulation‑labelling cost |
| He et al., 2023 (Quantum Inf. Process.) | GNN‑based predictor for quantum architecture search | Speeds up structure search via learned graph embeddings | Search‑assistive only; not applied to cancellation |

**Takeaway:** GNNs are proven on circuit‑property prediction; this project repurposes them to rank cancellation candidates.

### 3.3 SAT/SMT‑based global gate optimization

SAT/SMT solvers encode circuit equivalence and gate‑count minimization as
satisfiability, yielding either an optimal circuit for a gate budget or a proof
none is smaller. Strong guarantees, but the search space grows extremely fast, so
in practice these run on small circuits, sub‑circuits, or gate‑set
translation/adaptation.

| Study | Method | Key result | Limitation / gap |
|-------|--------|-----------|------------------|
| Meuli et al. (EPFL), SAT‑based {CNOT,T} synthesis | Exact SAT rewriting minimizing CNOT count without raising T‑count | Avg. 26.84% CNOT reduction across representative 5‑input functions | Exact SAT search does not scale to large circuits |
| Gouzien & Sangouard, 2025 (arXiv:2503.15452) | SAT encoding of unitary‑matrix equality for provably optimal exact synthesis | Finds minimum‑gate circuit or proves none exists for a given count | Time‑to‑solution scales double‑exponentially in qubit count |
| Guo et al., 2026 (eprint 2026/1815) | Novel SMT encodings (exact‑G, at‑most‑G with null gates) | Up to ~53× speed‑up over prior SAT models on S‑box synthesis benchmarks | Shown mainly on cryptographic S‑box permutations |
| Peham et al., 2023 (arXiv:2301.11725) | SMT model with fidelity + idle‑time objectives for hardware adaptation | Improves circuit and Hellinger fidelity vs. KAK / template optimization | Evaluated on small circuits (≤4 qubits, depth ≤160) |

**Takeaway:** SAT/SMT gives provable optimality/correctness but only on small or bounded sub‑circuits.

### 3.4 Evolutionary multi‑objective quantum compiler

Genetic/evolutionary algorithms — most commonly NSGA‑II — evolve a population of
candidate circuits toward a Pareto front across conflicting objectives (fidelity,
depth, gate / two‑qubit‑gate count). Recent focus: scaling past the ~12‑qubit
ceiling of exact unitary‑based fidelity evaluation using structure‑aware
surrogates.

| Study | Method | Key result | Limitation / gap |
|-------|--------|-----------|------------------|
| Rasconi & Oddi, 2019 | Genetic algorithm for circuit compilation/scheduling | GA scheduling shown viable vs. heuristic compilers | Single/limited objective; smaller benchmark scale |
| Potoček et al., 2018 / Altares‑López et al., 2021 | Multi‑objective GA for circuit construction / feature maps | Automatic circuit design balancing multiple structural objectives | Search cost grows with circuit/feature‑map complexity |
| Anon., 2026 (Scientific Reports) | Scalable NSGA‑II optimizer using fidelity surrogates (no full unitary) | Joint Pareto optimization of fidelity, depth, gate cost beyond 12 qubits | Novelty is the surrogate model, not the NSGA‑II search |
| Rasmussen et al., 2019 (arXiv:1812.04458) | Multi‑objective evolutionary search "from scratch" for circuit discovery | Rediscovers QFT, Grover; trades accuracy vs. depth/gate count/width | Aimed at discovery, not optimizing a given input circuit |

**Takeaway:** NSGA‑II‑style search yields a Pareto front of circuits instead of one fixed answer.

### 3.5 Baseline — classical rule‑based compiler passes

Production compilers (IBM Qiskit transpiler) apply a fixed pass sequence
(inverse‑gate cancellation, commutation‑based cancellation, two‑qubit block
consolidation + resynthesis) at levels O0–O3. These are the project baseline;
recent benchmarking quantifies how much benefit comes from cancellation alone vs.
the full pipeline, and shows the best configuration is circuit‑ and
backend‑dependent.

| Study | Method | Key result | Limitation / gap |
|-------|--------|-----------|------------------|
| IBM Qiskit Transpiler O0–O3 (docs, 2021/2026) | Rule‑based `InverseCancellation`, `CommutativeCancellation`, `Collect2qBlocks`/`ConsolidateBlocks` | O2/O3 apply commutation‑aware cancellation + block resynthesis; industry‑standard baseline | Fixed pass pipeline per level, not adaptive |
| Zulehner et al., 2021 (arXiv:2012.07711) | Relaxed peephole optimization for quantum circuits | Improves on rigid peephole rules used inside compilers | Still local, rule‑based (non‑learned) |
| Anon., 2026 (arXiv:2601.20871) | End‑to‑end fidelity benchmarking of individual Qiskit passes vs. combined pipeline | Cancellation pass alone: ~9.1% gate reduction (largest single‑pass effect); O2/O3: 47.7% gate / 9.3% two‑qubit / 37.9% depth reduction on the corpus | Cancellation dominates but plateaus without adaptive/global search |
| Anon., 2026, TuniQ (arXiv:2605.11375) | Autotuning of compiler pass selection per circuit/backend | Optimal pass combination varies by circuit structure and backend; fixed pipelines underperform selective ones | Brute‑force search over pass combinations is expensive — motivates learned selection |

---

## 4. References

1. K. D. Tao et al., "Quantum circuit optimization using deep reinforcement learning: Applications across multiple gate sets," arXiv:2608.19103, 2026.
2. J. Riu, J. Nogué, G. Vilaplana, A. Garcia‑Saez, M. P. Estarellas, "Reinforcement Learning Based Quantum Circuit Optimization via ZX‑Calculus," Quantum, 2025.
3. T. R. Rieckmann, S. Scheel, A. D. K. Plato, "Gate Sequence Optimization for Parameterized Quantum Circuits using Reinforcement Learning," arXiv:2511.08096, 2025.
4. Anon., "Reinforcement Learning for Optimizing Large Qubit Array based Quantum Sensor Circuits," arXiv:2508.21253, 2025.
5. H. Wang et al., "Output Prediction of Quantum Circuits based on Graph Neural Networks," arXiv:2504.00464, 2025.
6. A. Tudisco, D. Volpe, G. Orlandi, G. Turvani, "Graph Neural Network‑Based Predictor for Optimal Quantum Hardware Selection," arXiv:2507.19093, 2025.
7. A. Tudisco, D. Volpe, M. Graziano, G. Turvani, "Toward Quantum Circuit Execution Success Estimation via Graph Neural Network‑Based Prediction," Reversible Computation (RC 2026), LNCS 16626, Springer, 2026.
8. Z. He, X. Zhang, C. Chen, Z. Huang, Y. Zhou, H. Situ, "A GNN‑based predictor for quantum architecture search," Quantum Information Processing, 22(2):128, 2023.
9. G. Meuli et al., "SAT‑based {CNOT, T} quantum circuit synthesis," EPFL / RC, 2018.
10. É. Gouzien, N. Sangouard, "Provably optimal exact gate synthesis from a discrete gate set," arXiv:2503.15452, 2025.
11. Y. Guo et al., "Novel SMT Encoding for Quantum Circuit Optimization," IACR eprint 2026/1815, 2026.
12. L. Peham et al., "SAT‑Based Quantum Circuit Adaptation," arXiv:2301.11725, 2023.
13. R. Rasconi, A. Oddi, "An Innovative Genetic Algorithm for the Quantum Circuit Compilation Problem," AAAI, 2019.
14. V. Potoček et al., "Multi‑Objective Genetic Algorithm for Circuit Construction," 2018 (cited in arXiv:2504.17561).
15. Anon., "Scalable multi‑objective genetic algorithm for quantum circuit optimization," Scientific Reports, 2026.
16. L. Rasmussen, T. Ryan‑Anderson et al., "Multi‑objective evolutionary algorithms for quantum circuit discovery," arXiv:1812.04458, 2019.
17. IBM Quantum, "Qiskit Transpiler documentation (v1.4 / v2.0)," quantum.cloud.ibm.com, 2026.
18. A. Zulehner et al., "Relaxed Peephole Optimization: A Novel Compiler Optimization for Quantum Circuits," arXiv:2012.07711, 2021.
19. Anon., "End‑to‑End Fidelity Analysis of Quantum Circuit Optimization," arXiv:2601.20871, 2026.
20. Anon., "TuniQ: Autotuning Compilation Passes for Quantum Workloads at Scale," arXiv:2605.11375, 2026.
21. J. Andrés‑Martínez, C. Heunen, "Distributing circuits over heterogeneous, modular quantum computing network architectures," arXiv:2305.14148, 2023.
22. G. Karypis, V. Kumar, "Multilevel k‑way hypergraph partitioning," Proc. DAC '99, 1999.
