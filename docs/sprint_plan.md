# Sprint Plan — 6 Sprints

**Project:** Quantum Compiler Optimization — Gate Scheduling & Gate Cancellation
**Team 2**

Six sprints of ~2 weeks each, covering the work after the Review 1 checkpoint
(Phases 3–6 of [`project_timeline.md`](project_timeline.md)). The plan is built so
that **cross-person dependencies land as late as possible**: Sprints 1–2 are fully
parallel (everyone codes against the frozen IR), the first hand-off is Sprint 3,
and the main integration is Sprint 4. Each member carries roughly one chunky
deliverable plus supporting work per sprint.

## Team & GitHub handles

| | Member | GitHub | Module / area | Packages |
|:--:|--------|--------|---------------|----------|
| M1 | Ajay M | [@Ajay-1011-git](https://github.com/Ajay-1011-git) | Front-End & Graph Infrastructure · lit-review coordinator | `qco.ir`, `qco.graphs` |
| M2 | Maanas Nair | [@itsmebirdie](https://github.com/itsmebirdie) | Module A — RL Adaptive Gate Scheduler | `qco.modules.rl_scheduler` |
| M3 | Prisha | [@PrishaNagpal](https://github.com/PrishaNagpal) | Module B — GNN Gate-Cancellation Predictor | `qco.modules.gnn_cancellation` |
| M4 | Sahib Singh | [@sahib-1030](https://github.com/sahib-1030) | Module C — SAT/SMT Optimizer & Verifier | `qco.modules.smt_verifier` |
| M5 | Swaraj Rane | [@ArmouredOre](https://github.com/ArmouredOre) | Module D — NSGA-II · integration · evaluation · repo/CI | `qco.modules.evolutionary`, `qco.evaluation`, `qco.pipeline` |

## Checkpoints

| After | Checkpoint |
|-------|-----------|
| Sprint 3 | **Review 2** — independent working prototypes of all four modules |
| Sprint 5 | Full pipeline + evaluation run complete |
| Sprint 6 | **Review 3** — final report & demo |

---

## Sprint grid

### Sprint 1 — Foundations *(fully parallel — only the frozen IR is shared)*

| Owner | Deliverable |
|-------|-------------|
| **Ajay M** (M1) | Real front end: Qiskit / Cirq import, full `qelib1` gate set, parameter binding; DAG builder hardening + commutation-relation table |
| **Maanas Nair** (M2) | RL environment: `reset` / `step` / reward fully implemented over the gate DAG; exercised with a random policy |
| **Prisha** (M3) | Finalize gate/node feature encoding + PyTorch-Geometric `Data` builder; expand the deterministic rule-based candidate generator |
| **Sahib Singh** (M4) | Harden the dependency-free numeric equivalence checker (full gate set, compare up to global phase) + property tests |
| **Swaraj Rane** (M5) | Finalize the five evaluation metrics + device gate-time / error-rate calibration tables; benchmark-runner skeleton (no baseline yet) |

**Dependencies:** none. Everyone builds against the already-frozen `Gate` /
`IntermediateRepresentation` / `Block` dataclasses.

### Sprint 2 — Core algorithms *(still parallel)*

| Owner | Deliverable |
|-------|-------------|
| **Ajay M** (M1) | Qubit-interaction hypergraph builder + windowing; KaHyPar / METIS multilevel partitioner replacing the fallback |
| **Maanas Nair** (M2) | Deterministic Commutation-and-Reduction (CR) pass; PPO agent + training loop (trained against a local proxy reward) |
| **Prisha** (M3) | GraphSAGE / GAT model + training loop, trained on rule-based labels |
| **Sahib Singh** (M4) | SMT equivalence encoding in Z3 / CVC5 (unitary equality up to global phase); cross-checked against the numeric checker |
| **Swaraj Rane** (M5) | NSGA-II genetic operators via pymoo (crossover / mutation over gate sequences); structure-aware fidelity surrogate v1 |

**Dependencies:** only M1-internal (partitioner consumes M1's own hypergraph).

### Sprint 3 — First hand-offs *(→ Review 2)*

| Owner | Deliverable |
|-------|-------------|
| **Ajay M** (M1) | Block scheduler + cross-block dependency stitching; ships real `Block`s to M2 / M3 |
| **Maanas Nair** (M2) | Train RL on 5-qubit circuits; measure generalization to 10 / 20 qubits; integrate the CR pass |
| **Prisha** (M3) | Precision / recall evaluation + threshold calibration; **regenerate training labels using M4's checker** |
| **Sahib Singh** (M4) | Counterexample extraction + revert protocol; **expose the verified-label API to M3** |
| **Swaraj Rane** (M5) | Qiskit O0–O3 baseline harness in the benchmark runner; NSGA-II end-to-end on seed circuits |

**Dependencies introduced:** M3 ← M4 (verified labels); M2 & M3 start consuming
M1's `Block`s. One hop each — deliberately deferred to here.

### Sprint 4 — Stage-4 loop assembly *(main convergence)*

| Owner | Deliverable |
|-------|-------------|
| **Ajay M** (M1) | Partitioner tuning on large circuits; parallel block processing; support M5 integration |
| **Maanas Nair** (M2) | Wire the closed-loop reward from M5's Stage 6 scorer; retrain with the real reward |
| **Prisha** (M3) | Integrate ranked candidates into the pipeline apply-step; retrain on M4-verified labels |
| **Sahib Singh** (M4) | Bounded-window exact re-synthesis encoding; solver timeouts + result caching |
| **Swaraj Rane** (M5) | **Integrate Stages 1–4** into `HybridOptimizer` (blocks → A → B → C-verify loop); expose the reward hook to M2 |

**Dependencies:** first big convergence — M5 pulls A + B + C + partitioner
together. Everything before this was independently testable.

### Sprint 5 — Full pipeline + evaluation

| Owner | Deliverable |
|-------|-------------|
| **Ajay M** (M1) | Graph-layer profiling; Big-O write-up for the partitioner |
| **Maanas Nair** (M2) | RL ablations (CR on/off, reward weights); generalization to 40 qubits |
| **Prisha** (M3) | Final GNN evaluation; robustness under device noise; Big-O write-up for Module B |
| **Sahib Singh** (M4) | SMT scalability tests; bounded re-synthesis on real windows; Big-O write-up for Module C |
| **Swaraj Rane** (M5) | **Wire Stage 5 (NSGA-II) onto verified candidates + Stage 6 closed loop**; full benchmark run vs Qiskit O0–O3 on all five metrics; Pareto-front generation |

**Dependencies:** M5 needs A / B / C stable from Sprint 4; everyone else does
hardening + complexity analysis in parallel (non-blocking).

### Sprint 6 — Benchmarking, analysis, report *(→ Review 3)*

| Owner | Deliverable |
|-------|-------------|
| **Ajay M** (M1) | Final graph-layer bug-fixes; write the IR / partitioner section of the report |
| **Maanas Nair** (M2) | Write the RL section + generalization plots |
| **Prisha** (M3) | Write the GNN section + precision / recall plots |
| **Sahib Singh** (M4) | Write the SMT section + optimality / scaling plots |
| **Swaraj Rane** (M5) | Consolidate results; Pareto-front analysis; O0–O3 comparison plots; final report + demo + presentation |

**Dependencies:** report sections written in parallel; M5 assembles.

---

## Dependency timeline (why the ordering)

```
Sprint:        1        2        3        4        5        6
M1 Ajay     ── parser ─ hypergr ─ blocks ─┐ tune ── profile ─ report
                                          │
M2 Maanas   ── RL env ─ PPO ──── train ───┤ reward── ablation─ report
                                          │  ▲
M3 Prisha   ── feats ── model ── P/R ─────┤  │ apply ─ final ── report
                              ▲           │  │
M4 Sahib    ── numchk ─ SMT ── labels ────┤  │ re-synth─ scale─ report
                              │           │  │
M5 Swaraj   ── metrics─ NSGA-II─ baseline ─┴──┴ INTEGRATE─ pipeline+eval─ report
                                          (Stages 1-4)    (Stage 5-6 + loop)

first cross-person edge:  Sprint 3  (M3 ← M4 labels;  M2/M3 ← M1 blocks)
main convergence:         Sprint 4  (M5 integrates A+B+C+partitioner)
closed loop online:       Sprint 5  (M5 wires Stage 6 → M2 reward)
```

- **Sprints 1–2:** no one waits on anyone.
- **Sprint 3:** M3 ← M4 (verified labels); M2 / M3 consume M1's `Block`s.
- **Sprint 4:** M5 integrates A + B + C + partitioner.
- **Sprint 5:** M5 adds Module D + the closed loop; M1–M4 do analysis in parallel.
- **Sprint 6:** parallel report sections, assembled by M5.

## Load balance

Each member carries **one chunky deliverable + supporting work per sprint** —
roughly equal. The one imbalance is **M5 (Swaraj): Module D *plus* integration,
evaluation and the report.** That is deliberate — it is why Module D (NSGA-II) is
the smallest of the four algorithm modules. If it runs hot, the natural rebalance
is to move M5's Sprint-3 benchmark-runner + baseline-harness to whoever finishes
their module first, since it only needs the IR.
