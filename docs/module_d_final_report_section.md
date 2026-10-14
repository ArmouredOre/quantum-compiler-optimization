# Module D Final Report: Evolutionary Multi-Objective Search (NSGA-II) & Certified Pareto Fronts

**Project:** Quantum Compiler Optimization — Gate Scheduling & Gate Cancellation
**Team 2 · Review 3 Deliverable (Final Report Contribution)**
**Author / Module Owner:** Swaraj Rane ([@ArmouredOre](https://github.com/ArmouredOre)) · Module D + Stage 6 + integration (M5)
**Covers:** #13 (operators + surrogate), #30 (N3 block-level crossover), #31 (pipeline wiring), #40 (Pareto fronts)

> **Status.** Sections 1–3 and 5 are final. Section 4.3 (Pareto fronts inside
> the full pipeline) is **provisional**: it depends on the Stage-4 modules,
> which are still being finished (#35, #34), and will be regenerated with
> `make eval` once they land.

---

## 1. Module Overview & Role in the Pipeline

Module D is **Stage 5**. Stage 4 hands it, for every block, a set of
*verified* variants (the original block, the RL schedule, Module B's
cancellations, Module C's re-syntheses). Module D searches combinations of
those variants for the trade-off between three objectives (all minimized):

1. **gate count**,
2. **depth**,
3. **infidelity** = 1 − estimated fidelity, from a structure-aware surrogate.

Its output is the **Pareto front** handed to the caller and to Stage 6. The key
property (novelty **N3**) is that every point on the front is equivalent to the
input **by construction**, so no post-hoc filtering or re-verification is needed.

## 2. Design Choices

### 2.1 Search engine

NSGA-II via `pymoo` (population 60, 40 generations by default). Without pymoo
the module degrades to scoring the seeds with a plain non-dominated sort, so
the pipeline still runs on minimal installs (and in CI).

### 2.2 Fidelity surrogate v1 (#13)

Full unitary or noisy simulation does not scale past ~12 qubits, so fitness
uses `structure_aware_fidelity_surrogate`, which is sensitive to circuit
*structure*, not just gate counts:

- per-gate error by arity (1q / 2q / 3q), from the calibration tables;
- **idle decay**: every qubit idle while others run a gate decays as
  exp(−t_idle / T1), with T1 = 50 µs;
- **crosstalk**: a 2-qubit gate sharing its timing layer with another 2-qubit
  gate takes an extra 0.995 factor.

*Validation:* on 13 small circuits (≤ 8 qubits, from near-trivial to lossy) the
surrogate's ranking agrees with a density-matrix simulator with local
depolarizing noise (`qco.evaluation.noisy_sim`) with **Spearman ρ = 0.863** (p = 1.5 × 10⁻⁴),
above the #13 acceptance bar of 0.8
(`tests/test_fidelity_surrogate_validation.py`).

### 2.3 Genome and operators — certified by construction (N3, #30)

The Sprint-2 operators spliced two parents' gate sequences. Splicing two
different circuits is fundamentally not equivalence-preserving, so the pipeline
had to re-verify the front and throw invalid members away.

The N3 redesign changes what a genome *is*:

- **Genome** = a tuple of variant indices, one per block. Every variant of
  block *i* is equivalent to block *i*, and blocks compose in list order — the
  Stage-3 partitioner contract. So **every genome decodes to a circuit
  equivalent to the input**.
- **Crossover**: uniform per block — each block's index comes from either
  parent. Recombination never leaves the equivalence class.
- **Mutation**: pick a block, then either switch it to another existing
  variant, or derive a *new* variant by applying one equivalence-preserving
  rewrite (commute / drop a proven identity / reorder, built on Module B's
  `gates_commute` and `rule_based_candidates`). Variant pools grow during the
  search.
- **Interface**: `evolve(seeds, block_variants=None)`. With `block_variants`,
  the caller supplies the verified variants per block; without it, the first
  seed is partitioned and each block starts as its own only variant (the old
  call signature still works).

One implementation detail mattered: pymoo deep-copies the algorithm by default,
so the mutation operator was growing a *copy* of the pools while the problem
decoded from the originals. `minimize(..., copy_algorithm=False)` fixes it.

### 2.4 Integration in the pipeline (#31)

`HybridOptimizer` collects every verified variant of every block and calls
`evolve([circuit], block_variants=...)`. The old post-hoc filter became an
assertion of the invariant (checked when the circuit is small enough to verify
densely).

## 3. Empirical Results — N3 operators (final)

Hypervolume (HV) of the front after the equivalence filter the pre-N3 pipeline
applied, on the benchmarks of ≤ 6 qubits (population 30, 20 generations).
Objectives are normalized by 1.1 × the input's values, so the input alone scores
(1 − 1/1.1)³ ≈ 0.00075.

| circuit | old operators: equivalent members of 30 | N3: equivalent / front size | N3 HV |
|---|---|---|---|
| ghz_n5 | 0 | 1 / 1 | 0.0008 |
| grover_w3_i1 | 0 | 1 / 1 | 0.0178 |
| grover_w3_i2 | 0 | 1 / 1 | 0.0281 |
| qaoa_n6_p1 | 0 | 1 / 1 | 0.0010 |
| qaoa_n6_p2 | 0 | 1 / 1 | 0.0008 |
| qft_n5 | 0 | 1 / 1 | 0.0011 |
| rand_cliffordt_n5 | 0 | 1 / 1 | 0.0030 |
| rand_cnotpauli_n5 | 0 | 4 / 4 | 0.0050 |
| **total HV** | **0.0062** (input only) | | **0.0575** |

- **With the old operators, not one front member was equivalent to the input
  on any circuit.** The splicing crossover drove the population towards short,
  invalid circuits (down to the empty circuit, which "wins" every objective),
  and the pipeline always fell back to the unoptimized input.
- **With N3, every member is equivalent with no filtering**, and total HV is
  ≈ 9× higher.

*Reproduce:* `python benchmarks/bench_nsga2.py` (N3 numbers; stochastic, so
expect small run-to-run variation — 0.0582 was measured at #55). The old-operator
column was measured on commit `9173b92`, the last `main` before #55.

## 4. Results inside the full pipeline

### 4.1 Correctness (final)

On all 26 benchmarks (up to 40 qubits), every returned front member is
equivalent to its input: verification reports 0 unverified blocks, and an
independent Qiskit statevector check on random input states agrees for every
circuit up to 22 qubits (#31, #36).

### 4.2 Contribution in the ablation (provisional)

Removing Module D (`use_ea=False`, front = best stitched circuit only) changes
the suite-mean depth reduction from 8.1% to 7.7% in the current run. Gate count
and CX-cost are unchanged — the variants are few, so the front rarely holds more
than one or two points. To be re-measured once Modules A and B are final.

### 4.3 Pareto fronts per benchmark (provisional)

`make eval` writes `results/pareto/<circuit>.csv` and `<circuit>.png` (qco front
against Qiskit O3, TKET, PyZX and BQSKit; panels gates vs depth and gates vs
estimated fidelity). Fronts are small today (1–6 points), which reflects the
limited variant pool from unfinished Stage-4 modules rather than the search
itself. Figures for the report will be generated from the final run.

## 5. Architectural Synergy

1. **With Stage 3 (partitioner):** N3 rests entirely on the partitioner's
   composition contract; per-block variants are only interchangeable because
   blocks compose in list order.
2. **With Module C (Sahib):** every variant entering the pool is verified, so
   Module D never verifies anything itself — certification is a property of the
   representation, not a filter.
3. **With Modules A and B (Maanas, Prisha):** their outputs are the raw
   material. Richer, more diverse verified variants give NSGA-II more to
   recombine, so Module D's gains grow as A and B improve.
4. **With Stage 6:** Module D optimizes the same three quantities (gates, depth,
   fidelity) that Stage 6's `scalarized_reward` combines into the closed-loop
   reward for Module A (`reward_hook`, #35) — though the reward uses Stage 6's
   calibrated `estimated_fidelity`, not the Module D surrogate.

## 6. Limitations & Future Work

1. **Recombination only within blocks.** Crossover chooses among per-block
   variants; it cannot discover rewrites that span a block boundary. Larger or
   overlapping blocks would help but trade against verification width.
2. **Small fronts.** With few variants per block, the front often has a single
   point. More diverse Stage-4 proposals (e.g. several RL rollouts) are the
   cheapest way to widen it.
3. **Variant pools are not deduplicated.** They grow up to population ×
   generations entries; dedup by gate tuple if memory becomes an issue.
4. **No coupling map in the surrogate.** Crosstalk is charged for *any* two
   concurrent 2-qubit gates rather than only physically adjacent ones; tighten
   once `qco.graphs` carries a device coupling map.

## 7. Deliverable Summary for Review 3 Assembly

| Item | Location | Status |
|---|---|:---:|
| Module D section | `docs/module_d_final_report_section.md` | Final (4.2, 4.3 provisional) |
| NSGA-II + N3 operators | `src/qco/modules/evolutionary/nsga2.py` | Merged (#55) |
| N3 bench | `benchmarks/bench_nsga2.py` | Added |
| Surrogate validation | `tests/test_fidelity_surrogate_validation.py` | ρ = 0.863 |
| Pareto CSV + plots | `make eval` → `results/pareto/` | Code in #62; final run pending |
