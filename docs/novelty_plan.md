# Novelty Plan — what changes, and who changes it

**Project:** Quantum Compiler Optimization — Gate Scheduling & Gate Cancellation
**Team 2** · Maanas Nair (M2) · Prisha (M3) · Sahib Singh (M4) · Swaraj Rane (M5)

## Why

Every module on its own has close prior art: Quarl and IBM's RL transpiler
(RL rewriting), MQT Predictor (adaptive pass selection), BQSKit (partition +
block re‑synthesis), GUOQ / Quartz / Queso (rewriting + re‑synthesis),
VOQC / Giallar / MQT QCEC (verified optimization). "Combining all four" alone is
an integration contribution, not a research one. The additions below give the
project a defensible claim:

> **Verifier‑guided learned optimization:** a GNN trained on SMT‑verified
> re‑synthesis outcomes allocates a bounded exact‑solver budget, scalable
> phase‑polynomial SMT checks verify every rewrite, and an
> equivalence‑preserving block‑level NSGA‑II produces a certified Pareto front —
> closed‑loop with a calibration‑aware RL reward.

| ID | Addition | Novelty claim |
|----|----------|---------------|
| N1 | GNN ranks windows for exact re‑synthesis under a solver‑time budget | Same reduction as exhaustive re‑synthesis (BQSKit) at a fraction of solver time; beats random window choice (GUOQ) |
| N2 | Phase‑polynomial SMT equivalence for {CX, Rz, X} windows | Real SMT verification that scales past the 8‑qubit dense‑unitary ceiling |
| N3 | Block‑level crossover over verified block variants | Pareto front is equivalent **by construction** — no post‑hoc filtering |
| N4 | Closed loop: trained RL agent, reward under different calibration profiles | Learned schedules measurably differ and win per backend |
| N5 | GNN‑confidence‑gated verification | Lower verification cost at equal correctness |

## Who changes their code

Status checked against `main` @ `f85044f` and Sahib's open branch
`sprint2/smt-verifier-z3-encoding`.

| Member | Needs to change code? | For | Existing code modified | Stubs to fill / new code |
|--------|:---:|-----|------------------------|--------------------------|
| **Sahib Singh** (M4) | **Yes — most critical** | N1, N2 | `smt_verifier/equivalence.py` `check()` — route {CX, Rz, X} windows to the phase‑polynomial checker; return an explicit *unverified* status instead of the pipeline skipping > 8 qubits | `bounded_optimizer.py` `_solve` (the N1 label source); new `smt_verifier/phase_poly.py` |
| **Prisha** (M3) | **Yes** | N1, N5 | `gnn_cancellation/dataset.py` — labels come from M4's verified re‑synthesis outcomes instead of only `rule_based_candidates` (already planned for Sprint 3‑4) | New window‑level (pooled) scoring head in `model.py`; score calibration in `eval.py` |
| **Maanas Nair** (M2) | **Yes, small** | N4 | `rl_scheduler/agent.py` `schedule()` — use the trained policy, keep the greedy pass as fallback | `agent.py` `train` / `load` (PPO, e.g. `MaskablePPO`). Environment reward already uses `scalarized_reward(..., calibration=...)` — **no change needed there** |
| **Swaraj Rane** (M5) | **Yes — largest** | N1–N5, baselines | `modules/evolutionary/nsga2.py` — genome becomes a vector of block‑variant indices; rewrite `_GateSequenceCrossover` / `_GateSequenceMutation` (≈ lines 302–340); `evolve()` gains an optional `block_variants` argument. `pipeline.py` — window ranking + solver budget (N1), drop the `num_qubits <= 8` guards (lines 70, 93, 107) (N2), collect per‑block variants (N3), feed `reward` (line 98) back and add `calibration` to `PipelineConfig` (N4), confidence gating (N5), fix `blocks=len(front)` (line 111). Update NSGA‑II tests in `tests/test_modules_and_pipeline.py` | TKET / PyZX / BQSKit baselines + `solver_time` column in `evaluation/benchmark_runner.py`; ablation runs |

**No change needed for novelty:** `qco.ir`, `qco.graphs` (Swaraj, ex‑M1 —
KaHyPar/METIS remains normal sprint work, not novelty work),
`evaluation/metrics.py`, `evaluation/calibration.py`, `evaluation/noisy_sim.py`,
the benchmark generator, and Maanas's `environment.py` / `rewrites.py` /
`spaces.py`.

**Public interfaces stay stable:** `EquivalenceChecker.check`,
`RLScheduler.schedule`, `GNNCancellationPredictor.predict`,
`NSGA2Compiler.evolve` (new argument is optional) and `optimize()`.

## Order (dependencies)

```
M4  _solve + phase_poly ──┬──> M3 verified labels + window head ──> M5 pipeline N1/N5
                          └──> M5 pipeline N2 (drop <=8 guards)
M5  N3 block crossover        (independent — can start now)
M2  PPO train/load ──────────> M5 pipeline N4 (feed reward back)
M5  baselines + ablations     (after the above, Sprint 5)
```

Start now with no waiting: **M5 N3**, **M4 `_solve` + `phase_poly.py`**,
**M2 PPO training**. M3 starts once M4 can produce re‑synthesis labels.
