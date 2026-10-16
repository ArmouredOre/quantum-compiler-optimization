# Final evaluation results (Review 3)

Generated on 2026-10-16 by one command, from `main` at `234d3bb` plus this PR's
evaluation fixes (correctness column, baseline permutation fix):

```
make eval        # = python -m qco.evaluation.benchmark_runner --out results
```

- **Models:** Module A `models/rl_ppo_5q.json` (trained PPO), Module B
  `checkpoints/module_b.pt` (trained GNN with window head). Both load by default.
- **Settings:** calibration `default`, solver budget 2 s per circuit, 26 benchmark circuits.
- **Tools:** Python 3.12; qiskit 2.5.2, pytket 2.18.5, pyzx 0.10.7, bqskit 1.2.1, pymoo 0.6.2, torch 2.14.1.
- **Run time:** 35 min 46 s on a laptop (most of it the unbudgeted "exhaustive" window mode).

| File | Contents |
|---|---|
| [`summary.md`](summary.md) | Every table: qco vs baselines (+ per circuit), Pareto front sizes, ablations, N1 window selection, N5 gating |
| `<optimizer>.md` / `.csv` | Five-metric per-circuit tables for each optimizer and ablation |
| [`pareto/`](pareto/) | Per-circuit Pareto CSV + plot (qco front vs Qiskit O3, TKET, PyZX, BQSKit) |
| [`acceptance_36.log`](acceptance_36.log) | #36 acceptance: all 26 circuits with trained A + B and Module C, verified + statevector check |

**Correctness.** Every output in every table was checked equivalent to its input
(Module C's checker; otherwise a Qiskit statevector check on two random states up to
22 qubits). All optimizers are 25/25 (`ghz_n40` is too wide to simulate) except the
"minus C" ablation, which turns verification off: 9 of its outputs are wrong.

**Correction to earlier baseline numbers (#66).** The first version of these
tables overstated Qiskit O2/O3 and TKET: their wrappers dropped the qubit permutation
those compilers use to elide SWAPs, so 4 Qiskit outputs (all QFT) and 10 TKET outputs
were not equivalent to their inputs. The wrappers now restore the input order with
explicit SWAPs. Qiskit O2 went from 20.0 / 16.6 / 11.7 % (gates / depth / CX-cost)
to 18.9 / 16.0 / 7.4 %.
