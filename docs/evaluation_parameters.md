# Evaluation Parameters

Measured for every circuit in the benchmark suite and compared against the
Qiskit O0–O3 baselines.

| Parameter | Definition | How it is measured |
|-----------|-----------|--------------------|
| **Gate Count Reduction (%)** | % decrease in total gate count after optimization vs. the original circuit | `(Gates_before − Gates_after) / Gates_before × 100` |
| **Circuit Depth Reduction (%)** | % decrease in circuit depth (critical‑path length) | `(Depth_before − Depth_after) / Depth_before × 100` |
| **Execution Time (ns / µs)** | Estimated hardware execution time of the compiled circuit | Sum of per‑gate durations along the critical path, using device gate‑time calibration data |
| **Estimated Fidelity / Error Reduction** | Improvement in expected output fidelity / reduction in accumulated gate error | Simulator‑based fidelity estimate (process fidelity, noise‑model simulation) before vs. after |
| **Runtime Complexity** | Computational cost of the optimization pipeline itself, as a function of circuit size | Empirical wall‑clock scaling + theoretical Big‑O characterization per module |

These five parameters are implemented as pure functions in
[`src/qco/evaluation/metrics.py`](../src/qco/evaluation/metrics.py) and driven
across the suite by
[`src/qco/evaluation/benchmark_runner.py`](../src/qco/evaluation/benchmark_runner.py).
The benchmark runner also returns a scalarized score used as the RL reward signal
(closed‑loop edge in the architecture).
