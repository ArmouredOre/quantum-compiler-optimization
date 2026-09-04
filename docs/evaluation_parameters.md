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

## Device calibration profiles

`execution_time` and `estimated_fidelity` (and therefore `scalarized_reward` and
the benchmark runner) take a named **calibration profile** from
[`src/qco/evaluation/calibration.py`](../src/qco/evaluation/calibration.py):

| Profile | Notes |
|---------|-------|
| `default` | Coarse superconducting-style placeholder (the original Phase 2 numbers) |
| `superconducting_ibm_like` | Illustrative fixed-frequency transmon profile: fast gates, moderate 2‑qubit error |
| `trapped_ion_like` | Illustrative trapped-ion profile: microsecond gates, low error rates |

These are order-of-magnitude, hand-picked profiles for exercising the metrics
before real hardware data is available — **not** pulled from a specific live
backend. Real per-backend calibration (via Qiskit's `BackendProperties`) is
wired in Phase 5 without changing these function signatures — pass a profile
name (`run_benchmarks(optimizer, calibration="trapped_ion_like")`) or a custom
`Calibration` instance.

## Benchmark runner output

`BenchmarkReport.to_markdown()` renders a comparison table (calibration profile
in the header); `BenchmarkReport.to_csv()` renders the same rows as CSV for
spreadsheets/plotting. Every row also carries `baseline_name` /
`baseline_gate_count` / `baseline_depth` columns — `None` until the Phase 5
Qiskit O0–O3 harness (`run_benchmarks(..., baseline=qiskit_o2_pass)`) fills
them in, so the schema doesn't change shape when that lands.
