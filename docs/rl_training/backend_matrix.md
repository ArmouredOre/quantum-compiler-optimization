# Cross-backend matrix: train on A, test on B (issue #29)

Metric: mean execution-time reduction (%) under the *test* backend's calibration, 60 held-out 5-qubit circuits, mean ± std over seeds [0, 1, 2, 3, 4, 5, 6]. Rows = test backend, columns = backend the policy was trained on.

| test \ train | superconducting_ibm_like | trapped_ion_like | greedy one-step time heuristic |
|---|---|---|---|
| superconducting_ibm_like | **8.01 ± 0.35** | 7.48 ± 0.43 | 7.62 |
| trapped_ion_like | 7.51 ± 0.42 | **7.14 ± 0.36** | 7.33 |

* Policy wins on its own backend (mean): {'superconducting_ibm_like': True, 'trapped_ion_like': False}
* Seeds (out of 7) where the own-backend policy beats the other profile's policy: {'superconducting_ibm_like': 6, 'trapped_ion_like': 2}
* Fraction of (circuit, test-backend, seed) cases where the two backends' policies emit different schedules: 0.21
