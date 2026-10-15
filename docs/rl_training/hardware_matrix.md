# Cross-backend matrix: train on A, test on B (issue #29)

Metric: mean execution-time reduction (%) under the *test* backend's calibration, 40 held-out 5-qubit circuits, mean ± std over seeds [0, 1, 2, 3]. Rows = test backend, columns = backend the policy was trained on.

| test \ train | ibm_grid_like | line_superconducting_like | ion_all_to_all_like | greedy one-step time heuristic |
|---|---|---|---|---|
| ibm_grid_like | **8.49 ± 0.28** | 8.52 ± 1.04 | 6.26 ± 0.47 | 7.07 |
| line_superconducting_like | 7.46 ± 0.23 | **6.60 ± 0.34** | 5.88 ± 0.13 | 6.26 |
| ion_all_to_all_like | 1.90 ± 0.00 | 1.90 ± 0.00 | **1.90 ± 0.01** | 1.91 |

* Policy wins on its own backend (mean): {'ibm_grid_like': False, 'line_superconducting_like': False, 'ion_all_to_all_like': True}
* Seeds (out of 4) where the own-backend policy beats the other profile's policy: {'ibm_grid_like': 1, 'line_superconducting_like': 0, 'ion_all_to_all_like': 2}
* Fraction of (circuit, test-backend, seed) cases where the two backends' policies emit different schedules: 0.07
