# Hardware-aware profiles (`qco.evaluation.hardware`)

Opt-in cost model added for the per-backend RL study (#29 follow-up). **Nothing existing changes**: `Calibration`, `execution_time`, `estimated_fidelity` and every current caller are untouched; the module is new and only used when asked for.

## What it models
| Feature | How |
|---|---|
| Connectivity | `topology` = `all_to_all` / `line` / `ring` / `grid` coupling graph |
| SWAP routing | deterministic greedy router in program order, identity start layout, layout evolves; a SWAP costs `swap_two_qubit_gates` (3) two-qubit gates. Cost therefore depends on gate **order** |
| Native-gate cost | per-gate-name `gate_time_factor` / `gate_error_factor` on top of a base `Calibration` |
| Parallelism | `max_parallel_2q` (e.g. 1 for a single ion chain) |

The routed circuit is only costed, never returned, so the IR, SMT verifier and pipeline see logical circuits as before.

API: `get_hardware(name)`, `route(circuit, profile)`, `swap_count`, `hardware_execution_time` (ns), `hardware_fidelity`. Built-ins (illustrative, order-of-magnitude, not live-device data): `ibm_grid_like`, `line_superconducting_like`, `ion_all_to_all_like`.

## Use in Module A
`RLConfig(hardware="ibm_grid_like")` (or `SchedulerEnv(..., hardware=...)`): the per-step time and fidelity terms use the hardware model; terminal bonus keeps using the base `Calibration`. `backend_matrix.py` and `scripts/eval_backend_matrix.py --profiles ibm_grid_like line_superconducting_like ion_all_to_all_like` accept hardware names alongside calibration names.

## Result (`docs/rl_training/hardware_matrix.md`; 40 held-out 5q circuits, 4 seeds, 60k steps)
| test \ train | ibm_grid | line | ion_all_to_all | greedy time |
|---|---|---|---|---|
| ibm_grid | 8.49 ± 0.28 | 8.52 ± 1.04 | 6.26 ± 0.47 | 7.07 |
| line | 7.46 ± 0.23 | 6.60 ± 0.34 | 5.88 ± 0.13 | 6.26 |
| ion_all_to_all | 1.90 | 1.90 | 1.90 | 1.91 |

* **Structure matters (new):** a policy trained on a routing-constrained device beats one trained on the all-to-all profile by ~2.2 points on the grid (8.49 vs 6.26, ≫ seed std), and beats the greedy time heuristic by ~1.4 points there. The calibration-only profiles showed no such gap.
* **Own-backend win is still not clean:** ion_all_to_all wins its column only trivially (see next point); on `line`, the grid-trained policy beats the line-trained one (7.46 vs 6.60), and on `grid` the grid- and line-trained policies tie (8.49 vs 8.52 ± 1.04).
* **`ion_all_to_all_like` is degenerate for this action space:** with `max_parallel_2q = 1` total time is the sum of entangling-gate durations whatever the order, so commuting can gain ~1.9 % at best and every method ties. This is a profile-design finding, not a learning one; the follow-up should choose a profile where order can matter (e.g. no cap, or a cap of 2+).
* The line-trained policy is noisier (std 1.04 on grid); likely harder credit assignment from large SWAP-cost swings.

## Limits / caveats
Greedy routing with a fixed identity layout is a cost *estimate*, not an optimal router (no initial-layout search, no lookahead routing). Profiles are illustrative. 3-qubit gates are routed by moving operands next to the first operand. Equivalence of optimized circuits is unaffected (logical circuits only).
