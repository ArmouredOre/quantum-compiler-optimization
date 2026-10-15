# [S3][sub-issue of #29] Hardware-aware backend profiles (topology, SWAP cost, native gates) for per-backend RL

**Parent:** #29 (per-backend training, cross-backend matrix). **Owner:** Maanas Nair (Module A). **Depends on:** #26 (done), #10 (done).

## Why
#29 requires "policies trained under `superconducting_ibm_like` and `trapped_ion_like` produce measurably different schedules, and each wins on its own backend." With the existing `Calibration` profiles this is not achievable: they differ only by per-arity duration/error *scale*, fidelity depends on gate counts (which commuting never changes), and a policy run on another backend simply sees that backend's timings through its lookahead feature. Measured (7 seeds, `docs/rl_training/backend_matrix.md`): schedules differ in 21 % of cases, but the ion-like policy loses on its own backend. Backends need to differ in *structure*.

## Scope (all additive, nothing existing changes)
- [x] New `qco/evaluation/hardware.py`: `HardwareProfile` with topology, greedy SWAP-routing cost, per-gate-name time/error factors, 2q-parallelism cap; three illustrative profiles. `Calibration` / `metrics.py` untouched.
- [x] Module A opt-in: `RLConfig.hardware`, `SchedulerEnv(hardware=...)`, `lookahead_metric="time"` uses the hardware model; backend matrix accepts hardware names.
- [x] Tests: `tests/test_hardware_profiles.py` (topology, routing, order-dependence, metric consistency with `Calibration` when unstructured, env/config integration).
- [x] Study: `docs/rl_training/hardware_matrix.md` (3 profiles x 4 seeds).
- [ ] Decide `ion_all_to_all_like` (degenerate: serial 2q cap makes time order-independent) and re-run.
- [ ] Review with owners of Stage 6 (`evaluation/`) and Module D whether `HardwareProfile` should become a shared type (NSGA-II fitness already notes the missing connectivity model).
- [ ] Optional: expose the profile in `pipeline.py` / benchmark runner (out of scope here).

## Acceptance criteria
1. `HardwareProfile` is opt-in and does not change any existing metric or committed result (existing test suite unchanged and green).
2. Cross-backend matrix over >= 3 hardware profiles committed with mean +- std over >= 3 seeds.
3. Policies trained on different profiles produce measurably different schedules **and** each beats the other profiles' policies on its own backend by more than seed noise, **or** the report states clearly which profiles fail and why.

## Current status vs criteria
Criteria 1 and 2 met. Criterion 3: partially. Routing-constrained training beats all-to-all training by ~2.2 points on the grid and beats the greedy time heuristic by ~1.4, but grid and line policies tie/cross, and the ion profile is degenerate. See `docs/hardware_profiles.md`.

## Risks / questions for the team
- Does the team want hardware-aware routing in scope at all (touches evaluation, pipeline expectations, SMT)?
- Profiles are illustrative; real per-backend calibration (Qiskit `BackendProperties`) is Phase 5.
- Greedy router is an estimate; a stronger router changes absolute numbers.

## Files
`src/qco/evaluation/hardware.py`, `src/qco/modules/rl_scheduler/{environment,features,config,ppo,backend_matrix}.py`, `scripts/eval_backend_matrix.py`, `tests/test_hardware_profiles.py`, `docs/hardware_profiles.md`, `docs/rl_training/hardware_matrix.{md,json}`.
