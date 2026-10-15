# Sprint 2 — Module A report (Maanas Nair, issue #10)

## 1. What was delivered

| Issue task | Where |
|---|---|
| Deterministic CR pass | `cr_pass.py` (+ `gate_algebra.py`, `commutation.py`) |
| PPO agent wired to `SchedulerEnv` | `ppo.py`, `features.py`, `_nn.py` (built-in, pure Python); `sb3_backend.py` (MaskablePPO, optional) |
| Training loop, checkpoints, `RLConfig` | `training.py`, `config.py`, `scripts/train_rl.py`, `configs/*.yaml`, `make train-rl` |
| Logging: depth, gate-count, reward curves | `metrics.csv` + `curves.svg` per run |
| Committed smoke run | `docs/rl_training/smoke/` (plus the full run in `ppo_5q/`) |

All files live under `src/qco/modules/rl_scheduler/` unless noted. `environment.py`, `rewrites.py`, `agent.py`, `__init__.py`, `Makefile`, `pyproject.toml` were modified; everything else is new.

### Design decisions (yours)
* **COMMUTE extended** (opt-in `commutation="full"`): adjacent-or-DAG-edge flips of gates that share qubits but provably commute. Default stays `"disjoint"` so Sprint 1 behaviour/tests are untouched.
* **Pluggable PPO**: built-in (runs in CI, no deps) and SB3 `MaskablePPO` (`backend: sb3`, `.[rl]`).

### How it works
* **CR pass**: for each gate scan forward, skipping disjoint gates; a same-operand partner that `reduce_pair` can cancel/merge is combined if every gate in between commutes with it; repeat to a fixed point. Also removes identity rotations. Env runs it at reset and after every action (`reduction=True`), and the terminal bonus is measured against the CR-reduced start, so the agent is credited only for what it adds.
* **Policy**: one MLP scores each legal action's 28-dim feature vector, softmax over legal actions (implicit masking, independent of circuit size → usable for 10/20/40 qubits). Critic MLP on 7 global features. PPO-clip + GAE + entropy bonus, Adam.
* **Episodes** end on NOOP (`stop_on_noop`), fixed point, or `max_episode_steps`; `run_policy` returns the best circuit seen.

## 2. Results (5-qubit held-out circuits, 12 circuits, 60k steps, seed 0)

| Method | Depth red. % | Gate red. % | Equivalent |
|---|---|---|---|
| CR only | 4.7 | 12.4 | yes |
| Random policy (on CR) | 9.0 | 12.4 | yes |
| Greedy-depth (1-step lookahead) | 12.4 | 12.4 | yes |
| **PPO agent** | **13.4** | 12.4 | yes |
| PPO agent, lookahead feature off | 5.2 | 12.4 | yes |

Training return rises from −1.3 to ≈ +0.8 (`docs/rl_training/ppo_5q/curves.svg`). Acceptance criteria: trains without error on 5-qubit circuits ✔; reward trending up ✔; CR alone reduces gate count on `rand_cliffordt_*` ✔ (tested).

**Honest caveat:** the agent beats greedy by under 1 point on 12 circuits and one seed, which is within noise. Without the lookahead feature it does not beat greedy. Treat "agent ≈ strong greedy" as the real finding; see gaps doc §3.

## 3. How to test and deploy

```bash
pip install -e ".[test]"            # exactly what CI installs (no numpy/torch)
python -m pytest -q                 # ~30 s; SB3 tests skip
pip install -e ".[test,rl]"         # adds gymnasium, torch, SB3, sb3-contrib
python -m pytest -q                 # SB3 + Gymnasium tests run
pip install numpy                   # optional: ~10x faster built-in PPO

make train-rl-smoke                 # seconds
make train-rl                       # ~2 min with numpy; writes runs/rl_5q + comparison
python scripts/train_rl.py --config configs/rl_ppo_5q.yaml --out runs/x --backend sb3
```
Use a trained agent:
```python
from qco.modules.rl_scheduler import RLScheduler
s = RLScheduler().load("docs/rl_training/ppo_5q/final.json")
optimized = s.schedule(circuit)
```
Deploy = merge the branch; no services. `runs/` and `checkpoints/` are git-ignored.

## 4. How to write unit tests for this code

* **Pure functions first** (`reduce_pair`, `commutes`, `commute`, `reduce_circuit`): build a tiny `IntermediateRepresentation`, assert the gate list, and for soundness compare against `EquivalenceChecker(backend="numeric")` on ≤ 4 qubits. Property-style: loop over `sample_circuits(n, k, seed)`.
* **Invariants over examples**: CR output is idempotent (`reduce(reduce(c)) == reduce(c)`), never longer, equivalent; rewrites on legal actions preserve equivalence.
* **Learner**: finite-difference gradient check on `ScalarMLP`; hand-computed GAE on a 2-step episode; `update` changes weights and returns finite stats; same seed ⇒ identical weights.
* **Training loop**: shrink via `RLConfig(total_timesteps=300, n_steps=100, ...)`, run in `tmp_path`, assert CSV rows, checkpoint files, load round-trip, resume.
* **Optional deps**: guard with an `*_AVAILABLE` flag and class/function-scoped `skipif`, never a module-level skip (see `TestSB3Backend`).
* Template: copy `tests/test_rl_ppo.py::TestTraining`.

## 5. Deviations / things to know
1. Built-in PPO is pure Python because CI has no numpy.
2. SB3 backend uses a slot-indexed `Discrete(K)` + action masks rather than the Sprint 1 `MultiDiscrete` adapter (which remains available in `spaces.py`).
3. `pyproject.toml` `rl` extra now includes `sb3-contrib>=2.3` (shared file).
4. Sprint 1 adjacency bug fixed (changes behaviour for non-adjacent calls: now raises).
5. `RLScheduler.schedule()` is unchanged when untrained; pipeline integration is Sprint 3.
6. `algo: ddqn` raises `NotImplementedError`.
