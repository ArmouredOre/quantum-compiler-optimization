"""Measured scaling of Module A's cost drivers (#37), to back docs/module_a_complexity_and_scaling.md.

Times each component on random Clifford+T circuits of growing gate count G (fixed Q) and of growing
qubit count Q (fixed layers), and fits the log-log slope of time vs G. Components:

  depth           IR.depth()                                   (O(G) by construction)
  cr_pass         reduce_circuit (CR to a fixed point)
  legal_actions   SchedulerEnv.legal_actions()
  encode_state    candidate features incl. the one-step lookahead
  env_step        one SchedulerEnv.step() with the greedy-best action
  episode         run_policy() with the committed checkpoint (default 40 steps), only up to --episode-max-gates

    python benchmarks/bench_module_a_scaling.py [--out docs/module_a_scaling.md]
"""

from __future__ import annotations

import argparse
import math
import random
import time
import warnings
from pathlib import Path

from qco.modules.rl_scheduler.circuits import clifford_t
from qco.modules.rl_scheduler.cr_pass import reduce_circuit
from qco.modules.rl_scheduler.features import encode_state
from qco.modules.rl_scheduler.ppo import load_policy, make_env, run_policy
from qco.modules.rl_scheduler.config import RLConfig

warnings.simplefilter("ignore")


def best_of(fn, reps=3):
    best = math.inf
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def measure(ir, policy, cfg, episode: bool):
    env = make_env(ir, cfg)
    _, info = env.reset()
    acts = info["legal_actions"]
    rows, _ = encode_state(env, acts, True)
    la = [r[25] for r in rows]
    best = max(range(len(acts)), key=lambda k: la[k])
    res = {
        "G": ir.gate_count(),
        "A": len(acts),
        "depth": best_of(ir.depth, 5),
        "cr_pass": best_of(lambda: reduce_circuit(ir)),
        "legal_actions": best_of(env.legal_actions),
        "encode_state": best_of(lambda: encode_state(env, acts, True)),
    }
    e2 = make_env(ir, cfg)
    e2.reset()
    t = time.perf_counter()
    e2.step(acts[best])
    res["env_step"] = time.perf_counter() - t
    res["episode"] = best_of(lambda: run_policy(policy, ir, cfg), 1) if episode else None
    return res


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y and y > 1e-5]
    if len(pts) < 2:
        return float("nan")
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/rl_ppo_5q.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    ap.add_argument("--qubits", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--episode-max-gates", type=int, default=700)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global policy
    policy, payload = load_policy(a.policy)
    cfg = RLConfig.from_dict(payload["config"])
    comps = ["depth", "cr_pass", "legal_actions", "encode_state", "env_step", "episode"]
    lines = []

    def table(title, rows, key):
        lines.append(f"### {title}\n")
        lines.append(f"| {key} | G | legal actions | " + " | ".join(f"{c} (ms)" for c in comps) + " |")
        lines.append("|---|---|---|" + "---|" * len(comps))
        for k, r in rows:
            lines.append(f"| {k} | {r['G']} | {r['A']} | " + " | ".join("-" if r[c] is None else f"{r[c] * 1e3:.1f}" for c in comps) + " |")
        lines.append("")

    rows = []
    for L in a.layers:
        ir = clifford_t(20, L, random.Random(7))
        rows.append((f"L={L}", measure(ir, policy, cfg, ir.gate_count() <= a.episode_max_gates)))
        print(rows[-1], flush=True)
    table("Gate-count sweep (Q = 20)", rows, "layers")
    lines.append("Fitted log-log slope of time vs G (1 = linear, 2 = quadratic): " + ", ".join(
        f"{c} {slope([r['G'] for _, r in rows], [r[c] for _, r in rows]):.2f}" for c in comps) + "\n")
    rows2 = []
    for Q in a.qubits:
        ir = clifford_t(Q, 20, random.Random(7))
        rows2.append((f"Q={Q}", measure(ir, policy, cfg, ir.gate_count() <= a.episode_max_gates)))
        print(rows2[-1], flush=True)
    table("Qubit sweep (20 layers)", rows2, "qubits")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
