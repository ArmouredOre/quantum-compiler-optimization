"""Module A — reference policies and the held-out evaluation used by training.

``evaluate`` runs any *chooser* (``chooser(env, actions) -> index``) on a set of
circuits and reports mean depth / gate-count reduction versus the *raw* input
(so the CR pass's own contribution is included, and separable via
``cr_only``). Baselines: ``random`` and ``greedy_depth`` (one-step lookahead,
random ties) — the trained agent must beat both to be worth anything.
"""

from __future__ import annotations

import random

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.cr_pass import reduce_circuit
from qco.modules.rl_scheduler.features import _lookahead
from qco.modules.rl_scheduler.ppo import PPOPolicy, make_env, run_policy

EQUIV_QUBIT_CAP = 8


def random_chooser(rng: random.Random):
    return lambda env, actions: rng.randrange(len(actions))


def greedy_depth_chooser(rng: random.Random):
    """Apply the move with the largest strict one-step depth drop; NOOP if none."""

    def choose(env, actions):
        best, best_v = [0], 0.0
        for k, a in enumerate(actions):
            if a.kind.name == "NOOP":
                continue
            v = _lookahead(env, a)
            if v > best_v:
                best, best_v = [k], v
            elif v == best_v and v > 0:
                best.append(k)
        return rng.choice(best)

    return choose


def rollout(circuit: IntermediateRepresentation, config: RLConfig, chooser) -> IntermediateRepresentation:
    """Run ``chooser`` to termination; return the best (depth, gates) circuit seen."""
    env = make_env(circuit, config)
    _, info = env.reset()
    best = env.circuit
    key0 = (best.depth(), best.gate_count())
    while True:
        actions = info["legal_actions"]
        _, _, term, trunc, info = env.step(actions[chooser(env, actions)])
        key = (env.circuit.depth(), env.circuit.gate_count())
        if key < key0:
            best, key0 = env.circuit, key
        if term or trunc:
            return best


def _pct(a: int, b: int) -> float:
    return 100.0 * (a - b) / a if a else 0.0


def evaluate(
    circuits: list[IntermediateRepresentation],
    config: RLConfig,
    *,
    policy: PPOPolicy | None = None,
    chooser=None,
    cr_only: bool = False,
    verify: bool = False,
) -> dict:
    """Mean ``depth_red_pct`` / ``gate_red_pct`` vs the raw circuits.

    Exactly one of ``policy`` (greedy), ``chooser`` or ``cr_only`` is used.
    ``verify=True`` asserts unitary equivalence (circuits <= 8 qubits only).
    """
    from qco.modules.smt_verifier.equivalence import EquivalenceChecker

    checker = EquivalenceChecker() if verify else None
    d, g, ok = [], [], True
    for c in circuits:
        if cr_only:
            out = reduce_circuit(c)
        elif policy is not None:
            out, _ = run_policy(policy, c, config, greedy=True)
        else:
            out = rollout(c, config, chooser)
        d.append(_pct(c.depth(), out.depth()))
        g.append(_pct(c.gate_count(), out.gate_count()))
        if checker is not None and c.num_qubits <= EQUIV_QUBIT_CAP:
            ok = ok and bool(checker.check(c, out).equivalent)
    n = max(len(circuits), 1)
    res = {"depth_red_pct": sum(d) / n, "gate_red_pct": sum(g) / n, "n": len(circuits)}
    if verify:
        res["equivalent"] = ok
    return res
