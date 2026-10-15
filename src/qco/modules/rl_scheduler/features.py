"""Module A — state/candidate featurization for the built-in PPO.

The policy scores each *legal action* from a fixed-size feature vector, so the
same weights work for any circuit size (important for the Sprint 3
5 -> 10/20/40-qubit generalization study) and no padding or fixed action-index
space is needed. Pure Python: no numpy required.

Candidate features (``CANDIDATE_DIM`` = 28)
------------------------------------------
kind one-hot (noop / commute / merge) .................. 3
gate-class one-hot for the two gates involved ........... 14
shared-qubit count, index distance ...................... 2
ASAP layer and remaining-path length of both gates,
  normalised by depth, plus on-critical-path flags ...... 6
one-step depth change if applied (optional lookahead) ... 1
position in circuit ..................................... 1
bias .................................................... 1

Global features (``GLOBAL_DIM`` = 7) feed the value function.
"""

from __future__ import annotations

from qco.ir.intermediate_representation import Gate, IntermediateRepresentation
from qco.modules.rl_scheduler.rewrites import IllegalActionError, apply_action

CANDIDATE_DIM = 28
GLOBAL_DIM = 7

_CLASSES = ("z1", "xy1", "g1", "cx2", "diag2", "o2", "o3")
_Z1 = {"z", "s", "sdg", "t", "tdg", "rz", "p"}
_XY1 = {"x", "rx", "y", "ry"}
_CX2 = {"cx", "crx", "cry"}
_DIAG2 = {"cz", "cp", "crz", "rzz"}
_LOOKAHEAD_CLIP = 3.0


def gate_class(g: Gate) -> int:
    n = len(g.qubits)
    if n == 1:
        return 0 if g.name in _Z1 else (1 if g.name in _XY1 else 2)
    if n == 2:
        return 3 if g.name in _CX2 else (4 if g.name in _DIAG2 else 5)
    return 6


def asap_and_tail(ir: IntermediateRepresentation) -> tuple[list[int], list[int]]:
    """Per-gate ASAP finish layer (1-based) and longest remaining path (incl. self)."""
    gates = ir.gates
    n = len(gates)
    frontier = [0] * ir.num_qubits
    layer = [0] * n
    for k, g in enumerate(gates):
        layer[k] = max(frontier[q] for q in g.qubits) + 1
        for q in g.qubits:
            frontier[q] = layer[k]
    back = [0] * ir.num_qubits
    tail = [0] * n
    for k in range(n - 1, -1, -1):
        g = gates[k]
        tail[k] = max(back[q] for q in g.qubits) + 1
        for q in g.qubits:
            back[q] = tail[k]
    return layer, tail


def _lookahead(env, action) -> float:
    """One-step improvement of ``action``, clipped to [-1, 1].

    ``env.lookahead_metric`` = "depth" (default: layers saved) or "time" (calibrated
    critical-path ns saved, in average-layer units; backend-dependent, Sprint 3 / #29).
    """
    try:
        after = apply_action(env.circuit, action.kind.name, action.i, action.j, env.commutation)
    except IllegalActionError:
        return 0.0
    if getattr(env, "lookahead_metric", "depth") == "time":
        delta = (env.time_of(env.circuit) - env.time_of(after)) / env.time_unit
    else:
        delta = env.circuit.depth() - after.depth()
    return max(-_LOOKAHEAD_CLIP, min(_LOOKAHEAD_CLIP, float(delta))) / _LOOKAHEAD_CLIP


def encode_state(env, actions, lookahead: bool = True) -> tuple[list[list[float]], list[float]]:
    """Return ``(X, g)``: one feature row per action in ``actions`` and the global vector."""
    ir = env.circuit
    gates = ir.gates
    n = max(len(gates), 1)
    depth = max(ir.depth(), 1)
    layer, tail = asap_and_tail(ir)

    rows: list[list[float]] = []
    la: list[float] = []
    for a in actions:
        row = [0.0] * CANDIDATE_DIM
        row[26] = 1.0  # position slot default; overwritten below
        row[27] = 1.0  # bias
        if a.kind.name == "NOOP" or a.i < 0 or a.i >= len(gates) or a.j >= len(gates):
            row[0] = 1.0
            row[26] = 0.0
            rows.append(row)
            la.append(0.0)
            continue
        i, j = (a.i, a.j) if a.j >= 0 else (a.i, a.i)
        u, v = gates[i], gates[j]
        row[1 if a.kind.name == "COMMUTE" else 2] = 1.0
        row[3 + gate_class(u)] = 1.0
        row[10 + gate_class(v)] = 1.0
        row[17] = len(set(u.qubits) & set(v.qubits)) / 2.0
        row[18] = abs(j - i) / n
        row[19] = layer[i] / depth
        row[20] = layer[j] / depth
        row[21] = tail[i] / depth
        row[22] = tail[j] / depth
        row[23] = 1.0 if layer[i] + tail[i] - 1 == depth else 0.0
        row[24] = 1.0 if layer[j] + tail[j] - 1 == depth else 0.0
        row[25] = _lookahead(env, a) if lookahead else 0.0
        row[26] = min(i, j) / n
        rows.append(row)
        la.append(row[25])

    ref_depth = max(env.reference.depth(), 1)
    ref_gates = max(env.reference.gate_count(), 1)
    n_moves = sum(1 for a in actions if a.kind.name != "NOOP")
    g = [
        ir.depth() / ref_depth,
        ir.gate_count() / ref_gates,
        env._steps / max(env.max_episode_steps, 1),
        n_moves / n,
        max(la) if la else 0.0,
        (sum(1 for x in la if x > 0) / n_moves) if n_moves else 0.0,
        1.0,
    ]
    return rows, g
