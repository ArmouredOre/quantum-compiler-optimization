"""Module A — random training / evaluation circuits for the RL scheduler.

Self-contained generators (no dependency on ``benchmarks/``) that mirror the
*families* in ``benchmarks/suite.yaml`` — random Clifford+T, random CNOT+Pauli,
QAOA-style diagonal layers and QFT-style phase ladders — but use their own code
and seeds, so training never sees the benchmark circuits that Sprint 3's
generalization study evaluates on.
"""

from __future__ import annotations

import math
import random

from qco.ir.intermediate_representation import IntermediateRepresentation

FAMILIES = ("clifford_t", "cnot_pauli", "qaoa", "qft")


def _pairs_layer(n: int, rng: random.Random, p_cx: float) -> list[tuple[int, int]]:
    order = list(range(n))
    rng.shuffle(order)
    return [(order[i], order[i + 1]) for i in range(0, n - 1, 2) if rng.random() < p_cx]


def clifford_t(n: int, layers: int, rng: random.Random, t_fraction: float = 0.15) -> IntermediateRepresentation:
    ir = IntermediateRepresentation(n, name=f"train_clifford_t_n{n}")
    for _ in range(layers):
        for q in range(n):
            r = rng.random()
            if r < t_fraction:
                ir.add(rng.choice(["t", "tdg"]), [q])
            elif r < 0.55:
                ir.add(rng.choice(["h", "s", "sdg"]), [q])
        for a, b in _pairs_layer(n, rng, 0.6):
            ir.add("cx", [a, b])
    return ir


def cnot_pauli(n: int, layers: int, rng: random.Random) -> IntermediateRepresentation:
    ir = IntermediateRepresentation(n, name=f"train_cnot_pauli_n{n}")
    for _ in range(layers):
        for q in range(n):
            if rng.random() < 0.5:
                ir.add(rng.choice(["x", "y", "z"]), [q])
        for a, b in _pairs_layer(n, rng, 0.6):
            ir.add("cx", [a, b])
    return ir


def qaoa(n: int, layers: int, rng: random.Random) -> IntermediateRepresentation:
    """H layer, then ``p`` rounds of ``rzz`` cost edges and ``rx`` mixers."""
    ir = IntermediateRepresentation(n, name=f"train_qaoa_n{n}")
    all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    for q in range(n):
        ir.add("h", [q])
    p = max(1, layers // 8)
    for _ in range(p):
        edges = rng.sample(all_edges, k=min(len(all_edges), max(n, int(1.5 * n))))
        rng.shuffle(edges)
        gamma = rng.uniform(0.1, math.pi - 0.1)
        for a, b in edges:
            ir.add("rzz", [a, b], [2 * gamma])
        beta = rng.uniform(0.1, math.pi / 2)
        for q in range(n):
            ir.add("rx", [q], [2 * beta])
    return ir


def qft(n: int, layers: int, rng: random.Random) -> IntermediateRepresentation:  # noqa: ARG001
    ir = IntermediateRepresentation(n, name=f"train_qft_n{n}")
    for j in range(n):
        ir.add("h", [j])
        for k in range(j + 1, n):
            ir.add("cp", [k, j], [math.pi / 2 ** (k - j)])
    return ir


_BUILDERS = {"clifford_t": clifford_t, "cnot_pauli": cnot_pauli, "qaoa": qaoa, "qft": qft}


def random_circuit(num_qubits: int, rng: random.Random, family: str | None = None, layers: int = 16) -> IntermediateRepresentation:
    """One random circuit; ``family=None`` picks one uniformly."""
    family = family or rng.choice(FAMILIES)
    if family not in _BUILDERS:
        raise ValueError(f"unknown circuit family {family!r}; choose from {FAMILIES}")
    return _BUILDERS[family](num_qubits, layers, rng)


def sample_circuits(
    num_qubits: int,
    count: int,
    seed: int,
    families: tuple[str, ...] = FAMILIES,
    layers: int = 16,
) -> list[IntermediateRepresentation]:
    """``count`` circuits cycling through ``families`` (deterministic in ``seed``)."""
    rng = random.Random(f"qco-rl-circuits-{seed}-{num_qubits}")
    return [random_circuit(num_qubits, rng, families[i % len(families)], layers) for i in range(count)]
