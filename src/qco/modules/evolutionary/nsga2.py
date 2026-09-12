"""Module D — NSGA-II over the population of verified candidate circuits.

Objectives (minimized): gate count, depth, (1 - estimated fidelity).
Fitness uses a structure-aware fidelity surrogate, not full unitary simulation,
so it scales past ~12 qubits (Scientific Reports 2026).

Sprint 2 (Swaraj Rane): real NSGA-II via pymoo, with crossover/mutation genetic
operators over variable-length gate sequences, plus fidelity surrogate v1
(layer/idle-time aware instead of a flat per-gate product). Falls back to the
Sprint-1 seeds-only non-dominated sort when pymoo isn't installed, same
optional-dependency pattern as the other modules (fallback numeric checker,
rule-based cancellation, ...).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from qco.ir.intermediate_representation import ARITY, Gate, IntermediateRepresentation

try:
    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.optimize import minimize

    _HAVE_PYMOO = True
except ImportError:  # pragma: no cover - exercised in envs without the "ea" extra
    _HAVE_PYMOO = False


@dataclass(frozen=True, slots=True)
class Objective:
    gate_count: float
    depth: float
    infidelity: float           # 1 - estimated fidelity

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.gate_count, self.depth, self.infidelity)


@dataclass(slots=True)
class ParetoPoint:
    circuit: IntermediateRepresentation
    objective: Objective


@dataclass(slots=True)
class NSGA2Config:
    pop_size: int = 60
    generations: int = 40
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    backend: str = "pymoo"      # "pymoo" | "deap"


def dominates(a: Objective, b: Objective) -> bool:
    at, bt = a.as_tuple(), b.as_tuple()
    return all(x <= y for x, y in zip(at, bt)) and any(x < y for x, y in zip(at, bt))


def non_dominated_front(points: list[ParetoPoint]) -> list[ParetoPoint]:
    front: list[ParetoPoint] = []
    for p in points:
        if any(dominates(q.objective, p.objective) for q in points if q is not p):
            continue
        front.append(p)
    # De-duplicate identical objective vectors.
    seen: set[tuple[float, float, float]] = set()
    unique: list[ParetoPoint] = []
    for p in front:
        key = p.objective.as_tuple()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


class NSGA2Compiler:
    def __init__(self, config: NSGA2Config | None = None,
                 fidelity_surrogate=None):
        self.config = config or NSGA2Config()
        self.fidelity_surrogate = fidelity_surrogate or structure_aware_fidelity_surrogate

    def evolve(self, seeds: list[IntermediateRepresentation]) -> list[ParetoPoint]:
        """Run NSGA-II (genetic search) seeded with ``seeds``, return the Pareto front.

        Without pymoo installed, degrades to scoring the seeds as-is (Sprint 1
        behaviour) rather than failing outright.
        """
        if not seeds:
            return []
        if _HAVE_PYMOO:
            return self._evolve_pymoo(seeds)
        return self._score_only(seeds)

    def _score_only(self, seeds: list[IntermediateRepresentation]) -> list[ParetoPoint]:
        points = [self.score(c) for c in seeds]
        return non_dominated_front(points)

    def score(self, circuit: IntermediateRepresentation) -> ParetoPoint:
        return ParetoPoint(
            circuit=circuit,
            objective=Objective(
                gate_count=float(circuit.gate_count()),
                depth=float(circuit.depth()),
                infidelity=1.0 - self.fidelity_surrogate(circuit),
            ),
        )

    def _evolve_pymoo(self, seeds: list[IntermediateRepresentation]) -> list[ParetoPoint]:
        problem = _CircuitProblem(self.fidelity_surrogate)
        algorithm = NSGA2(
            pop_size=max(self.config.pop_size, len(seeds)),
            sampling=_SeedSampling(seeds),
            crossover=_GateSequenceCrossover(self.config.crossover_prob),
            mutation=_GateSequenceMutation(self.config.mutation_prob),
            eliminate_duplicates=False,  # object-dtype genome: no cheap distance metric
        )
        res = minimize(problem, algorithm, ("n_gen", self.config.generations),
                        seed=1, verbose=False)
        circuits = [row[0] for row in res.X]
        return [
            ParetoPoint(circuit=c, objective=Objective(*f))
            for c, f in zip(circuits, res.F)
        ]


# --------------------------------------------------------------------------
# Fidelity surrogate v1 — structure-aware (accounts for circuit layering /
# idle time), not just a flat product of per-gate success probabilities.
# --------------------------------------------------------------------------

# ponytail: no device-connectivity model exists yet (calibration tables are
# per gate-type only), so crosstalk is charged for *any* two 2-qubit gates
# sharing a timing layer rather than only adjacent-qubit pairs. Tighten once
# qco.graphs carries a coupling map.
_CROSSTALK_PENALTY = 0.995   # extra per-gate factor when a 2q gate shares its layer with another
_T1_NS = 50_000.0            # coarse idle-decoherence time constant


def structure_aware_fidelity_surrogate(
    circuit: IntermediateRepresentation,
    gate_time_ns: dict[str, float] | None = None,
    error_rate: dict[str, float] | None = None,
    t1_ns: float = _T1_NS,
) -> float:
    """Cheap fidelity proxy that is sensitive to circuit *structure*, not just
    gate count: idle qubits lose fidelity to T1 decay while other qubits are
    busy, and concurrent 2-qubit gates take a small crosstalk hit.
    """
    from qco.evaluation.metrics import DEFAULT_ERROR_RATE, DEFAULT_GATE_TIME_NS

    t = gate_time_ns or DEFAULT_GATE_TIME_NS
    e = error_rate or DEFAULT_ERROR_RATE

    frontier = [0.0] * circuit.num_qubits          # each qubit's busy-until time
    layer_end = [0.0] * circuit.num_qubits          # kept only to detect same-layer 2q gates
    two_q_layer_count: dict[float, int] = {}

    fidelity = 1.0
    for g in circuit.gates:
        key = f"{len(g.qubits)}q"
        dur = t.get(key, t["2q"])
        start = max(frontier[q] for q in g.qubits)

        # Idle decay for every other qubit that sat out this step.
        for q in range(circuit.num_qubits):
            if q not in g.qubits:
                idle = max(0.0, start - frontier[q])
                if idle:
                    fidelity *= math.exp(-idle / t1_ns)

        fidelity *= 1.0 - e.get(key, e["2q"])
        if g.is_two_qubit:
            two_q_layer_count[start] = two_q_layer_count.get(start, 0) + 1
            if two_q_layer_count[start] > 1:
                fidelity *= _CROSSTALK_PENALTY

        for q in g.qubits:
            frontier[q] = start + dur

    return fidelity


# --------------------------------------------------------------------------
# pymoo plumbing: variable-length gate-sequence genome as an object-dtype
# "single variable" (pymoo's standard trick for non-array representations).
# --------------------------------------------------------------------------

if _HAVE_PYMOO:

    _PARAM_GATES = {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"}

    def _random_gate(num_qubits: int, rng: random.Random) -> Gate:
        choices = [n for n, arity in ARITY.items() if arity <= num_qubits]
        name = rng.choice(choices)
        arity = ARITY[name]
        qubits = tuple(rng.sample(range(num_qubits), arity))
        params = (rng.uniform(0, 6.283185307),) if name in _PARAM_GATES else ()
        return Gate(name=name, qubits=qubits, params=params)

    class _CircuitProblem(ElementwiseProblem):
        def __init__(self, fidelity_surrogate):
            super().__init__(n_var=1, n_obj=3, n_constr=0)
            self.fidelity_surrogate = fidelity_surrogate

        def _evaluate(self, x, out, *args, **kwargs):
            circuit: IntermediateRepresentation = x[0]
            out["F"] = [
                float(circuit.gate_count()),
                float(circuit.depth()),
                1.0 - self.fidelity_surrogate(circuit),
            ]

    class _SeedSampling(Sampling):
        """Seeds the initial population by cycling the pipeline's candidate(s)."""

        def __init__(self, seeds: list[IntermediateRepresentation]):
            super().__init__()
            self.seeds = seeds

        def _do(self, problem, n_samples, **kwargs):
            X = np.empty((n_samples, 1), dtype=object)
            for i in range(n_samples):
                X[i, 0] = self.seeds[i % len(self.seeds)].copy()
            return X

    class _GateSequenceCrossover(Crossover):
        """One-point crossover over the two parents' gate lists."""

        def __init__(self, prob: float):
            super().__init__(2, 2, prob=prob)

        def _do(self, problem, X, **kwargs):
            _, n_matings, _ = X.shape
            Y = np.empty_like(X)
            rng = random.Random()
            for k in range(n_matings):
                p1: IntermediateRepresentation = X[0, k, 0]
                p2: IntermediateRepresentation = X[1, k, 0]
                g1, g2 = p1.gates, p2.gates
                c1 = rng.randint(0, len(g1))
                c2 = rng.randint(0, len(g2))
                child1 = p1.copy()
                child1.gates = g1[:c1] + g2[c2:]
                child2 = p2.copy()
                child2.gates = g2[:c2] + g1[c1:]
                Y[0, k, 0] = child1
                Y[1, k, 0] = child2
            return Y

    class _GateSequenceMutation(Mutation):
        """Random insert / delete / swap over one circuit's gate list."""

        def __init__(self, prob: float):
            super().__init__(prob=prob)

        def _do(self, problem, X, **kwargs):
            # pymoo's Mutation.do() already decides, per individual, whether to keep
            # this result or the original based on `self.prob` - _do must mutate
            # unconditionally.
            rng = random.Random()
            for i in range(len(X)):
                circuit: IntermediateRepresentation = X[i, 0]
                gates = list(circuit.gates)
                op = rng.choice(("delete", "insert", "swap"))
                if op == "delete" and gates:
                    gates.pop(rng.randrange(len(gates)))
                elif op == "insert" and circuit.num_qubits:
                    gates.insert(rng.randrange(len(gates) + 1), _random_gate(circuit.num_qubits, rng))
                elif op == "swap" and len(gates) >= 2:
                    a, b = rng.sample(range(len(gates)), 2)
                    gates[a], gates[b] = gates[b], gates[a]
                mutated = circuit.copy()
                mutated.gates = gates
                X[i, 0] = mutated
            return X
