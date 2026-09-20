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

Mutation (commute / drop-candidate / reorder) is equivalence-preserving by
construction, built on Module B's commutation/inverse-pair logic
(``gates_commute`` / ``rule_based_candidates`` - issues #6 and #11, closed, so
this is their finished Sprint 1+2 work, not a stand-in subject to rework).
Crossover is deliberately NOT equivalence-preserving - splicing two different
parents' gate sequences fundamentally isn't - which is why the pipeline still
re-verifies the evolved front before returning it (see qco.pipeline).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from qco.ir.intermediate_representation import Gate, IntermediateRepresentation
from qco.modules.gnn_cancellation.features import gates_commute, rule_based_candidates

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
# Validity-preserving mutation ops (commute / drop-candidate / reorder).
#
# Each is equivalence-preserving *by construction*: commute and reorder only
# ever swap gates ``gates_commute`` has confirmed commute, and drop-candidate
# only ever removes a gate span ``rule_based_candidates`` has confirmed
# multiplies to identity (inverse_pair / identity_chain - "rotation_merge" is
# left alone since removing it changes the circuit, it needs replacement with
# the merged rotation, not deletion). Neither pymoo-specific - usable and
# testable on their own.
# --------------------------------------------------------------------------

def _legal_commute_swaps(gates: list[Gate]) -> list[int]:
    """Indices ``i`` where ``gates[i]``/``gates[i+1]`` can swap in place."""
    return [i for i in range(len(gates) - 1) if gates_commute(gates[i], gates[i + 1])]


def _reorder_gates(gates: list[Gate], rng: random.Random) -> list[Gate]:
    """Slide one gate a few positions earlier/later via chained legal commute
    swaps, stopping as soon as a hop isn't legal (may end up a no-op)."""
    if len(gates) < 2:
        return gates
    gates = list(gates)
    pos = rng.randrange(len(gates))
    direction = rng.choice((-1, 1))
    for _ in range(rng.randint(1, 3)):
        j = pos + direction
        if not (0 <= j < len(gates)):
            break
        left, right = (pos, j) if direction == 1 else (j, pos)
        if not gates_commute(gates[left], gates[right]):
            break
        gates[pos], gates[j] = gates[j], gates[pos]
        pos = j
    return gates


def mutate_validity_preserving(
    circuit: IntermediateRepresentation, rng: random.Random | None = None
) -> IntermediateRepresentation:
    """Apply one random commute / drop-candidate / reorder op, or return the
    circuit unchanged if none currently applies (e.g. a 1-gate circuit)."""
    rng = rng or random.Random()
    gates = list(circuit.gates)
    commute_points = _legal_commute_swaps(gates)
    droppable = [c for c in rule_based_candidates(circuit) if c.kind in ("inverse_pair", "identity_chain")]

    ops = []
    if commute_points:
        ops += ["commute", "reorder"]
    if droppable:
        ops.append("drop_candidate")
    if not ops:
        return circuit

    op = rng.choice(ops)
    if op == "commute":
        i = rng.choice(commute_points)
        gates[i], gates[i + 1] = gates[i + 1], gates[i]
    elif op == "reorder":
        gates = _reorder_gates(gates, rng)
    else:
        drop = set(rng.choice(droppable).gate_indices)
        gates = [g for idx, g in enumerate(gates) if idx not in drop]

    mutated = circuit.copy()
    mutated.gates = gates
    return mutated


# --------------------------------------------------------------------------
# pymoo plumbing: variable-length gate-sequence genome as an object-dtype
# "single variable" (pymoo's standard trick for non-array representations).
# --------------------------------------------------------------------------

if _HAVE_PYMOO:

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
        """Random commute / drop-candidate / reorder over one circuit's gate
        list - see ``mutate_validity_preserving`` above."""

        def __init__(self, prob: float):
            super().__init__(prob=prob)

        def _do(self, problem, X, **kwargs):
            # pymoo's Mutation.do() already decides, per individual, whether to keep
            # this result or the original based on `self.prob` - _do must mutate
            # unconditionally.
            rng = random.Random()
            for i in range(len(X)):
                X[i, 0] = mutate_validity_preserving(X[i, 0], rng)
            return X
