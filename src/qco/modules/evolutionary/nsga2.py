"""Module D — NSGA-II over the population of verified candidate circuits.

Objectives (minimized): gate count, depth, (1 - estimated fidelity).
Fitness uses a structure-aware fidelity surrogate, not full unitary simulation,
so it scales past ~12 qubits (Scientific Reports 2026).

Phase 3 (Member 5) implements selection/crossover/mutation via pymoo or DEAP.
Phase 2 ships:
* the ``Objective`` / ``ParetoPoint`` contracts, and
* a working non-dominated-sort so the pipeline can already return a Pareto front
  from whatever candidates Modules A-C produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qco.ir.intermediate_representation import IntermediateRepresentation


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
        self.fidelity_surrogate = fidelity_surrogate or _default_surrogate

    def evolve(self, seeds: list[IntermediateRepresentation]) -> list[ParetoPoint]:
        """Phase 2: score the seeds and return their non-dominated front.

        Phase 3 (Member 5): run the full NSGA-II loop with genetic operators over
        gate sequences, seeded with these circuits.
        """
        points = [
            ParetoPoint(
                circuit=c,
                objective=Objective(
                    gate_count=float(c.gate_count()),
                    depth=float(c.depth()),
                    infidelity=1.0 - self.fidelity_surrogate(c),
                ),
            )
            for c in seeds
        ]
        return non_dominated_front(points)


def _default_surrogate(circuit: IntermediateRepresentation) -> float:
    """Cheap fidelity proxy: product of per-gate survival probabilities.

    Placeholder error rates (1q: 1e-3, 2q: 1e-2) until device calibration data is
    wired in (Phase 5).
    """
    f = 1.0
    for g in circuit.gates:
        f *= (1 - 1e-2) if g.is_two_qubit else (1 - 1e-3)
    return f
