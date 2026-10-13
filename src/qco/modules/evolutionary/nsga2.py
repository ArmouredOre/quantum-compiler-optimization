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

Novelty N3 (#30): the genome is a vector of *variant indices*, one per block.
Every variant of block ``i`` is equivalent to block ``i``, and blocks compose
in list order (``qco.graphs.partitioning`` contract), so every genome decodes
to a circuit equivalent to the input - crossover (uniform, per block) and
mutation (switch a block's variant, or derive a new variant with the
equivalence-preserving commute / drop-candidate / reorder ops built on Module
B's ``gates_commute`` / ``rule_based_candidates``) never leave the
equivalence class. The front is certified by construction, no post-hoc filter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from qco.graphs.partitioning import partition_circuit
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

    def evolve(
        self,
        seeds: list[IntermediateRepresentation],
        block_variants: list[list[IntermediateRepresentation]] | None = None,
    ) -> list[ParetoPoint]:
        """Run NSGA-II over block-variant genomes, return the Pareto front.

        ``block_variants[i]`` lists circuits each equivalent to block ``i`` (the
        caller verified them); concatenating one variant per block, in order,
        must give a circuit equivalent to the input. Without it, ``seeds[0]`` is
        partitioned and each block starts with itself as its only variant.
        Extra seeds are scored as-is and merged into the front.

        Without pymoo installed, degrades to scoring the seeds (plus the first
        variant of every block) rather than failing outright.
        """
        if not seeds and not block_variants:
            return []
        if block_variants is None:
            block_variants = [[b.ir] for b in partition_circuit(seeds[0])]
        pools = [list(v) for v in block_variants if v]
        extra = [self.score(c) for c in seeds]
        if not _HAVE_PYMOO or not pools:
            base = [self.score(_decode(pools, (0,) * len(pools)))] if pools else []
            return non_dominated_front(base + extra)
        return non_dominated_front(self._evolve_pymoo(pools) + extra)

    def score(self, circuit: IntermediateRepresentation) -> ParetoPoint:
        return ParetoPoint(
            circuit=circuit,
            objective=Objective(
                gate_count=float(circuit.gate_count()),
                depth=float(circuit.depth()),
                infidelity=1.0 - self.fidelity_surrogate(circuit),
            ),
        )

    def _evolve_pymoo(self, pools: list[list[IntermediateRepresentation]]) -> list[ParetoPoint]:
        problem = _BlockProblem(pools, self.fidelity_surrogate)
        algorithm = NSGA2(
            pop_size=self.config.pop_size,
            sampling=_BlockSampling(pools),
            crossover=_BlockCrossover(self.config.crossover_prob),
            mutation=_BlockMutation(pools, self.config.mutation_prob),
            eliminate_duplicates=False,  # object-dtype genome: no cheap distance metric
        )
        # copy_algorithm=False: the mutation operator must grow the *same* pools
        # the problem decodes from (pymoo deep-copies the algorithm otherwise).
        res = minimize(problem, algorithm, ("n_gen", self.config.generations),
                       copy_algorithm=False, seed=1, verbose=False)
        return [
            ParetoPoint(circuit=_decode(pools, row[0]), objective=Objective(*f))
            for row, f in zip(res.X, res.F)
        ]


def _decode(pools: list[list[IntermediateRepresentation]], genome: tuple[int, ...]) -> IntermediateRepresentation:
    """Concatenate the chosen variant of every block."""
    first = pools[0][0]
    out = IntermediateRepresentation(first.num_qubits, name=first.name.split("#")[0])
    for pool, idx in zip(pools, genome):
        out.gates.extend(pool[idx].gates)
    return out


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
# pymoo plumbing: genome = tuple of per-block variant indices, stored as an
# object-dtype "single variable" (pymoo's trick for non-array representations).
# The variant pools are shared and grow as mutation derives new variants.
# --------------------------------------------------------------------------

if _HAVE_PYMOO:

    class _BlockProblem(ElementwiseProblem):
        def __init__(self, pools, fidelity_surrogate):
            super().__init__(n_var=1, n_obj=3, n_constr=0)
            self.pools = pools
            self.fidelity_surrogate = fidelity_surrogate

        def _evaluate(self, x, out, *args, **kwargs):
            circuit = _decode(self.pools, x[0])
            out["F"] = [
                float(circuit.gate_count()),
                float(circuit.depth()),
                1.0 - self.fidelity_surrogate(circuit),
            ]

    class _BlockSampling(Sampling):
        """Individual 0 is the unmodified circuit; the rest pick random variants."""

        def __init__(self, pools):
            super().__init__()
            self.pools = pools

        def _do(self, problem, n_samples, **kwargs):
            rng = random.Random(0)
            X = np.empty((n_samples, 1), dtype=object)
            for i in range(n_samples):
                X[i, 0] = tuple(0 if i == 0 else rng.randrange(len(p)) for p in self.pools)
            return X

    class _BlockCrossover(Crossover):
        """Uniform crossover: each block's variant index comes from either parent."""

        def __init__(self, prob: float):
            super().__init__(2, 2, prob=prob)

        def _do(self, problem, X, **kwargs):
            _, n_matings, _ = X.shape
            Y = np.empty_like(X)
            rng = random.Random()
            for k in range(n_matings):
                a, b = X[0, k, 0], X[1, k, 0]
                swap = [rng.random() < 0.5 for _ in a]
                Y[0, k, 0] = tuple(y if s else x for x, y, s in zip(a, b, swap))
                Y[1, k, 0] = tuple(x if s else y for x, y, s in zip(a, b, swap))
            return Y

    class _BlockMutation(Mutation):
        """Pick one block; switch it to another existing variant, or derive a
        new variant via ``mutate_validity_preserving`` and switch to that."""

        def __init__(self, pools, prob: float):
            super().__init__(prob=prob)
            self.pools = pools

        def _do(self, problem, X, **kwargs):
            # pymoo's Mutation.do() already decides, per individual, whether to keep
            # this result or the original based on `self.prob` - _do must mutate
            # unconditionally.
            rng = random.Random()
            for i in range(len(X)):
                genome = list(X[i, 0])
                b = rng.randrange(len(genome))
                pool = self.pools[b]
                if len(pool) > 1 and rng.random() < 0.5:
                    genome[b] = rng.randrange(len(pool))
                else:
                    # ponytail: pools grow without dedup (<= pop_size * generations
                    # variants); dedupe by gate tuple if memory ever matters.
                    pool.append(mutate_validity_preserving(pool[genome[b]], rng))
                    genome[b] = len(pool) - 1
                X[i, 0] = tuple(genome)
            return X
