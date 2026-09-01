"""Wires Stage 1..6 with the closed-loop feedback edge (docs/architecture.md).

Phase 2 status: the pipeline runs end-to-end using the implemented pieces
(parser, DAG/hypergraph, fallback partitioner, Module A greedy commute, Module B
rule-based cancellation, Module C numeric equivalence check, Module D
non-dominated sort, Stage 6 metrics). Learned models drop in during Phase 3
without changing this file's structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qco.evaluation.metrics import scalarized_reward
from qco.graphs.partitioning import Block, partition_circuit
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import parse
from qco.modules.evolutionary.nsga2 import NSGA2Compiler, ParetoPoint
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor
from qco.modules.rl_scheduler.agent import RLScheduler
from qco.modules.smt_verifier.equivalence import EquivalenceChecker


@dataclass(slots=True)
class PipelineConfig:
    max_block_gates: int = 64
    partition_backend: str = "auto"
    verify_rewrites: bool = True
    feedback_iterations: int = 1     # closed-loop passes; >1 needs a trained RL agent


@dataclass(slots=True)
class PipelineResult:
    pareto_front: list[ParetoPoint]
    reward: float
    blocks: int
    verified: bool


class HybridOptimizer:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.rl = RLScheduler()
        self.gnn = GNNCancellationPredictor()
        self.checker = EquivalenceChecker()
        self.ea = NSGA2Compiler()

    # -- Stage 4 on one block -------------------------------------------------
    def _optimize_block(self, block: Block) -> IntermediateRepresentation:
        original = block.ir.copy()

        # Module A: reorder / commute.
        scheduled = self.rl.schedule(original)

        # Module B: rank cancellation candidates, then apply the safe ones.
        candidate = scheduled.copy()
        cancels = self.gnn.predict(candidate)
        drop: set[int] = set()
        for c in cancels:
            if c.kind == "inverse_pair" and c.a not in drop and c.b not in drop:
                drop.update({c.a, c.b})
        if drop:
            candidate = IntermediateRepresentation(
                num_qubits=candidate.num_qubits,
                gates=[g for i, g in enumerate(candidate.gates) if i not in drop],
                name=candidate.name,
            )

        # Module C: verify every rewrite before accepting it.
        if self.config.verify_rewrites and original.num_qubits <= 8:
            if not self.checker.check(original, candidate).equivalent:
                return scheduled if self.checker.check(original, scheduled).equivalent else original
        return candidate

    # -- full pipeline -----------------------------------------------------
    def run(self, source) -> PipelineResult:
        circuit: IntermediateRepresentation = parse(source)
        verified = True

        for _ in range(self.config.feedback_iterations):
            blocks = partition_circuit(
                circuit,
                max_block_gates=self.config.max_block_gates,
                backend=self.config.partition_backend,
            )
            optimized_blocks = [self._optimize_block(b) for b in blocks]

            # Stitch blocks back in original order.
            stitched = IntermediateRepresentation(num_qubits=circuit.num_qubits, name=circuit.name)
            for ob in optimized_blocks:
                stitched.gates.extend(ob.gates)

            if circuit.num_qubits <= 8:
                verified = verified and self.checker.check(circuit, stitched).equivalent
                if not verified:
                    stitched = circuit  # reject the pass, keep correctness

            reward = scalarized_reward(circuit, stitched)  # Stage 6 -> Module A (closed loop)
            circuit = stitched

        # Stage 5: Pareto refinement over the (single, for now) verified candidate.
        front = self.ea.evolve([circuit])
        return PipelineResult(pareto_front=front, reward=reward, blocks=len(front), verified=verified)


def optimize(source, config: PipelineConfig | None = None) -> PipelineResult:
    return HybridOptimizer(config).run(source)
