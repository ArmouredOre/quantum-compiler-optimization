"""Wires Stage 1..6 with the closed-loop feedback edge (docs/architecture.md).

Phase 2 status: the pipeline runs end-to-end using the implemented pieces
(parser, DAG/hypergraph, fallback partitioner, Module A greedy commute, Module B
rule-based cancellation, Module C numeric equivalence check, Module D
non-dominated sort, Stage 6 metrics). Learned models drop in during Phase 3
without changing this file's structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any

from qco.evaluation.metrics import scalarized_reward
from qco.graphs.partitioning import Block, partition_circuit
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import parse
from qco.modules.evolutionary.nsga2 import NSGA2Compiler, ParetoPoint
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor
from qco.modules.rl_scheduler.agent import RLScheduler
from qco.modules.smt_verifier.bounded_optimizer import BoundedResynthesizer, ResynthConfig
from qco.modules.smt_verifier.equivalence import EquivalenceChecker, EquivalenceResult


@dataclass(slots=True)
class RevertEvent:
    """Audit log entry recorded whenever a proposed rewrite violates equivalence."""
    block_index: int
    stage: str
    reason: str
    pre_rewrite: IntermediateRepresentation
    rejected_rewrite: IntermediateRepresentation
    counterexample: dict[str, Any] | None = None


@dataclass(slots=True)
class PipelineConfig:
    max_block_gates: int = 64
    partition_backend: str = "auto"
    verify_rewrites: bool = True
    enable_resynthesis: bool = True
    feedback_iterations: int = 1     # closed-loop passes; >1 needs a trained RL agent
    resynth_config: ResynthConfig | None = None


@dataclass(slots=True)
class PipelineResult:
    pareto_front: list[ParetoPoint]
    reward: float
    blocks: int
    verified: bool
    reverts: int = 0
    revert_events: list[RevertEvent] = field(default_factory=list)


class HybridOptimizer:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.rl = RLScheduler()
        self.gnn = GNNCancellationPredictor()
        self.checker = EquivalenceChecker()
        self.resynthesizer = BoundedResynthesizer(self.config.resynth_config)
        self.ea = NSGA2Compiler()
        self.revert_events: list[RevertEvent] = []

    def revert_rewrite(
        self,
        original: IntermediateRepresentation,
        proposed: IntermediateRepresentation,
        fallback: IntermediateRepresentation | None = None,
        block_idx: int = 0,
        stage: str = "stage_4",
    ) -> tuple[IntermediateRepresentation, bool, EquivalenceResult]:
        """Formal revert protocol contract (Sprint 3 / Module C verification invariant).

        Evaluates proposed rewrite against pre-rewrite original block.
        If equivalence holds:
            Accepts proposed rewrite.
        If equivalence check fails:
            Rejects proposed rewrite, logs counterexample and reverts to fallback
            (or original pre-rewrite block if fallback also fails or is omitted).

        Returns:
            (restored_or_accepted_block, accepted: bool, EquivalenceResult)
        """
        result = self.checker.check(original, proposed)
        if result.equivalent:
            return proposed, True, result

        # Proposed rewrite rejected! Execute revert protocol.
        self.revert_events.append(
            RevertEvent(
                block_index=block_idx,
                stage=stage,
                reason=result.detail,
                pre_rewrite=original.copy(),
                rejected_rewrite=proposed.copy(),
                counterexample=result.counterexample,
            )
        )

        # Check fallback (e.g. scheduled block from Module A)
        if fallback is not None:
            fallback_res = self.checker.check(original, fallback)
            if fallback_res.equivalent:
                return fallback, False, result

        # Revert to pre-rewrite original block
        return original, False, result

    # -- Stage 4 on one block -------------------------------------------------
    def _optimize_block(self, block: Block, block_idx: int = 0) -> IntermediateRepresentation:
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
        current = original
        if self.config.verify_rewrites and original.num_qubits <= 8:
            accepted_block, accepted, result = self.revert_rewrite(
                original=original,
                proposed=candidate,
                fallback=scheduled,
                block_idx=block_idx,
                stage="module_b_cancellation",
            )
            current = accepted_block
        else:
            current = candidate

        # Stage 4 / Module C: Bounded-window exact re-synthesis in the loop (Issue #33)
        if self.config.enable_resynthesis and self.resynthesizer.applicable(current):
            resynth_ir, status, solve_s = self.resynthesizer.resynthesize(current)
            if resynth_ir.gate_count() < current.gate_count():
                if self.config.verify_rewrites and original.num_qubits <= 8:
                    current, accepted, result = self.revert_rewrite(
                        original=original,
                        proposed=resynth_ir,
                        fallback=current,
                        block_idx=block_idx,
                        stage="bounded_resynthesis",
                    )
                else:
                    current = resynth_ir

        return current

    # -- full pipeline -----------------------------------------------------
    def run(self, source) -> PipelineResult:
        circuit: IntermediateRepresentation = parse(source)
        verified = True
        self.revert_events = []

        for _ in range(self.config.feedback_iterations):
            blocks = partition_circuit(
                circuit,
                max_block_gates=self.config.max_block_gates,
                backend=self.config.partition_backend,
            )
            optimized_blocks = [self._optimize_block(b, idx) for idx, b in enumerate(blocks)]

            # Stitch blocks back in original order.
            stitched = IntermediateRepresentation(num_qubits=circuit.num_qubits, name=circuit.name)
            for ob in optimized_blocks:
                stitched.gates.extend(ob.gates)

            if circuit.num_qubits <= 8:
                stitch_res = self.checker.check(circuit, stitched)
                if not stitch_res.equivalent:
                    self.revert_events.append(
                        RevertEvent(
                            block_index=-1,
                            stage="block_stitching",
                            reason=stitch_res.detail,
                            pre_rewrite=circuit.copy(),
                            rejected_rewrite=stitched.copy(),
                            counterexample=stitch_res.counterexample,
                        )
                    )
                    verified = False
                    stitched = circuit  # reject the pass, keep correctness

            reward = scalarized_reward(circuit, stitched)  # Stage 6 -> Module A (closed loop)
            circuit = stitched

        # Stage 5: NSGA-II searches gate-sequence variants of the verified candidate.
        front = self.ea.evolve([circuit])
        if self.config.verify_rewrites and circuit.num_qubits <= 8:
            verified_front = []
            for p in front:
                p_res = self.checker.check(circuit, p.circuit)
                if p_res.equivalent:
                    verified_front.append(p)
                else:
                    self.revert_events.append(
                        RevertEvent(
                            block_index=-1,
                            stage="nsga2_evolution",
                            reason=p_res.detail,
                            pre_rewrite=circuit.copy(),
                            rejected_rewrite=p.circuit.copy(),
                            counterexample=p_res.counterexample,
                        )
                    )
            front = verified_front
            if not front:
                front = [self.ea.score(circuit)]
        return PipelineResult(
            pareto_front=front,
            reward=reward,
            blocks=len(front),
            verified=verified,
            reverts=len(self.revert_events),
            revert_events=list(self.revert_events),
        )


def optimize(source, config: PipelineConfig | None = None) -> PipelineResult:
    return HybridOptimizer(config).run(source)

