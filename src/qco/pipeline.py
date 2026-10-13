"""Wires Stage 1..6 with the closed-loop feedback edge (docs/architecture.md).

Novelty wiring (#31, docs/novelty_plan.md):

* N2 - every check runs on the block/window *compacted* to its active qubits
  (``_compact``), and the partitioner caps blocks at 8 qubits, so blocks of
  arbitrarily wide circuits are verified densely; phase-polynomial / explicit
  ``unverified`` results come from ``EquivalenceChecker`` beyond that.
  Unverifiable rewrites are rejected, never silently accepted.
* N1 - blocks are carved into re-synthesis-sized windows, ranked by GNN
  candidate score, and re-synthesized best-first until ``solver_budget_s``.
* N3 - each block's verified variants go to ``NSGA2Compiler.evolve(...,
  block_variants=...)``; the front is equivalent by construction.
* N4 - reward uses ``PipelineConfig.calibration``; the feedback loop stops once
  a pass no longer improves it. (Online reward into the PPO agent is #35.)
* N5 - rule-proven cancellations skip verification, confident GNN ones get the
  numeric check, the rest the full (SMT-capable) checker.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from qco.evaluation.metrics import scalarized_reward
from qco.graphs.partitioning import Block, partition_circuit
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import parse
from qco.modules.evolutionary.nsga2 import NSGA2Compiler, ParetoPoint
from qco.modules.gnn_cancellation.features import rule_based_candidates
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
    feedback_iterations: int = 1     # max closed-loop passes; stops early once reward <= 0
    resynth_config: ResynthConfig | None = None
    calibration: str = "default"     # N4: device profile for reward + RL (qco.evaluation.calibration)
    solver_budget_s: float = 5.0     # N1: total exact re-synthesis time per run
    confident_threshold: float = 0.9 # N5: GNN score above which a numeric check suffices
    # Ablations (#32): switch Modules A / B / D off; Module C off = enable_resynthesis
    # and verify_rewrites both False.
    use_rl: bool = True
    use_gnn: bool = True
    use_ea: bool = True
    # N1 window selection: "gnn" | "rule" | "random" | "exhaustive" (no budget).
    window_ranking: str = "gnn"


@dataclass(slots=True)
class PipelineResult:
    pareto_front: list[ParetoPoint]
    reward: float
    blocks: int
    verified: bool
    reverts: int = 0
    revert_events: list[RevertEvent] = field(default_factory=list)
    unverified_blocks: int = 0       # blocks where a rewrite was rejected as unverifiable
    solver_time_s: float = 0.0       # exact re-synthesis time actually spent (N1)
    reward_history: list[float] = field(default_factory=list)


def _compact(
    a: IntermediateRepresentation, b: IntermediateRepresentation
) -> tuple[IntermediateRepresentation, IntermediateRepresentation]:
    """Relabel both circuits onto their joint active qubits 0..k-1 (k <= block width)."""
    active = sorted({q for g in (*a.gates, *b.gates) for q in g.qubits})
    if len(active) == a.num_qubits:
        return a, b
    m = {q: i for i, q in enumerate(active)}
    k = max(len(active), 1)

    def remap(ir: IntermediateRepresentation) -> IntermediateRepresentation:
        gates = [type(g)(g.name, tuple(m[q] for q in g.qubits), g.params, g.tag) for g in ir.gates]
        return IntermediateRepresentation(k, gates, ir.name)

    return remap(a), remap(b)


class HybridOptimizer:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.rl = RLScheduler()
        self.rl.config.calibration = self.config.calibration
        self.gnn = GNNCancellationPredictor()
        self.checker = EquivalenceChecker()
        self.cheap_checker = EquivalenceChecker(backend="numeric")
        self.resynthesizer = BoundedResynthesizer(self.config.resynth_config)
        self.ea = NSGA2Compiler()
        self.revert_events: list[RevertEvent] = []
        self._solver_left = self.config.solver_budget_s
        self._solver_spent = 0.0
        self._unverified: set[int] = set()

    def _check(self, original, rewrite, checker: EquivalenceChecker | None = None) -> EquivalenceResult:
        return (checker or self.checker).check(*_compact(original, rewrite))

    def revert_rewrite(
        self,
        original: IntermediateRepresentation,
        proposed: IntermediateRepresentation,
        fallback: IntermediateRepresentation | None = None,
        block_idx: int = 0,
        stage: str = "stage_4",
        checker: EquivalenceChecker | None = None,
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
        result = self._check(original, proposed, checker)
        if result.equivalent:
            return proposed, True, result
        if result.method == "unverified":
            self._unverified.add(block_idx)

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
            fallback_res = self._check(original, fallback)
            if fallback_res.equivalent:
                return fallback, False, result

        # Revert to pre-rewrite original block
        return original, False, result

    # -- Stage 4 on one block -------------------------------------------------
    def _optimize_block(self, block: Block, block_idx: int = 0) -> IntermediateRepresentation:
        """Best verified variant of ``block`` (fewest gates, then depth)."""
        return self._block_variants(block, block_idx)[0]

    def _block_variants(self, block: Block, block_idx: int = 0) -> list[IntermediateRepresentation]:
        """Verified variants of ``block``, best first; ``block.ir`` is always included."""
        original = block.ir.copy()
        verify = self.config.verify_rewrites
        variants = [original]

        # Module A: reorder / commute.
        scheduled = self.rl.schedule(original) if self.config.use_rl else original
        if not verify or self.revert_rewrite(original, scheduled, None, block_idx, "module_a_schedule")[1]:
            variants.append(scheduled)
        base = variants[-1]

        # Module B: drop non-overlapping inverse pairs, gated by confidence (N5).
        drop: set[int] = set()
        applied = []
        for c in (self.gnn.predict(base) if self.config.use_gnn else []):
            if c.kind == "inverse_pair" and c.a not in drop and c.b not in drop:
                drop.update({c.a, c.b})
                applied.append(c)
        if drop:
            candidate = IntermediateRepresentation(
                base.num_qubits, [g for i, g in enumerate(base.gates) if i not in drop], base.name
            )
            proven = {(c.kind, c.gate_indices) for c in rule_based_candidates(base)}
            if not verify or all((c.kind, c.gate_indices) in proven for c in applied):
                variants.append(candidate)
            else:
                confident = min(c.score for c in applied) >= self.config.confident_threshold
                checker = self.cheap_checker if confident else self.checker
                if self.revert_rewrite(original, candidate, None, block_idx,
                                       "module_b_cancellation", checker)[1]:
                    variants.append(candidate)

        # N1: exact re-synthesis on GNN-ranked windows, under the solver budget.
        resynth = self._resynthesize_windows(min(variants, key=len), block_idx)
        if resynth is not None:
            variants.append(resynth)

        variants.sort(key=lambda v: (v.gate_count(), v.depth()))
        return variants

    def _resynthesize_windows(self, ir: IntermediateRepresentation, block_idx: int):
        """Replace windows of ``ir`` with smaller verified re-syntheses; None if nothing improved."""
        if not self.config.enable_resynthesis or self._solver_left <= 0:
            return None
        rc = self.resynthesizer.config
        windows = partition_circuit(ir, max_block_gates=rc.max_gates, max_block_qubits=rc.max_qubits)
        ranked = self._rank_windows(windows)
        replaced: dict[int, IntermediateRepresentation] = {}
        for w in ranked:
            if self._solver_left <= 0:
                break
            if not self.resynthesizer.applicable(w.ir):
                continue
            t0 = time.perf_counter()
            timeout = min(rc.timeout_s, self._solver_left)
            new_ir, _status, _ = self.resynthesizer.resynthesize(w.ir, timeout_s=timeout)
            spent = time.perf_counter() - t0
            self._solver_left -= spent
            self._solver_spent += spent
            if new_ir.gate_count() >= w.ir.gate_count():
                continue
            if not self.config.verify_rewrites or self.revert_rewrite(
                w.ir, new_ir, None, block_idx, "bounded_resynthesis"
            )[1]:
                replaced[w.index] = new_ir
        if not replaced:
            return None
        out = IntermediateRepresentation(ir.num_qubits, name=ir.name)
        for w in windows:  # windows compose in list order (partitioner contract)
            out.gates.extend(replaced.get(w.index, w.ir).gates)
        return out

    def _rank_windows(self, windows: list[Block]) -> list[Block]:
        """Order windows for re-synthesis (N1); ties broken by size."""
        mode = self.config.window_ranking
        if mode == "random":
            return random.Random(len(windows)).sample(windows, len(windows))
        if mode == "exhaustive":
            return windows
        if mode == "rule":
            score = lambda w: sum(c.score for c in rule_based_candidates(w.ir))
        elif mode == "gnn":
            # ponytail: summed GNN candidate scores (rule-based until #28's
            # window-level head lands); swap in the pooled head then.
            score = lambda w: sum(c.score for c in self.gnn.predict(w.ir, threshold=0.0))
        else:
            raise ValueError(f"unknown window_ranking {mode!r}")
        return sorted(windows, key=lambda w: (-score(w), -len(w.ir)))

    # -- full pipeline -----------------------------------------------------
    def run(self, source) -> PipelineResult:
        circuit: IntermediateRepresentation = parse(source)
        self.revert_events = []
        self._unverified = set()
        exhaustive = self.config.window_ranking == "exhaustive"
        self._solver_left = math.inf if exhaustive else self.config.solver_budget_s
        self._solver_spent = 0.0
        verified = True
        rewards: list[float] = []
        variants: list[list[IntermediateRepresentation]] = []

        for _ in range(max(self.config.feedback_iterations, 1)):
            blocks = partition_circuit(
                circuit,
                max_block_gates=self.config.max_block_gates,
                backend=self.config.partition_backend,
            )
            variants = [self._block_variants(b, idx) for idx, b in enumerate(blocks)]

            # Stitch the best variant of each block, in partition order.
            stitched = IntermediateRepresentation(num_qubits=circuit.num_qubits, name=circuit.name)
            for v in variants:
                stitched.gates.extend(v[0].gates)

            # Blocks compose by the partitioner's contract, so per-block checks
            # cover the whole circuit; the global check is a cheap extra when small.
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
                    variants = [[b.ir] for b in blocks]

            # Stage 6 -> closed loop (N4): stop once a pass stops paying off.
            rewards.append(scalarized_reward(circuit, stitched, calibration=self.config.calibration))
            circuit = stitched
            if rewards[-1] <= 0:
                break

        # Stage 5 (N3): NSGA-II recombines verified block variants -> certified front.
        if self.config.use_ea:
            front = self.ea.evolve([circuit], block_variants=variants)
        else:
            front = [self.ea.score(circuit)]
        if self.config.verify_rewrites and circuit.num_qubits <= 8:
            assert all(self.checker.check(circuit, p.circuit).equivalent for p in front), \
                "N3 invariant broken: NSGA-II front member not equivalent"
        return PipelineResult(
            pareto_front=front,
            reward=rewards[-1],
            blocks=len(variants),
            verified=verified and self.config.verify_rewrites,
            reverts=len(self.revert_events),
            revert_events=list(self.revert_events),
            unverified_blocks=len(self._unverified),
            solver_time_s=self._solver_spent,
            reward_history=rewards,
        )


def optimize(source, config: PipelineConfig | None = None) -> PipelineResult:
    return HybridOptimizer(config).run(source)

