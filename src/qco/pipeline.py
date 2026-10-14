"""Wires Stage 1..6 with the closed-loop feedback edge (docs/architecture.md).

Novelty wiring (#31, docs/novelty_plan.md):

* N2 - every check runs on the block/window *compacted* to its active qubits
  (``_compact``), and the partitioner caps blocks at 8 qubits, so blocks of
  arbitrarily wide circuits are verified densely; phase-polynomial / explicit
  ``unverified`` results come from ``EquivalenceChecker`` beyond that.
  Unverifiable rewrites are rejected, never silently accepted.
* N1 - blocks are carved into re-synthesis-sized windows, ranked by the GNN
  window head's calibrated P(reducible) (#34; summed pair scores when no
  trained checkpoint is loaded), and re-synthesized best-first until
  ``solver_budget_s``.
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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qco.evaluation.metrics import scalarized_reward
from qco.graphs.partitioning import Block, partition_circuit
from qco.ir.intermediate_representation import Gate, IntermediateRepresentation
from qco.ir.parser import parse
from qco.modules.evolutionary.nsga2 import NSGA2Compiler, ParetoPoint
from qco.modules.gnn_cancellation.features import rule_based_candidates
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor
from qco.modules.rl_scheduler.agent import RLScheduler
from qco.modules.smt_verifier.bounded_optimizer import BoundedResynthesizer, ResynthConfig
from qco.modules.smt_verifier.equivalence import EquivalenceChecker, EquivalenceResult

# Assumes this file is src/qco/pipeline.py; adjust to wherever the checkpoint is committed.
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "module_b.pt"


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
    # #34: "auto" = DEFAULT_CHECKPOINT if present (else rule-based); None = rule-based
    # fallback; any other string = explicit checkpoint path (errors are raised, never hidden).
    gnn_checkpoint: str | None = "auto"


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


def apply_cancellations(ir: IntermediateRepresentation, candidates) -> tuple[IntermediateRepresentation, list]:
    """Apply a non-overlapping subset of Module B candidates (in the given order).

    ``inverse_pair`` / ``identity_chain`` drop their gates; ``rotation_merge``
    replaces its run with one rotation at the first index whose angles are the
    exact sum (no wrapping - crz(t + 2*pi) != crz(t)). Rotation runs are
    wire-adjacent (nothing else on that support in between), so this is exact.
    """
    used: set[int] = set()
    drop: set[int] = set()
    merged: dict[int, Gate] = {}
    applied = []
    for c in candidates:
        idx = c.gate_indices
        if used & set(idx):
            continue
        if c.kind in ("inverse_pair", "identity_chain"):
            drop.update(idx)
        elif c.kind == "rotation_merge":
            first = ir.gates[idx[0]]
            params = tuple(sum(ir.gates[k].params[i] for k in idx) for i in range(len(first.params)))
            merged[idx[0]] = Gate(first.name, first.qubits, params, first.tag)
            drop.update(idx[1:])
        else:
            continue
        used.update(idx)
        applied.append(c)
    gates = [merged.get(i, g) for i, g in enumerate(ir.gates) if i not in drop]
    return IntermediateRepresentation(ir.num_qubits, gates, ir.name), applied


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


def _idx(c) -> tuple[int, ...]:
    """Gate indices a candidate touches; GNN-built inverse pairs carry () so fall back to (a, b)."""
    idx = tuple(c.gate_indices or ())
    return idx if idx else (c.a, c.b)


def _with_indices(c):
    """``c`` itself if it has gate_indices, else a duck-typed copy that does (for apply_cancellations)."""
    if c.gate_indices:
        return c
    return SimpleNamespace(a=c.a, b=c.b, kind=c.kind, score=c.score, gate_indices=_idx(c))


def _load_gnn(spec: str | None) -> GNNCancellationPredictor:
    """Trained predictor from a checkpoint, or the untrained (rule-based) one."""
    if spec is None:
        return GNNCancellationPredictor()
    path = DEFAULT_CHECKPOINT if spec == "auto" else Path(spec)
    if spec == "auto" and not path.exists():
        return GNNCancellationPredictor()
    try:
        return GNNCancellationPredictor().load(path)
    except Exception as exc:
        if spec != "auto":
            raise
        warnings.warn(
            f"GNN checkpoint {path} unusable ({exc!r}); using rule-based candidates",
            stacklevel=2,
        )
        return GNNCancellationPredictor()


def _window_spans(ir: IntermediateRepresentation, windows) -> list[tuple[int, int]] | None:
    """(start, end) of each window in ir.gates if the windows tile ir in order, else None."""
    if any(len(w.ir) == 0 for w in windows):
        return None
    key = lambda gs: [(g.name, g.qubits, g.params) for g in gs]  # noqa: E731
    if key([g for w in windows for g in w.ir.gates]) != key(ir.gates):
        return None
    spans, s = [], 0
    for w in windows:
        spans.append((s, s + len(w.ir)))
        s += len(w.ir)
    return spans


class HybridOptimizer:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.rl = RLScheduler()
        self.rl.config.calibration = self.config.calibration
        self.gnn = _load_gnn(self.config.gnn_checkpoint)
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

        # Module B (N5): rule-proven rewrites first - no check needed, exactly the rule-based
        # fallback's behaviour. GNN extras go on top and are verified, so they can only help.
        if self.config.use_gnn:
            all_rules = rule_based_candidates(base)
            proven = {(c.kind, _idx(c)) for c in all_rules}
            rules = [_with_indices(c) for c in sorted(
                (c for c in all_rules if c.score >= 0.5), key=lambda c: -c.score)]
            safe_ir, safe = apply_cancellations(base, rules)
            if safe:
                variants.append(safe_ir)
            extras = [_with_indices(c)
                      for c in self.gnn.rank_candidates(base, include_rules=False).pairs
                      if (c.kind, _idx(c)) not in proven]
            if extras:
                candidate, applied = apply_cancellations(base, rules + extras)
                new = [c for c in applied if (c.kind, _idx(c)) not in proven]
                if new:
                    if not verify:
                        variants.append(candidate)
                    else:
                        confident = min(c.score for c in new) >= self.config.confident_threshold
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
        ranked = self._rank_windows(ir, windows)
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

    def _rank_windows(self, ir: IntermediateRepresentation, windows: list[Block]) -> list[Block]:
        """Order windows for re-synthesis (N1); ties broken by size."""
        mode = self.config.window_ranking
        if mode == "random":
            return random.Random(len(windows)).sample(windows, len(windows))
        if mode == "exhaustive":
            return windows
        if mode == "gnn" and self.gnn.has_window_head:
            # #34: calibrated P(reducible) from the window head.
            spans = _window_spans(ir, windows)
            if spans is not None:  # full-circuit context, as in training
                probs = self.gnn.score_windows(ir, spans)
            else:
                probs = [
                    self.gnn.score_windows(w.ir, [(0, len(w.ir))])[0] if len(w.ir) else 0.0
                    for w in windows
                ]
                # #28 out-of-fold results: rule evidence + GNN tie-break beat either alone
                # (and the head alone is worse on unseen families), so rules rank, GNN breaks ties.
            rule = [sum(c.score for c in rule_based_candidates(w.ir)) for w in windows]
            order = sorted(range(len(windows)),
                           key=lambda i: (-rule[i], -probs[i], -len(windows[i].ir)))
            return [windows[i] for i in order]
        if mode == "rule":
            score = lambda w: sum(c.score for c in rule_based_candidates(w.ir))  # noqa: E731
        elif mode == "gnn":
            # No trained window head: summed pair scores (rule-based when untrained).
            score = lambda w: sum(c.score for c in self.gnn.predict(w.ir, threshold=0.0))  # noqa: E731
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
