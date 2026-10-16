"""Module A — closed-loop reward from Stage 6 (issue #35).

Three pieces:

``RewardLog``
    A consumer for ``PipelineConfig.reward_hook``: records every ``RewardEvent`` the
    pipeline emits (per block and per feedback iteration) so Module A can inspect,
    export (CSV) and summarise the Stage 6 signal. Pass an instance as the hook.

``PipelineReward``
    The *real* reward used to train the agent: ``fn(original, scheduled)`` runs the
    downstream pipeline (blocks -> Module B rewrites -> verification, Module A
    switched off because ``scheduled`` already is its output) on the agent's schedule
    and returns Stage 6 ``scalarized_reward(original, pipeline_output)``. Plugged into
    ``SchedulerEnv(terminal_reward_fn=...)`` via ``RLConfig(reward_source="pipeline")``.
    Results are memoised (the CR-only reference run repeats every episode).

``evaluate_end_to_end``
    Runs the full ``HybridOptimizer`` with a given policy and reports real reward and
    depth / gate-count reductions per circuit - the comparison used to judge proxy- vs
    pipeline-trained policies.

Defaults keep reward cheap and deterministic: rule-based Module B (no GNN checkpoint),
no SMT re-synthesis (``resynthesis=True`` adds it, costing ~``solver_budget_s`` per
call), numeric verification on <= 8 qubits as in the pipeline.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from qco.evaluation.metrics import scalarized_reward
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.pipeline import HybridOptimizer, PipelineConfig, RewardEvent


def _key(ir: IntermediateRepresentation) -> tuple:
    return (ir.num_qubits, tuple((g.name, g.qubits, g.params) for g in ir.gates))


def _pct(a: int, b: int) -> float:
    return 100.0 * (a - b) / a if a else 0.0


# ---------------------------------------------------------------------------
# hook consumer
# ---------------------------------------------------------------------------

class RewardLog:
    """Callable ``reward_hook`` that records ``RewardEvent`` s as light-weight rows."""

    FIELDS = ["iteration", "block_index", "reward", "gates_before", "gates_after", "depth_before", "depth_after", "calibration"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def __call__(self, event: RewardEvent) -> None:
        self.rows.append(
            {
                "iteration": event.iteration,
                "block_index": event.block_index,
                "reward": event.reward,
                "gates_before": event.before.gate_count(),
                "gates_after": event.after.gate_count(),
                "depth_before": event.before.depth(),
                "depth_after": event.after.depth(),
                "calibration": event.calibration,
            }
        )

    def clear(self) -> None:
        self.rows.clear()

    def summary(self) -> dict:
        blocks = [r["reward"] for r in self.rows if r["block_index"] >= 0]
        whole = [r["reward"] for r in self.rows if r["block_index"] == -1]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        return {
            "events": len(self.rows),
            "block_events": len(blocks),
            "mean_block_reward": mean(blocks),
            "iteration_events": len(whole),
            "mean_iteration_reward": mean(whole),
            "last_iteration_reward": whole[-1] if whole else 0.0,
        }

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)


# ---------------------------------------------------------------------------
# real reward
# ---------------------------------------------------------------------------

def _pipeline_config(calibration: str, resynthesis: bool, hook=None, **kw) -> PipelineConfig:
    return PipelineConfig(
        calibration=calibration,
        enable_resynthesis=resynthesis,
        use_ea=False,
        gnn_checkpoint=None,          # rule-based Module B: deterministic, no torch needed
        feedback_iterations=1,
        reward_hook=hook,
        **kw,
    )


class PipelineReward:
    """``fn(original, scheduled) -> Stage 6 reward of the pipeline run on ``scheduled``."""

    def __init__(self, calibration: str = "default", resynthesis: bool = False, hook=None, cache_size: int = 8192):
        self.calibration = calibration
        self.hook = hook
        self._opt = HybridOptimizer(_pipeline_config(calibration, resynthesis, hook, use_rl=False))
        self._cache: dict[tuple, float] = {}
        self._cache_size = cache_size
        self.calls = 0
        self.cache_hits = 0

    def output(self, scheduled: IntermediateRepresentation) -> IntermediateRepresentation:
        """The circuit the downstream pipeline produces from ``scheduled``."""
        return self._opt.run(scheduled).pareto_front[0].circuit

    def __call__(self, original: IntermediateRepresentation, scheduled: IntermediateRepresentation) -> float:
        self.calls += 1
        k = (_key(original), _key(scheduled))
        if k in self._cache:
            self.cache_hits += 1
            return self._cache[k]
        value = scalarized_reward(original, self.output(scheduled), calibration=self.calibration)
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[k] = value
        return value


_SHARED: dict[tuple, PipelineReward] = {}


def get_pipeline_reward(calibration: str = "default", resynthesis: bool = False) -> PipelineReward:
    """Process-wide cached ``PipelineReward`` (building a ``HybridOptimizer`` is not free)."""
    k = (calibration, resynthesis)
    if k not in _SHARED:
        _SHARED[k] = PipelineReward(calibration, resynthesis)
    return _SHARED[k]


# ---------------------------------------------------------------------------
# end-to-end evaluation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class E2EResult:
    name: str
    qubits: int
    reward: float
    depth_red_pct: float
    gate_red_pct: float
    seconds: float
    verified: bool | None = None   # pipeline's block-wise equivalence verdict (None = not recorded)
    blocks: int = 0
    unverified_blocks: int = 0     # blocks whose rewrite could not be decided by the checker


def evaluate_end_to_end(
    circuits: list[IntermediateRepresentation],
    *,
    policy=None,
    rl_config=None,
    rl_policy_path: str | None = None,
    calibration: str = "default",
    resynthesis: bool = False,
    hook=None,
    **pipeline_kw,
) -> list[E2EResult]:
    """Run the *full* ``HybridOptimizer`` (Module A in the loop) on each circuit.

    Module A is the untrained greedy stand-in unless ``policy`` (an in-memory
    ``PPOPolicy`` plus its ``rl_config``) or ``rl_policy_path`` (a saved checkpoint) is
    given. ``reward`` is Stage 6 ``scalarized_reward(original, pipeline output)``.
    """
    cfg = _pipeline_config(calibration, resynthesis, hook, rl_policy_path=rl_policy_path, **pipeline_kw)
    opt = HybridOptimizer(cfg)
    if policy is not None:
        opt.rl._model = policy
        if rl_config is not None:
            opt.rl.config = rl_config
        opt.rl.config.calibration = calibration
    out = []
    for c in circuits:
        t = time.time()
        res = opt.run(c)
        final = res.pareto_front[0].circuit
        out.append(
            E2EResult(
                c.name or "circuit",
                c.num_qubits,
                scalarized_reward(c, final, calibration=calibration),
                _pct(c.depth(), final.depth()),
                _pct(c.gate_count(), final.gate_count()),
                time.time() - t,
                bool(res.verified),
                res.blocks,
                res.unverified_blocks,
            )
        )
    return out


def summarize_e2e(results: list[E2EResult]) -> dict:
    n = max(len(results), 1)
    return {
        "n": len(results),
        "reward": sum(r.reward for r in results) / n,
        "depth_red_pct": sum(r.depth_red_pct for r in results) / n,
        "gate_red_pct": sum(r.gate_red_pct for r in results) / n,
    }


# ---------------------------------------------------------------------------
# reward curves (pure-Python SVG, same style as training.write_curves_svg)
# ---------------------------------------------------------------------------

def mean_curve(histories: list[list[dict]], key: str) -> tuple[list[int], list[float | None]]:
    """Seed-averaged curve of ``key`` aligned by iteration (``None`` where not logged)."""
    n = min((len(h) for h in histories), default=0)
    xs, ys = [], []
    for i in range(n):
        xs.append(int(sum(h[i]["timesteps"] for h in histories) / len(histories)))
        vals = [h[i][key] for h in histories if h[i][key] not in ("", None)]
        ys.append(sum(vals) / len(vals) if vals else None)
    return xs, ys


def write_reward_curves(proxy_runs: list[list[dict]], pipeline_runs: list[list[dict]], path: str | Path) -> None:
    """Two panels: end-to-end Stage 6 reward on held-out circuits, and the real terminal reward seen in training."""
    from qco.modules.rl_scheduler.training import _panel

    xp, yp = mean_curve(proxy_runs, "eval_pipeline_reward")
    xr, yr = mean_curve(pipeline_runs, "eval_pipeline_reward")
    xt, yt = mean_curve(pipeline_runs, "mean_terminal_reward")
    W, H = 700, 320
    panels = [
        _panel("End-to-end Stage 6 reward (held-out, seed mean)", xr or xp, [("proxy-trained", yp), ("pipeline-trained", yr)], 60, 40, 270, 220, "reward"),
        _panel("Real terminal reward in training (pipeline run)", xt, [("pipeline-trained", yt)], 385, 40, 270, 220, "reward vs CR-only"),
    ]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">'
           f'<rect width="{W}" height="{H}" fill="white"/>' + "\n".join(panels) + "</svg>")
    Path(path).write_text(svg, encoding="utf-8")
