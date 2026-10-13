"""Module A — PPO training loop: logging, checkpoints, held-out evaluation, curves.

``train(config, out_dir)`` writes (all under ``out_dir``):

* ``metrics.csv``   one row per PPO iteration (return, depth/gate reduction, losses, ...)
* ``latest.json``   policy + config + progress (every ``checkpoint_every`` iterations)
* ``best.json``     best held-out policy so far
* ``final.json``    policy at the end of training
* ``curves.svg``    reward / depth / gate-count curves (pure-Python SVG, no matplotlib)
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from qco.modules.rl_scheduler.baselines import evaluate, greedy_depth_chooser, random_chooser
from qco.modules.rl_scheduler.circuits import sample_circuits
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.ppo import PPOPolicy, PPOTrainer, load_policy, save_policy

CSV_FIELDS = [
    "iteration", "timesteps", "episodes", "mean_return", "mean_ep_len",
    "train_depth_red_pct", "train_gate_red_pct", "eval_depth_red_pct", "eval_gate_red_pct",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "seconds",
]


def build_pools(config: RLConfig):
    q = config.train_qubits[0]
    train, evals = [], []
    for k, n in enumerate(config.train_qubits):
        train += sample_circuits(n, max(config.n_train_circuits // len(config.train_qubits), 1), 1000 + k, config.circuit_families, config.circuit_layers)
    evals = sample_circuits(q, config.n_eval_circuits, 777, config.circuit_families, config.circuit_layers)
    return train, evals


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def train(
    config: RLConfig,
    out_dir: str | Path | None = None,
    *,
    policy: PPOPolicy | None = None,
    resume: bool = False,
    log=print,
) -> tuple[PPOPolicy, list[dict]]:
    """Train the built-in PPO. Returns ``(policy, history)``; ``history`` mirrors metrics.csv."""
    config.validate()
    if config.algo != "ppo":
        raise NotImplementedError(f"algo={config.algo!r} is not implemented (only 'ppo')")
    out = Path(out_dir) if out_dir else None
    start_iter = 0
    if resume and out and (out / "latest.json").exists():
        policy, payload = load_policy(out / "latest.json", config.numeric_backend)
        start_iter = int(payload.get("iteration", 0))
    trainer = PPOTrainer(config, policy)
    if resume and out and start_iter:
        trainer.timesteps = int(payload.get("timesteps", 0))
        trainer._cursor = start_iter
    train_pool, eval_pool = build_pools(config)

    history: list[dict] = []
    writer = fh = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        fh = open(out / "metrics.csv", "a" if resume and start_iter else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not (resume and start_iter):
            writer.writeheader()

    best_eval = float("-inf")
    it = start_iter
    target = trainer.timesteps + config.total_timesteps if resume else config.total_timesteps
    try:
        while trainer.timesteps < target:
            t0 = time.time()
            batch, episodes = trainer.collect(train_pool)
            stats = trainer.update(batch)
            it += 1
            row = {
                "iteration": it, "timesteps": trainer.timesteps, "episodes": len(episodes),
                "mean_return": _mean([e["return"] for e in episodes]),
                "mean_ep_len": _mean([e["length"] for e in episodes]),
                "train_depth_red_pct": _mean([e["depth_red_pct"] for e in episodes]),
                "train_gate_red_pct": _mean([e["gate_red_pct"] for e in episodes]),
                "eval_depth_red_pct": "", "eval_gate_red_pct": "",
                **stats,
            }
            if config.eval_every and it % config.eval_every == 0:
                ev = evaluate(eval_pool, config, policy=trainer.policy)
                row["eval_depth_red_pct"], row["eval_gate_red_pct"] = ev["depth_red_pct"], ev["gate_red_pct"]
                score = ev["depth_red_pct"] + ev["gate_red_pct"]
                if out and score > best_eval:
                    best_eval = score
                    save_policy(trainer.policy, out / "best.json", {"config": config.to_dict(), "iteration": it, "timesteps": trainer.timesteps, "eval": ev})
            row["seconds"] = time.time() - t0
            history.append(row)
            if writer:
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
                fh.flush()
            if log:
                log(f"[it {it:3d}] steps={row['timesteps']:6d} return={row['mean_return']:+.3f} "
                    f"depth-{row['train_depth_red_pct']:.1f}% gates-{row['train_gate_red_pct']:.1f}% H={row['entropy']:.2f}")
            if out and config.checkpoint_every and it % config.checkpoint_every == 0:
                save_policy(trainer.policy, out / "latest.json", {"config": config.to_dict(), "iteration": it, "timesteps": trainer.timesteps})
    finally:
        if fh:
            fh.close()
    if out:
        extra = {"config": config.to_dict(), "iteration": it, "timesteps": trainer.timesteps}
        save_policy(trainer.policy, out / "latest.json", extra)
        save_policy(trainer.policy, out / "final.json", extra)
        write_curves_svg(history, out / "curves.svg")
    return trainer.policy, history


def compare(config: RLConfig, policy: PPOPolicy, *, seed: int = 0, verify: bool = False) -> dict:
    """Held-out comparison: CR only / random / greedy-depth / trained agent."""
    _, pool = build_pools(config)
    rng = random.Random(seed)
    return {
        "cr_only": evaluate(pool, config, cr_only=True, verify=verify),
        "random": evaluate(pool, config, chooser=random_chooser(rng), verify=verify),
        "greedy_depth": evaluate(pool, config, chooser=greedy_depth_chooser(rng), verify=verify),
        "agent": evaluate(pool, config, policy=policy, verify=verify),
    }


# ---------------------------------------------------------------------------
# Curves (pure-Python SVG)
# ---------------------------------------------------------------------------

def _panel(title, xs, series, x0, y0, w, h, ylabel):
    pts_all = [v for _, ys in series for v in ys if v is not None]
    if not xs or not pts_all:
        return ""
    lo, hi = min(pts_all), max(pts_all)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    xmin, xmax = min(xs), max(xs) or 1

    def sx(x): return x0 + (x - xmin) / max(xmax - xmin, 1) * w
    def sy(y): return y0 + h - (y - lo) / (hi - lo) * h

    s = [f'<text x="{x0}" y="{y0-10}" font-size="13" font-weight="600" fill="#222">{title}</text>',
         f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#bbb"/>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        s.append(f'<line x1="{x0}" x2="{x0+w}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#eee"/>')
        s.append(f'<text x="{x0-6}" y="{sy(v)+4:.1f}" font-size="10" text-anchor="end" fill="#555">{v:.2f}</text>')
    s.append(f'<text x="{x0+w/2}" y="{y0+h+28}" font-size="11" text-anchor="middle" fill="#555">timesteps ({ylabel})</text>')
    colors = ["#2a6fdb", "#d9822b", "#2e9d5b"]
    for n, (name, ys) in enumerate(series):
        p = [(sx(x), sy(y)) for x, y in zip(xs, ys) if y is not None]
        if len(p) > 1:
            s.append(f'<polyline fill="none" stroke="{colors[n%3]}" stroke-width="2" points="{" ".join(f"{a:.1f},{b:.1f}" for a,b in p)}"/>')
        elif p:
            s.append(f'<circle cx="{p[0][0]:.1f}" cy="{p[0][1]:.1f}" r="3" fill="{colors[n%3]}"/>')
        s.append(f'<text x="{x0+8+n*110}" y="{y0+14}" font-size="10" fill="{colors[n%3]}">{name}</text>')
    return "\n".join(s)


def write_curves_svg(history: list[dict], path: str | Path) -> None:
    xs = [r["timesteps"] for r in history]

    def col(k):
        return [(r[k] if r[k] != "" else None) for r in history]

    W, H = 1000, 320
    panels = [
        _panel("Episode return (proxy reward)", xs, [("train", col("mean_return"))], 60, 40, 270, 220, "reward"),
        _panel("Depth reduction %", xs, [("train", col("train_depth_red_pct")), ("eval", col("eval_depth_red_pct"))], 385, 40, 270, 220, "%"),
        _panel("Gate-count reduction %", xs, [("train", col("train_gate_red_pct")), ("eval", col("eval_gate_red_pct"))], 710, 40, 270, 220, "%"),
    ]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">'
           f'<rect width="{W}" height="{H}" fill="white"/>' + "\n".join(panels) + "</svg>")
    Path(path).write_text(svg, encoding="utf-8")
