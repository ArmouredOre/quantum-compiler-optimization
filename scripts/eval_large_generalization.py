#!/usr/bin/env python
"""Issue #37: zero-shot generalization of the 5-qubit policies to 40-qubit circuits.

Reuses the #26 policies (docs/rl_training/gen_runs/seed<k>_{cr,nocr}/final.json) and the same
method set, on benchmarks/circuits_40q (qft, qaoa x2, random Clifford+T, random CNOT+Pauli) plus
benchmarks/circuits/ghz_n40. One seed per process (a 40-qubit rollout takes seconds to minutes):

    python scripts/gen_circuits_40q.py
    python scripts/eval_large_generalization.py --stage run --seed 0 --out docs/rl_training/large &
    python scripts/eval_large_generalization.py --stage run --seed 1 --out docs/rl_training/large
    python scripts/eval_large_generalization.py --stage run --seed 2 --out docs/rl_training/large
    python scripts/eval_large_generalization.py --stage report --seeds 0 1 2 --out docs/rl_training/large

Numeric equivalence checking is capped at 8 qubits, so nothing here is equivalence-checked; the
end-to-end run (scripts/eval_closed_loop.py / HybridOptimizer) verifies 40-qubit outputs block by block.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from qco.ir.parser import parse
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import Policies, load_benchmarks, study, summarize, to_markdown
from qco.modules.rl_scheduler.plots import grouped_bars_svg
from qco.modules.rl_scheduler.ppo import load_policy


def policies_for(seed: int, gen_runs: Path) -> Policies:
    cr, payload = load_policy(gen_runs / f"seed{seed}_cr" / "final.json")
    nocr, payload_no = load_policy(gen_runs / f"seed{seed}_nocr" / "final.json")
    return Policies(cr, RLConfig.from_dict(payload["config"]), nocr, RLConfig.from_dict(payload_no["config"]))


def circuits(a):
    cs = load_benchmarks(a.circuits_40q, 64)
    ghz = parse(f"{a.circuits}/ghz_n40.qasm")
    ghz.name = "ghz_n40"
    return cs + [ghz]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run", "report", "e2e"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gen-runs", default="docs/rl_training/gen_runs")
    ap.add_argument("--circuits", default="benchmarks/circuits")
    ap.add_argument("--circuits-40q", default="benchmarks/circuits_40q")
    ap.add_argument("--out", default="docs/rl_training/large")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = circuits(a)

    if a.stage == "run":
        res = study(cs, {a.seed: policies_for(a.seed, Path(a.gen_runs))}, verify=None, log=lambda m: print(m, flush=True))
        (out / f"run_seed{a.seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    if a.stage == "e2e":
        from qco.modules.rl_scheduler.closed_loop import evaluate_end_to_end

        rows = {}
        for label, path in (("stub (untrained)", None), ("trained agent (default)", "auto")):
            rows[label] = evaluate_end_to_end(cs, rl_policy_path=path)
            print(label, [(r.name, round(r.reward, 4), r.verified, r.seconds) for r in rows[label]], flush=True)
        md = ["# 40-qubit end-to-end runs through `HybridOptimizer` (issue #37)", "",
              "Module A in the loop (stub or the committed default policy `models/rl_ppo_5q.json`), rule-based Module B, Module C "
              "verification per block, no SMT re-synthesis, no NSGA-II. Reward = Stage 6 `scalarized_reward(original, output)` "
              "with default weights; `verified` = the pipeline's block-wise equivalence verdict.", "",
              "| Circuit | gates | " + " | ".join(f"{k}: reward / depth % / gate % / s / verified (blocks, undecided)" for k in rows) + " |",
              "|---|---|" + "---|" * len(rows)]
        for i, c in enumerate(cs):
            cells = []
            for k in rows:
                r = rows[k][i]
                cells.append(f"{r.reward:.4f} / {r.depth_red_pct:.1f} / {r.gate_red_pct:.1f} / {r.seconds:.0f} / {r.verified} ({r.blocks}, {r.unverified_blocks})")
            md.append(f"| {c.name} | {c.gate_count()} | " + " | ".join(cells) + " |")
        (out / "e2e_40q.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        (out / "e2e_40q.json").write_text(json.dumps({k: [asdict(r) for r in v] for k, v in rows.items()}, indent=1), encoding="utf-8")
        print("\n".join(md))
        return 0

    parts = [json.loads((out / f"run_seed{s}.json").read_text()) for s in a.seeds]
    result = {"circuits": parts[0]["circuits"], "runs": {s: p["runs"][str(s)] for s, p in zip(a.seeds, parts)}}
    summary = summarize(result)
    info = ("Policies from #26 (trained on generated 5-qubit circuits, 100000 timesteps, `commutation=full`), run zero-shot on "
            "40-qubit circuits: `benchmarks/circuits_40q/` (qft, qaoa p1/p2, random Clifford+T, random CNOT+Pauli) and `ghz_n40`. "
            "Not equivalence-checked here (numeric cap is 8 qubits); see the end-to-end table for block-wise verification.")
    (out / "generalization_40q.json").write_text(json.dumps({"summary": summary, "result": result}, indent=1), encoding="utf-8")
    md = to_markdown(result, summary, info, title="Zero-shot generalization to 40 qubits (issue #37)")
    pb = summary["per_bucket"]["40"]
    methods = [m for m in ("stub", "cr_only", "greedy", "agent_cr", "agent_cr_x", "agent_nocr", "agent_cr_off") if m in pb]
    labels = {"stub": "stub", "cr_only": "CR only", "greedy": "greedy+CR", "agent_cr": "PPO+CR", "agent_cr_x": "PPO+CR scaled", "agent_nocr": "PPO no CR", "agent_cr_off": "CR off at test"}
    grouped_bars_svg("40-qubit zero-shot: depth / gate-count reduction %", ["depth", "gate count"],
                     {labels[m]: [pb[m]["depth_red_pct"], pb[m]["gate_red_pct"]] for m in methods}, "% reduction",
                     {labels[m]: [pb[m]["depth_red_std_over_seeds"], 0.0] for m in methods}, out / "generalization_40q.svg")
    md += "\n![40-qubit generalization](generalization_40q.svg)\n"
    (out / "generalization_40q.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
