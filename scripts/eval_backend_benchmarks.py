#!/usr/bin/env python
"""Issue #37 / #29: per-backend policy results on the benchmark circuits (5-40 qubits).

The #29 matrix (docs/rl_training/backend_matrix.md) used 60 held-out 5-qubit *generated* circuits.
This reuses the saved per-backend policies (models/backend/<profile>_seed<k>.json, no retraining) and
asks the same train-on-A / test-on-B question on the committed benchmarks, one qubit bucket at a time.

    python scripts/eval_backend_benchmarks.py --bucket 5-6    --out docs/rl_training/backend_benchmarks
    python scripts/eval_backend_benchmarks.py --bucket 8-12   ...   (and 16-20, 40)
    python scripts/eval_backend_benchmarks.py --report        --out docs/rl_training/backend_benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qco.ir.parser import parse
from qco.modules.rl_scheduler.backend_matrix import DEFAULT_PROFILES, BackendPolicy, cross_matrix
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import BUCKETS, load_benchmarks
from qco.modules.rl_scheduler.plots import grouped_bars_svg
from qco.modules.rl_scheduler.ppo import load_policy


def load_policies(models: Path, profiles, seeds) -> list[BackendPolicy]:
    out = []
    for prof in profiles:
        for s in seeds:
            pol, payload = load_policy(models / f"{prof}_seed{s}.json")
            out.append(BackendPolicy(prof, s, pol, RLConfig.from_dict(payload["config"])))
    return out


def bucket_circuits(a, bucket: str):
    lo, hi = next((lo, hi) for b, lo, hi in BUCKETS if b == bucket)
    cs = load_benchmarks(a.circuits, 20) + load_benchmarks(a.circuits_40q, 64)
    ghz = parse(f"{a.circuits}/ghz_n40.qasm")
    ghz.name = "ghz_n40"
    return [c for c in cs + [ghz] if lo <= c.num_qubits <= hi]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", choices=[b for b, _, _ in BUCKETS])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", default="models/backend")
    ap.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--circuits", default="benchmarks/circuits")
    ap.add_argument("--circuits-40q", default="benchmarks/circuits_40q")
    ap.add_argument("--out", default="docs/rl_training/backend_benchmarks")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = tuple(a.profiles)

    if not a.report:
        pols = load_policies(Path(a.models), profiles, a.seeds)
        cs = bucket_circuits(a, a.bucket)
        res = cross_matrix(pols, cs, profiles)
        res["circuits"] = [c.name for c in cs]
        (out / f"bucket_{a.bucket}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    buckets = {b: json.loads(f.read_text()) for b, _, _ in BUCKETS if (f := out / f"bucket_{b}.json").exists()}
    md = ["# Per-backend policies on the benchmark circuits (issue #37 / #29)", "",
          f"Policies from `{a.models}` (trained on generated 5-qubit circuits with the time-aware reward, seeds {a.seeds}); "
          "metric = mean execution-time reduction (%) under the *test* backend's calibration; mean ± std over seeds. "
          "Rows = test backend, columns = backend the policy was trained on.", ""]
    short = {"superconducting_ibm_like": "superconducting", "trapped_ion_like": "trapped ion"}
    for b, r in buckets.items():
        md += [f"## {b} qubits ({r['n_circuits']} circuits)", "",
               "| test (rows) / trained on (columns) | " + " | ".join(profiles) + " | greedy one-step time heuristic |", "|---|" + "---|" * (len(profiles) + 1)]
        for t in profiles:
            row = [f"{r['matrix'][t][p]['mean']:.2f} ± {r['matrix'][t][p]['std_over_seeds']:.2f}" for p in profiles]
            md.append(f"| {t} | " + " | ".join(row) + f" | {r['matrix'][t]['greedy_time']['mean']:.2f} |")
        md += ["", f"* own-backend policy has the best mean: {r['wins_own_backend']}",
               f"* seeds (of {len(r['seeds'])}) where the own-backend policy beats the other's: {r['own_wins_per_seed']}",
               f"* fraction of (circuit, test backend, seed) cases with different schedules: {r['fraction_schedules_differ']:.2f}", ""]
    names = list(buckets)
    for t in profiles:
        series = {f"trained on {short.get(p, p)}": [buckets[b]["matrix"][t][p]["mean"] for b in names] for p in profiles}
        series["greedy time heuristic"] = [buckets[b]["matrix"][t]["greedy_time"]["mean"] for b in names]
        errs = {f"trained on {short.get(p, p)}": [buckets[b]["matrix"][t][p]["std_over_seeds"] for b in names] for p in profiles}
        errs["greedy time heuristic"] = [0.0] * len(names)
        fn = f"test_{t}.svg"
        grouped_bars_svg(f"Test backend: {short.get(t, t)}", [f"{b}q" for b in names], series, "execution-time reduction %", errs, out / fn)
        md += [f"![{t}]({fn})", ""]
    (out / "backend_benchmarks.md").write_text("\n".join(md), encoding="utf-8")
    (out / "backend_benchmarks.json").write_text(json.dumps(buckets, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
