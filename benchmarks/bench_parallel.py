"""Serial vs parallel block stage (#46): identical circuits + wall-clock speedup.

    python benchmarks/bench_parallel.py [--workers 1 2 4] [--repeat 3] [--out docs/parallel_blocks_bench.md]
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from pathlib import Path

from qco.ir import parse
from qco.pipeline import HybridOptimizer, PipelineConfig

warnings.simplefilter("ignore")
CIRCUITS = ["rand_cliffordt_n20", "rand_cnotpauli_n20"]


def run(ir, workers, **kw):
    cfg = PipelineConfig(use_ea=False, gnn_checkpoint=None, parallel_blocks=workers, **kw)
    t = time.perf_counter()
    res = HybridOptimizer(cfg).run(ir)
    return res, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    modes = {"no resynthesis": dict(enable_resynthesis=False),
             "resynthesis, 20 s budget": dict(enable_resynthesis=True, solver_budget_s=20.0)}
    rows = ["| circuit | mode | workers | block stage s | total s | speedup (block stage) | identical to serial |",
            "|---|---|---|---|---|---|---|"]
    for name in CIRCUITS:
        ir = parse(f"benchmarks/circuits/{name}.qasm")
        for mode, kw in modes.items():
            base_sig, base_t = None, None
            for w in a.workers:
                ts, tot, sig = [], [], None
                for _ in range(a.repeat):
                    res, dt = run(ir, w, **kw)
                    ts.append(res.block_stage_s)
                    tot.append(dt)
                    sig = [p.circuit.to_qasm2() for p in res.pareto_front]
                bt, tt = statistics.median(ts), statistics.median(tot)
                if base_sig is None:
                    base_sig, base_t = sig, bt
                rows.append(f"| {name} | {mode} | {w} | {bt:.3f} | {tt:.3f} | {base_t / bt:.2f}x | "
                            f"{'yes' if sig == base_sig else 'NO'} |")
                print(rows[-1], flush=True)
    text = "\n".join(rows)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
