"""Partitioner tuning on large circuits (#46).

Sweeps ``max_block_gates`` x backend x ``max_block_qubits`` on the >=20-qubit
benchmarks and reports cut size, block-size spread, width and partition time.
``--downstream`` also runs the hybrid pipeline (no GNN/EA/resynthesis) per
setting and reports the gate reduction, so the table shows the quality cost of
small blocks (gates cannot cancel across a block boundary).

    python benchmarks/tune_partition.py [--min-qubits 20] [--downstream]
        [--out docs/partitioner_tuning.md]
"""

from __future__ import annotations

import argparse
import glob
import statistics
import time
from pathlib import Path

from qco.graphs.partitioning import cut_size, partition_circuit
from qco.ir import parse

GATES = (16, 32, 48, 64, 96, 128, 192)
BACKENDS = ("greedy", "fallback", "metis")
QUBITS = (6, 8)


def measure(ir, gates, backend, qubits):
    t = time.perf_counter()
    try:
        bs = partition_circuit(ir, max_block_gates=gates, backend=backend, max_block_qubits=qubits)
    except ImportError:
        return None
    dt = time.perf_counter() - t
    sizes = [len(b.ir) for b in bs]
    return {
        "blocks": len(bs),
        "cut": cut_size(ir, bs),
        "mean": statistics.mean(sizes),
        "sd": statistics.pstdev(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "width": max(len({q for g in b.ir for q in g.qubits}) for b in bs),
        "ms": dt * 1e3,
    }


def downstream(ir, gates, backend, qubits):
    from qco.pipeline import HybridOptimizer, PipelineConfig

    cfg = PipelineConfig(
        use_ea=False, gnn_checkpoint=None, max_block_gates=gates,
        partition_backend=backend, feedback_iterations=1,
    )
    t = time.perf_counter()
    res = HybridOptimizer(cfg).run(ir)
    best = min(len(pt.circuit) for pt in res.pareto_front)
    return 100.0 * (1 - best / max(1, len(ir))), time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-qubits", type=int, default=20)
    ap.add_argument("--downstream", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = []
    for path in sorted(glob.glob("benchmarks/circuits/*.qasm")):
        ir = parse(path)
        if ir.num_qubits < a.min_qubits:
            continue
        name = Path(path).stem
        lines += [f"### {name} ({ir.num_qubits}q, {len(ir)} gates)", ""]
        head = "| backend | max_gates | max_qubits | blocks | cut | mean | sd | min | max | width | ms |"
        if a.downstream:
            head += " gate red % | pipeline s |"
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        for backend in BACKENDS:
            for q in QUBITS:
                for g in GATES:
                    m = measure(ir, g, backend, q)
                    if m is None:
                        continue
                    row = (f"| {backend} | {g} | {q} | {m['blocks']} | {m['cut']} | {m['mean']:.1f} | "
                           f"{m['sd']:.1f} | {m['min']} | {m['max']} | {m['width']} | {m['ms']:.1f} |")
                    if a.downstream and q == 8:  # the pipeline fixes max_block_qubits=8
                        red, s = downstream(ir, g, backend, q)
                        row += f" {red:.1f} | {s:.2f} |"
                    elif a.downstream:
                        row += " - | - |"
                    lines.append(row)
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
