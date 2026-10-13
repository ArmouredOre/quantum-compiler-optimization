"""Partitioner bench (#9): wire cut, block count, size spread, max width per backend.

    python benchmarks/bench_partition.py [min_qubits]
"""

import glob
import statistics
import sys

from qco.graphs.partitioning import cut_size, partition_circuit
from qco.ir import parse


def main(min_qubits: int = 16) -> None:
    backends = ["fallback", "greedy", "metis"]
    print(f"{'circuit':24} {'n':>3} {'gates':>5}  " + "  ".join(f"{b:>26}" for b in backends))
    print(" " * 36 + "  ".join(f"{'cut/blocks/sd/width':>26}" for _ in backends))
    for path in sorted(glob.glob("benchmarks/circuits/*.qasm")):
        ir = parse(path)
        if ir.num_qubits < min_qubits:
            continue
        cells = []
        for backend in backends:
            try:
                bs = partition_circuit(ir, backend=backend)
            except ImportError:
                cells.append(f"{'n/a':>26}")
                continue
            sd = statistics.pstdev(len(b.ir) for b in bs)
            width = max(len({q for g in b.ir for q in g.qubits}) for b in bs)
            cells.append(f"{cut_size(ir, bs):>8}/{len(bs):>4}/{sd:>5.1f}/{width:>4}")
        print(f"{path.split('/')[-1][:-5]:24} {ir.num_qubits:>3} {len(ir):>5}  " + "  ".join(cells))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
