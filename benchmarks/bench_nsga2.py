"""Module D bench (#30 / N3): hypervolume of the NSGA-II front and how many front
members are equivalent to the input, on the small benchmarks.

    python benchmarks/bench_nsga2.py [max_qubits]

The front is filtered by the numeric checker exactly as the pre-N3 pipeline did;
with N3 every member should pass. Objectives are normalized by 1.1 x the input's
(gate count, depth, infidelity), so HV of the input alone is (1 - 1/1.1)^3.
"""

import glob
import sys

import numpy as np
from pymoo.indicators.hv import HV

from qco.ir import parse
from qco.modules.evolutionary.nsga2 import NSGA2Compiler, NSGA2Config
from qco.modules.smt_verifier.equivalence import EquivalenceChecker


def main(max_qubits: int = 6) -> None:
    checker = EquivalenceChecker(backend="numeric")
    total = 0.0
    print(f"{'circuit':24} {'front':>5} {'equiv':>5} {'hv':>8}")
    for path in sorted(glob.glob("benchmarks/circuits/*.qasm")):
        ir = parse(path)
        if ir.num_qubits > max_qubits or len(ir) > 180:
            continue
        ea = NSGA2Compiler(NSGA2Config(pop_size=30, generations=20))
        front = ea.evolve([ir])
        ok = [p for p in front if checker.check(ir, p.circuit).equivalent]
        ref = [v * 1.1 + 1e-3 for v in ea.score(ir).objective.as_tuple()]
        pts = [[a / r for a, r in zip(p.objective.as_tuple(), ref)] for p in (ok or [ea.score(ir)])]
        hv = HV(ref_point=[1.0, 1.0, 1.0])(np.array(pts))
        total += hv
        print(f"{path.split('/')[-1][:-5]:24} {len(front):5} {len(ok):5} {hv:8.4f}", flush=True)
    print(f"{'TOTAL':24} {'':5} {'':5} {total:8.4f}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
