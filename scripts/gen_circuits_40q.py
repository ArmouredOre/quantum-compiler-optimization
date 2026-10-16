#!/usr/bin/env python
"""40-qubit benchmark circuits for the Module A generalization study (#37).

benchmarks/circuits only has ghz_n40 at 40 qubits, and GHZ is a pure chain (nothing to schedule).
This writes qft / qaoa / random Clifford+T / random CNOT+Pauli at 40 qubits to
benchmarks/circuits_40q/ using the same generators and seed as benchmarks/generate_circuits.py,
so benchmarks/circuits (and suite.yaml) stay untouched.

    python scripts/gen_circuits_40q.py [--out benchmarks/circuits_40q]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import generate_circuits as gc  # noqa: E402

N = 40


def build() -> dict[str, "gc.QASM"]:
    d = gc.DEFAULTS
    return {
        f"qft_n{N}": gc.qft(N),
        f"qaoa_n{N}_p1": gc.qaoa(N, 1, d["seed"]),
        f"qaoa_n{N}_p2": gc.qaoa(N, 2, d["seed"]),
        f"rand_cliffordt_n{N}": gc.random_clifford_t(N, d["rand_layers"], d["t_fraction"], d["seed"]),
        f"rand_cnotpauli_n{N}": gc.random_cnot_pauli(N, d["rand_layers"], d["seed"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "circuits_40q"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, c in sorted(build().items()):
        (out / f"{name}.qasm").write_text(c.text(), encoding="utf-8")
        print(f"{name:22s} qubits={c.n} gates={len(c.lines)}")


if __name__ == "__main__":
    main()
