#!/usr/bin/env python3
"""Generate the benchmark circuit suite as OpenQASM 2.0 files.

Dependency-free (stdlib + optional PyYAML) so the suite is reproducible on a bare
Python 3.10+ install. Reads ``benchmarks/suite.yaml`` when PyYAML is available,
otherwise falls back to the defaults baked in below.

    python benchmarks/generate_circuits.py            # write benchmarks/circuits/*.qasm
    python benchmarks/generate_circuits.py --list     # dry run
    python benchmarks/generate_circuits.py --out DIR
"""

from __future__ import annotations

import argparse
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "circuits")
SUITE_PATH = os.path.join(HERE, "suite.yaml")

DEFAULTS = {
    "seed": 20260901,
    "ghz": [5, 10, 20, 40],
    "qft": [5, 8, 12, 16],
    "grover_working": [3, 5, 7],
    "grover_iters": [1, 2],
    "qaoa_sizes": [6, 10, 16],
    "qaoa_p": [1, 2],
    "rand_sizes": [5, 10, 20],
    "rand_layers": 40,
    "t_fraction": 0.15,
}


class QASM:
    def __init__(self, n: int):
        self.n = n
        self.lines: list[str] = []

    def g(self, name: str, *qubits: int, params: tuple[float, ...] = ()):
        p = f"({','.join(f'{x:.10g}' for x in params)})" if params else ""
        args = ",".join(f"q[{q}]" for q in qubits)
        self.lines.append(f"{name}{p} {args};")

    def text(self) -> str:
        head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.n}];"]
        return "\n".join(head + self.lines) + "\n"


# ----------------------------------------------------------------------------
# circuit families
# ----------------------------------------------------------------------------

def ghz(n: int) -> QASM:
    c = QASM(n)
    c.g("h", 0)
    for i in range(n - 1):
        c.g("cx", i, i + 1)
    return c


def qft(n: int, swaps: bool = True) -> QASM:
    c = QASM(n)
    for j in range(n):
        c.g("h", j)
        for k in range(j + 1, n):
            c.g("cp", k, j, params=(math.pi / (2 ** (k - j)),))
    if swaps:
        for i in range(n // 2):
            c.g("swap", i, n - 1 - i)
    return c


def _mcx(c: QASM, controls: list[int], target: int, ancillas: list[int]) -> None:
    """Multi-controlled X via a clean-ancilla ccx ladder."""
    if len(controls) == 1:
        c.g("cx", controls[0], target)
        return
    if len(controls) == 2:
        c.g("ccx", controls[0], controls[1], target)
        return
    assert len(ancillas) >= len(controls) - 2
    c.g("ccx", controls[0], controls[1], ancillas[0])
    used = [ancillas[0]]
    for i in range(2, len(controls) - 1):
        c.g("ccx", controls[i], used[-1], ancillas[i - 1])
        used.append(ancillas[i - 1])
    c.g("ccx", controls[-1], used[-1], target)
    for i in range(len(controls) - 2, 1, -1):
        c.g("ccx", controls[i], used[-2], used[-1])
        used.pop()
    c.g("ccx", controls[0], controls[1], ancillas[0])


def grover(working: int, iterations: int) -> QASM:
    ancillas_needed = max(working - 2, 0)
    n = working + ancillas_needed
    w = list(range(working))
    anc = list(range(working, n))
    c = QASM(n)
    for q in w:
        c.g("h", q)
    for _ in range(iterations):
        # Oracle: phase-flip the all-ones state.
        c.g("h", w[-1])
        _mcx(c, w[:-1], w[-1], anc)
        c.g("h", w[-1])
        # Diffuser.
        for q in w:
            c.g("h", q)
            c.g("x", q)
        c.g("h", w[-1])
        _mcx(c, w[:-1], w[-1], anc)
        c.g("h", w[-1])
        for q in w:
            c.g("x", q)
            c.g("h", q)
    return c


def _random_3_regular_edges(n: int, rng: random.Random) -> list[tuple[int, int]]:
    if n < 4:
        return [(i, (i + 1) % n) for i in range(n)]
    for _ in range(200):
        stubs = [v for v in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges, ok = set(), True
        for a, b in zip(stubs[::2], stubs[1::2]):
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            return sorted(edges)
    return sorted({(i, (i + 1) % n) for i in range(n)})


def qaoa(n: int, p: int, seed: int) -> QASM:
    rng = random.Random(f"{seed}-{n}-{p}-qaoa")
    edges = _random_3_regular_edges(n, rng)
    c = QASM(n)
    for q in range(n):
        c.g("h", q)
    for layer in range(p):
        gamma = rng.uniform(0.1, math.pi - 0.1)
        beta = rng.uniform(0.1, math.pi / 2)
        for a, b in edges:                       # cost layer  exp(-i gamma Z Z)
            c.g("cx", a, b)
            c.g("rz", b, params=(2 * gamma,))
            c.g("cx", a, b)
        for q in range(n):                       # mixer layer
            c.g("rx", q, params=(2 * beta,))
    return c


def random_clifford_t(n: int, layers: int, t_fraction: float, seed: int) -> QASM:
    rng = random.Random(f"{seed}-{n}-{layers}-cliffordt")
    c = QASM(n)
    for _ in range(layers):
        for q in range(n):
            r = rng.random()
            if r < t_fraction:
                c.g(rng.choice(["t", "tdg"]), q)
            elif r < 0.55:
                c.g(rng.choice(["h", "s", "sdg"]), q)
        a = list(range(n))
        rng.shuffle(a)
        for i in range(0, n - 1, 2):
            if rng.random() < 0.6:
                c.g("cx", a[i], a[i + 1])
    return c


def random_cnot_pauli(n: int, layers: int, seed: int) -> QASM:
    rng = random.Random(f"{seed}-{n}-{layers}-cnotpauli")
    c = QASM(n)
    for _ in range(layers):
        for q in range(n):
            if rng.random() < 0.5:
                c.g(rng.choice(["x", "y", "z"]), q)
        a = list(range(n))
        rng.shuffle(a)
        for i in range(0, n - 1, 2):
            if rng.random() < 0.6:
                c.g("cx", a[i], a[i + 1])
    return c


# ----------------------------------------------------------------------------

def _load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        import yaml  # type: ignore
    except ImportError:
        return cfg
    try:
        with open(SUITE_PATH, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return cfg
    cfg["seed"] = raw.get("seed", cfg["seed"])
    fam = raw.get("families", {})
    if "ghz" in fam:
        cfg["ghz"] = fam["ghz"].get("sizes", cfg["ghz"])
    if "qft" in fam:
        cfg["qft"] = fam["qft"].get("sizes", cfg["qft"])
    if "grover" in fam:
        cfg["grover_working"] = fam["grover"].get("working_qubits", cfg["grover_working"])
        cfg["grover_iters"] = fam["grover"].get("iterations", cfg["grover_iters"])
    if "qaoa" in fam:
        cfg["qaoa_sizes"] = fam["qaoa"].get("sizes", cfg["qaoa_sizes"])
        cfg["qaoa_p"] = fam["qaoa"].get("p", cfg["qaoa_p"])
    if "random_clifford_t" in fam:
        cfg["rand_sizes"] = fam["random_clifford_t"].get("sizes", cfg["rand_sizes"])
        cfg["rand_layers"] = fam["random_clifford_t"].get("layers", cfg["rand_layers"])
        cfg["t_fraction"] = fam["random_clifford_t"].get("t_fraction", cfg["t_fraction"])
    return cfg


def build_all(cfg: dict) -> dict[str, QASM]:
    seed = cfg["seed"]
    out: dict[str, QASM] = {}
    for n in cfg["ghz"]:
        out[f"ghz_n{n}"] = ghz(n)
    for n in cfg["qft"]:
        out[f"qft_n{n}"] = qft(n)
    for w in cfg["grover_working"]:
        for it in cfg["grover_iters"]:
            out[f"grover_w{w}_i{it}"] = grover(w, it)
    for n in cfg["qaoa_sizes"]:
        for p in cfg["qaoa_p"]:
            out[f"qaoa_n{n}_p{p}"] = qaoa(n, p, seed)
    for n in cfg["rand_sizes"]:
        out[f"rand_cliffordt_n{n}"] = random_clifford_t(n, cfg["rand_layers"], cfg["t_fraction"], seed)
        out[f"rand_cnotpauli_n{n}"] = random_cnot_pauli(n, cfg["rand_layers"], seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--list", action="store_true", help="dry run: list circuits only")
    args = ap.parse_args()

    cfg = _load_config()
    circuits = build_all(cfg)

    if args.list:
        for name, c in sorted(circuits.items()):
            print(f"{name:24s}  qubits={c.n:3d}  gates={len(c.lines)}")
        print(f"\n{len(circuits)} circuits (seed={cfg['seed']})")
        return

    os.makedirs(args.out, exist_ok=True)
    for name, c in sorted(circuits.items()):
        path = os.path.join(args.out, f"{name}.qasm")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(c.text())
    print(f"wrote {len(circuits)} circuits to {args.out}")


if __name__ == "__main__":
    main()
