#!/usr/bin/env python
"""Extra random circuits for issue #28 (more non-Grover positives).

Writes the ORIGINAL benchmark circuits plus new rand_* circuits to --out (default
data/circuits_v2), so it can be passed to eval_windows.py as --circuits.
Names stay compatible with family(): rand_cliffordt_n6_s1 -> family 'rand_cliffordt'.
"""
import argparse, math, random, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import generate_circuits as gc   # noqa: E402

SEED = gc.DEFAULTS["seed"]


def random_clifford_rz(n, layers, seed):
    """h/s/sdg + rz(k*pi/4) + cx."""
    rng = random.Random(f"{seed}-{n}-{layers}-cliffordrz")
    c = gc.QASM(n)
    for _ in range(layers):
        for q in range(n):
            r = rng.random()
            if r < 0.20:
                c.g("rz", q, params=(rng.choice([1, 2, 3, 5, 6, 7]) * math.pi / 4,))
            elif r < 0.55:
                c.g(rng.choice(["h", "s", "sdg"]), q)
        a = list(range(n)); rng.shuffle(a)
        for i in range(0, n - 1, 2):
            if rng.random() < 0.6:
                c.g("cx", a[i], a[i + 1])
    return c


def random_cx_rz(n, layers, seed):
    """cx + rz(k*pi/4) only (phase-polynomial style)."""
    rng = random.Random(f"{seed}-{n}-{layers}-cxrz")
    c = gc.QASM(n)
    for _ in range(layers):
        for q in range(n):
            if rng.random() < 0.4:
                c.g("rz", q, params=(rng.choice([1, 2, 3, 5, 6, 7]) * math.pi / 4,))
        a = list(range(n)); rng.shuffle(a)
        for i in range(0, n - 1, 2):
            if rng.random() < 0.7:
                c.g("cx", a[i], a[i + 1])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "circuits_v2")
    ap.add_argument("--src", type=Path, default=ROOT / "benchmarks" / "circuits")
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3, help="circuits per (family, size)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    extra = {}
    for n in a.sizes:
        for k in range(1, a.seeds + 1):
            s = f"{SEED}-s{k}"
            extra[f"rand_cliffordt_n{n}_s{k}"] = gc.random_clifford_t(n, a.layers, 0.15, s)
            extra[f"rand_cnotpauli_n{n}_s{k}"] = gc.random_cnot_pauli(n, a.layers, s)
            extra[f"rand_cliffordrz_n{n}_s{k}"] = random_clifford_rz(n, a.layers, s)
            extra[f"rand_cxrz_n{n}_s{k}"] = random_cx_rz(n, a.layers, s)

    if a.list:
        for name, c in sorted(extra.items()):
            print(f"{name:32s} qubits={c.n:3d} gates={len(c.lines)}")
        print(f"\n{len(extra)} new circuits")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    for p in a.src.glob("*.qasm"):
        shutil.copy(p, a.out / p.name)
    for name, c in extra.items():
        (a.out / f"{name}.qasm").write_text(c.text(), encoding="utf-8")
    print(f"wrote {len(list(a.src.glob('*.qasm')))} original + {len(extra)} new circuits to {a.out}")


if __name__ == "__main__":
    main()
    