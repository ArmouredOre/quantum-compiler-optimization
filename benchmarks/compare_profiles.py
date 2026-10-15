"""Task 6: rules vs GNN across calibration profiles.

Per profile, joins results/profiles/<profile>/{rules,gnn}/qco.csv per circuit.
A circuit is WORSE if the GNN run ends with more gates or a higher CX-cost than
the rules run. Pass condition: worse == 0 in every row and min dFidelity >= 0.
"""
import csv, sys
from pathlib import Path


def load(p):
    return {r["circuit"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def main(root):
    root = Path(root)
    profiles = sorted(d.name for d in root.iterdir() if (d / "rules" / "qco.csv").exists())
    if not profiles:
        sys.exit(f"no profiles found under {root}")
    hdr = (f"{'profile':28s} {'n':>3s} {'gates r':>8s} {'gates g':>8s} {'cx r':>6s} {'cx g':>6s} "
           f"{'worse':>6s} {'mean dF':>10s} {'min dF':>10s}  status")
    print(hdr); print("-" * len(hdr))
    ok = True
    for p in profiles:
        a = load(root / p / "rules" / "qco.csv")
        b = load(root / p / "gnn" / "qco.csv")
        keys = sorted(a.keys() & b.keys())
        if len(keys) != len(a) or len(keys) != len(b):
            print(f"WARNING {p}: circuit sets differ (rules {len(a)}, gnn {len(b)}, shared {len(keys)})")
        gr = gg = cr = cg = worse = 0
        dfs = []
        for k in keys:
            ga, gb = int(a[k]["gate_count_after"]), int(b[k]["gate_count_after"])
            ca, cb = int(a[k]["cx_cost_after"]), int(b[k]["cx_cost_after"])
            gr += ga; gg += gb; cr += ca; cg += cb
            if gb > ga or cb > ca:
                worse += 1
                print(f"  WORSE {p}/{k}: gates {ga}->{gb}, cx {ca}->{cb}")
            dfs.append(float(b[k]["fidelity_after"]) - float(a[k]["fidelity_after"]))
        mean_df = sum(dfs) / len(dfs) if dfs else float("nan")
        min_df = min(dfs) if dfs else float("nan")
        passed = worse == 0 and min_df >= -1e-9
        ok &= passed
        print(f"{p:28s} {len(keys):3d} {gr:8d} {gg:8d} {cr:6d} {cg:6d} {worse:6d} "
              f"{mean_df:+10.4f} {min_df:+10.4f}  {'PASS' if passed else 'FAIL'}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/profiles")
    