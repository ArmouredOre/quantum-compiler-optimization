import csv, sys

def load(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    key = next(k for k in ("circuit", "name", "benchmark") if k in rows[0])
    return {r[key]: r for r in rows}, list(rows[0].keys())

ra_path = sys.argv[1] if len(sys.argv) > 1 else "results/rules/qco.csv"
gn_path = sys.argv[2] if len(sys.argv) > 2 else "results/gnn/qco.csv"
a, cols = load(ra_path)
b, _ = load(gn_path)
bef = "gate_count_before" if "gate_count_before" in cols else next(c for c in cols if "before" in c)
aft = "gate_count_after"
tr = tg = 0
for k in sorted(a.keys() & b.keys()):
    ra, rb = int(a[k][bef]) - int(a[k][aft]), int(b[k][bef]) - int(b[k][aft])
    tr += ra; tg += rb
    if rb != ra:
        print(f"{k:28s} rules {ra:4d}  gnn {rb:4d}  {'<-- WORSE' if rb < ra else 'better'}")
print(f"TOTAL removed: rules {tr}, gnn {tg} -> {'PASS' if tg >= tr else 'FAIL'}")
