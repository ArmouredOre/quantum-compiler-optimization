import csv, sys

def load(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    key = next(k for k in ("circuit", "name", "benchmark") if k in rows[0])
    return {r[key]: r for r in rows}, list(rows[0].keys())

a, cols = load("results/rules/qco.csv")
b, _ = load("results/gnn/qco.csv")
print("columns:", cols)
bef = next(c for c in cols if "before" in c)
aft = next(c for c in cols if c == "gate_count_after")
tr = tg = 0
for k in a:
    ra, rb = int(a[k][bef]) - int(a[k][aft]), int(b[k][bef]) - int(b[k][aft])
    tr += ra; tg += rb
    flag = "  <-- WORSE" if rb < ra else ""
    print(f"{k:24s} rules {ra:4d}  gnn {rb:4d}{flag}")
print(f"TOTAL removed: rules {tr}, gnn {tg} -> {'PASS' if tg >= tr else 'FAIL'}")
