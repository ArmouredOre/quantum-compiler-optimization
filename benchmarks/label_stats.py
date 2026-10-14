import collections
import glob
import json
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "build/module_b"
pc, wc = collections.Counter(), collections.Counter()
for f in sorted(glob.glob(f"{d}/pairs/*.json")):
    j = json.load(open(f, encoding="utf-8"))
    c = collections.Counter(str(r["label"]) for r in j["candidates"])
    pc.update(c)
    print("pairs  ", j["name"], dict(c))
for f in sorted(glob.glob(f"{d}/windows/*.json")):
    j = json.load(open(f, encoding="utf-8"))
    c = collections.Counter(f'{w["status"]}/{w["label"]}' for w in j["windows"])
    wc.update(c)
    print("windows", j["name"], dict(c))
print("\nTOTAL pairs  ", dict(pc))
print("TOTAL windows", dict(wc))
