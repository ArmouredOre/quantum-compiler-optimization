#!/usr/bin/env python
"""Issue #25: P/R/F1 of rule-label vs verified-label GNNs, scored on verified ground truth.

Usage:
    python scripts/eval_gnn.py --epochs 2     # quick smoke run
    python scripts/eval_gnn.py --epochs 50
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.dataset import verify_labels
from qco.modules.gnn_cancellation.eval import (
    adjacent_inverse_baseline,
    build_pr_table,
    calibrate_threshold,
    train_val_test_split,
    write_pr_table,
)
from qco.modules.gnn_cancellation.features import rule_based_candidates
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor, GNNConfig
from qco.modules.smt_verifier.equivalence import EquivalenceChecker

ROOT = Path(__file__).resolve().parents[1]


def load_circuits(directory: Path, max_qubits: int):
    out = []
    for p in sorted(directory.glob("*.qasm")):
        ir = from_qasm2(p.read_text(encoding="utf-8"))
        ir.name = p.stem
        if ir.num_qubits <= max_qubits:
            out.append(ir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=Path, default=ROOT / "benchmarks" / "circuits")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "gnn_eval" / "pr_table.md")
    ap.add_argument("--max-qubits", type=int, default=8)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--conv", default="sage", choices=["sage", "gat"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    circuits = load_circuits(args.circuits, args.max_qubits)
    if not circuits:
        raise SystemExit(f"no .qasm circuits <= {args.max_qubits} qubits in {args.circuits}")

    # 1. Verified ground truth: rule candidates that pass Module C's check.
    checker = EquivalenceChecker(backend=args.backend)
    usable, skipped = [], []
    total: Counter = Counter()
    by_kind: dict[str, Counter] = {}
    for ir in circuits:
        cands = rule_based_candidates(ir)
        status = verify_labels(ir, cands, checker=checker, backend=args.backend)
        if any(s is None for s in status):
            skipped.append(ir.name)
            continue
        gold = [c for c, ok in zip(cands, status) if ok]
        for c, ok in zip(cands, status):
            key = "verified" if ok else "rejected"
            total[key] += 1
            by_kind.setdefault(c.kind, Counter())[key] += 1
        usable.append((ir, gold))

    print(f"circuits: {len(circuits)} loaded, {len(usable)} usable, {len(skipped)} skipped {skipped}")
    print(f"rule candidates: {dict(total)}")
    for kind, cnt in sorted(by_kind.items()):
        print(f"  {kind}: {dict(cnt)}")
    if len(usable) < 5:
        print("WARNING: fewer than 5 usable circuits; splits will be tiny and noisy.")

    # 2. Seeded split.
    train, val, test = train_val_test_split(usable, seed=args.seed)
    if not val or not test:
        raise SystemExit("val or test split is empty; need more circuits")
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")

    # 3. Train one GNN per label source (train circuits only).
    cfg = GNNConfig(conv=args.conv, epochs=args.epochs)
    datasets = {
        "rule": [(ir, rule_based_candidates(ir)) for ir, _ in train],
        "verified": [(ir, gold) for ir, gold in train],
    }
    systems = {
        "Rule-based generator": rule_based_candidates,
        "Adjacent-inverse baseline": adjacent_inverse_baseline,
    }
    thresholds = {}
    for source, data in datasets.items():
        pred = GNNCancellationPredictor(cfg)
        pred.fit(data, seed=args.seed, val_fraction=0.0)
        # 4. Calibrate on the validation split, against verified gold.
        t, _sweep = calibrate_threshold(pred, val)
        thresholds[source] = t
        pred._threshold = t
        systems[f"GNN ({source} labels, t={t:.2f})"] = (
            lambda ir, p=pred, t=t: p.predict(ir, threshold=t)
        )
        print(f"{source}: calibrated threshold {t:.2f}")

    # 5. Score everything on the held-out test split, same verified gold.
    table = build_pr_table(test, systems)
    header = (
        "## GNN vs rule-based: precision / recall / F1 (held-out test, verified ground truth)\n\n"
        f"- seed {args.seed}, conv `{args.conv}`, {args.epochs} epochs\n"
        f"- circuits: {len(usable)} usable of {len(circuits)} (skipped: {len(skipped)}); "
        f"split {len(train)}/{len(val)}/{len(test)} (train/val/test)\n"
        f"- rule candidates passing Module C's check: {total['verified']} of "
        f"{total['verified'] + total['rejected']}\n"
        f"- thresholds calibrated on val: {thresholds}"
    )
    write_pr_table(args.out, table, header=header)
    print(header + "\n\n" + table)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
    