#!/usr/bin/env python3
"""Train Module B on verified labels and write the checkpoint the pipeline loads (#34).

    python benchmarks/train_module_b.py --conv sage --out checkpoints/sage.pt
    python benchmarks/train_module_b.py --conv gat  --out checkpoints/gat.pt

Labels are generated once into --work and reused, so both runs share the same
circuits and the same by-circuit train/val/test split. Training circuits come
from generate_circuits.py with a DIFFERENT seed and different sizes than the
benchmark suite, so bench_gnn_vs_rules.py is not evaluated on training data.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.dataset import (
    dump_resynth_labels,
    dump_suite,
    load_resynth_labels,
    load_suite,
)
from qco.modules.gnn_cancellation.eval import (
    build_pr_table,
    calibrate_threshold,
    expected_calibration_error,
    rule_window_scorer,
    sigmoid,
    train_val_test_split,
    window_ranking_table,
)
from qco.modules.gnn_cancellation.features import rule_based_candidates
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor, GNNConfig
from qco.modules.smt_verifier.bounded_optimizer import BoundedResynthesizer

HERE = Path(__file__).resolve().parent


def generate(out: Path, seed: int, n_seeds: int) -> None:
    spec = importlib.util.spec_from_file_location("generate_circuits", HERE / "generate_circuits.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(n_seeds):
        cfg = {
            **gc.DEFAULTS,
            "seed": seed + k,
            "ghz": [6, 12], "qft": [6, 10], "grover_working": [4, 6],
            "rand_sizes": [5, 8], "rand_layers": 12,
        }
        for name, c in gc.build_all(cfg).items():
            if name.startswith("qaoa_") or (k and not name.startswith("rand_")):
                continue
            (out / f"{name}_s{seed + k}.qasm").write_text(c.text(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/module_b.pt")
    ap.add_argument("--work", default="build/module_b")
    ap.add_argument("--conv", default="sage", choices=["sage", "gat"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=2.0, help="per-window solver timeout")
    ap.add_argument("--extra-labels", nargs="*", default=["data/labels_resynth_v2", "data/lab_new"])
    ap.add_argument("--timeouts-as-zero", action="store_true")
    ap.add_argument("--exclude-prefix", nargs="*", default=[])
    a = ap.parse_args()

    work = Path(a.work)
    circ, pair_dir, win_dir = work / "circuits", work / "pairs", work / "windows"

    if not any(circ.glob("*.qasm")):
        generate(circ, a.seed, a.n_seeds)
    if not any(pair_dir.glob("*.json")):
        dump_suite(circ, pair_dir, label_source="verified")
    if not any(win_dir.glob("*.json")):
        rc = BoundedResynthesizer(None).config  # same window shape the pipeline re-synthesizes
        irs = []
        for p in sorted(circ.glob("*.qasm")):
            ir = from_qasm2(p.read_text(encoding="utf-8"))
            ir.name = p.stem
            irs.append(ir)
        dump_resynth_labels(
            irs, win_dir, timeout_s=a.timeout, max_qubits=rc.max_qubits, max_gates=rc.max_gates
        )

    pairs = {ir.name: (ir, c) for ir, c in load_suite(pair_dir)}
    wins = {ir.name: (ir, r) for ir, r in load_resynth_labels(win_dir)}
    # extra #28 window labels; suite circuits excluded because acceptance is scored on them
    suite = {p.stem for p in Path("benchmarks/circuits").glob("*.qasm")}
    extra_c, extra_p = work / "extra_circuits", work / "extra_pairs"
    extra_c.mkdir(parents=True, exist_ok=True)
    for d in a.extra_labels:
        for ir, rows in load_resynth_labels(d):
            if ir.name in suite or ir.name in wins:
                continue
            wins[ir.name] = (ir, rows)
            (extra_c / f"{ir.name}.qasm").write_text(ir.to_qasm2(), encoding="utf-8")
    if len(list(extra_p.glob("*.json"))) < len(list(extra_c.glob("*.qasm"))):
        dump_suite(extra_c, extra_p, label_source="verified")
    for ir, c in load_suite(extra_p):
        pairs[ir.name] = (ir, c)
    if a.timeouts_as_zero:
        for _, rows in wins.values():
            for r in rows:
                if r.label is None and r.status == "timeout":
                    r.label = 0
    allrows = [r for _, rows in wins.values() for r in rows]
    rc = BoundedResynthesizer(None).config
    print(f"windows: {len(wins)} circuits, {sum(r.label == 1 for r in allrows)} positives, "
          f"max window {max(r.end - r.start for r in allrows)} gates "
          f"(pipeline max_gates={rc.max_gates}, max_qubits={rc.max_qubits}); "
          f"pair positives: {sum(len(c) for _, c in pairs.values())}")
    names = [n for n in sorted(pairs.keys() & wins.keys()) if not n.startswith(tuple(a.exclude_prefix))]
    assert not any(n.startswith(tuple(a.exclude_prefix)) for n in names) if a.exclude_prefix else True
    print("excluded:", sorted(n for n in pairs.keys() & wins.keys() if a.exclude_prefix and n.startswith(tuple(a.exclude_prefix))))
    tr, va, te = train_val_test_split(names, seed=a.seed)  # split BY CIRCUIT
    P = lambda ns: [pairs[n] for n in ns]  # noqa: E731
    W = lambda ns: [wins[n] for n in ns]   # noqa: E731
    print(f"{len(names)} circuits -> train {len(tr)} / val {len(va)} / test {len(te)}")

    pred = GNNCancellationPredictor(GNNConfig(conv=a.conv, epochs=a.epochs))
    pred.fit(P(tr), seed=a.seed, val_fraction=0.0)          # pair head, verified positives
    wfit = pred.fit_windows(W(tr), W(va), seed=a.seed)      # window head + temperature
    thr, _ = calibrate_threshold(pred, P(va))               # AFTER fit_windows: shared encoder moved
    pred.set_threshold(thr)
    dest = pred.save(a.out)
    print(f"saved {dest}  pair threshold={thr:.2f}  window T={wfit.temperature:.2f}")

    tag = f"GNN-{a.conv}"
    print(build_pr_table(P(te), {"rules": rule_based_candidates, tag: pred.predict}))
    combo = lambda ir, w: [r + g for r, g in zip(rule_window_scorer(ir, w), pred.score_windows(ir, w))]  # noqa: E731
    print(window_ranking_table(W(te), {"rules": rule_window_scorer, tag: pred.score_windows,
    f"rules+{tag}": combo}))

    ys, zs = [], []
    for ir, rows in W(te):
        rows = [r for r in rows if r.label is not None]
        ys += [int(r.label) for r in rows]
        zs += pred.window_logits(ir, [(r.start, r.end) for r in rows])
    raw = expected_calibration_error([sigmoid(z) for z in zs], ys)
    cal = expected_calibration_error([sigmoid(z / wfit.temperature) for z in zs], ys)
    print(f"window ECE on test: raw {raw:.3f} -> calibrated {cal:.3f}  (n={len(ys)})")


if __name__ == "__main__":
    main()
    