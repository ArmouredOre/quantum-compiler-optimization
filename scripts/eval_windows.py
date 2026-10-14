#!/usr/bin/env python
"""Issue #28: window-ranking vs random / rule-based / length-only baselines, plus calibration
(NLL, Brier, ECE) before/after temperature scaling.

Two cross-validation modes, both with every window scored out-of-fold:
  * default: leave-one-family-out (all circuits of a family, e.g. grover_*, are held out together)
  * --no-family-folds: leave-workload-out (sibling circuits such as grover_w5_i1 / grover_w5_i2
    always share a fold, so near-duplicates cannot leak from train into test)
"""
import argparse, math, random, re, statistics as st
from collections import Counter
from pathlib import Path

from qco.ir.parser import from_qasm2
from qco.modules.gnn_cancellation.dataset import dump_resynth_labels, load_resynth_labels
from qco.modules.gnn_cancellation.eval import (
    expected_calibration_error as ece, fit_temperature, plot_reliability,
    precision_at_k, rule_window_scorer, sigmoid, write_pr_table)
from qco.modules.gnn_cancellation.model import GNNCancellationPredictor, GNNConfig

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5)


# ----------------------------------------------------------------------------- metrics
def average_precision(scores, labels, rng, repeats=20):
    """AP with random tie-breaking (averaged), so tie-heavy systems get no positional bonus."""
    pairs = list(zip(scores, labels))
    n_pos = sum(y for _, y in pairs)
    if not n_pos:
        return 0.0
    total = 0.0
    for _ in range(repeats):
        rng.shuffle(pairs)
        ranked = sorted(pairs, key=lambda t: -t[0])
        hits, ap = 0, 0.0
        for rank, (_, y) in enumerate(ranked, 1):
            if y:
                hits += 1
                ap += hits / rank
        total += ap / n_pos
    return total / repeats


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties."""
    n = len(scores)
    n_pos = sum(labels); n_neg = n - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def nll(probs, labels):
    eps = 1e-12
    return -sum(y * math.log(min(max(p, eps), 1 - eps)) + (1 - y) * math.log(min(max(1 - p, eps), 1 - eps))
                for p, y in zip(probs, labels)) / max(len(labels), 1)


def brier(probs, labels):
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / max(len(labels), 1)


# ----------------------------------------------------------------------------- folds / CV
def family(name):
    """grover_w5_i1 -> grover, rand_cnotpauli_n10 -> rand_cnotpauli"""
    return re.split(r"_[wni]\d", name)[0]


def workload(name):
    """grover_w5_i1 -> grover_w5 (strip the instance suffix so siblings stay together)."""
    return re.sub(r"_i\d+$", "", name)


def make_folds(data, k, seed, by_family=True):
    """Assign circuits to k folds. Groups (families, or workloads in circuit mode) are never
    split across folds; groups with positives are spread out first."""
    rng = random.Random(seed)
    groups = {}
    for i, (ir, _) in enumerate(data):
        groups.setdefault(family(ir.name) if by_family else workload(ir.name), []).append(i)
    keys = list(groups)
    has_pos = {g: any(r.label == 1 for i in groups[g] for r in data[i][1]) for g in keys}
    pos = [g for g in keys if has_pos[g]]; neg = [g for g in keys if not has_pos[g]]
    rng.shuffle(pos); rng.shuffle(neg)
    folds = [[] for _ in range(k)]
    for j, g in enumerate(pos + neg):
        folds[j % k].extend(groups[g])
    return folds


def out_of_fold_logits(data, epochs, k, seed, by_family, rule_features=True):
    oof, fold_of = {}, {}
    for f, held in enumerate(make_folds(data, k, seed, by_family)):
        if not held:
            continue
        held_set = set(held)
        train = [data[i] for i in range(len(data)) if i not in held_set]
        pred = GNNCancellationPredictor(GNNConfig(epochs=epochs, rule_features=rule_features))
        pred.fit_windows(train, [], seed=seed + f)       # T is fitted later, cross-fit
        for i in held:
            ir, rs = data[i]
            oof[i] = pred.window_logits(ir, [(r.start, r.end) for r in rs])
            fold_of[i] = f
    return oof, fold_of


def crossfit_probs(data, oof, fold_of):
    """Calibrated probs: fold f's temperature is fitted on the OTHER folds' out-of-fold logits."""
    Ts = {}
    for f in sorted(set(fold_of.values())):
        z = [zz for i in oof if fold_of[i] != f for zz, r in zip(oof[i], data[i][1]) if r.label is not None]
        y = [r.label for i in oof if fold_of[i] != f for r in data[i][1] if r.label is not None]
        Ts[f] = fit_temperature(z, y) if len(set(y)) > 1 else 1.0
    probs = [[sigmoid(v / Ts[fold_of[i]]) for v in oof[i]] for i in range(len(data))]
    return probs, list(Ts.values())


def p1_diff(data, idx, sa, sb, rng):
    """P@1 difference (A - B) over resampled circuits; ties broken randomly, as in evaluate()."""
    def p1(sc):
        hits = n = 0
        for i in idx:
            pairs = [(x, r.label) for x, r in zip(sc[i], data[i][1]) if r.label is not None]
            if not pairs:
                continue
            rng.shuffle(pairs)
            hits += max(pairs, key=lambda t: t[0])[1]
            n += 1
        return hits / max(n, 1)
    return p1(sa) - p1(sb)

def paired_bootstrap(data, sa, sb, n_boot=1000, seed=0):
    """Resample circuits with replacement; return {metric: (mean diff A-B, lo95, hi95, n_valid)}."""
    rng = random.Random(seed)

    def flat(idx, sc):
        s, y = [], []
        for i in idx:
            for x, r in zip(sc[i], data[i][1]):
                if r.label is not None:
                    s.append(x); y.append(r.label)
        return s, y

    diffs = {"AP": [], "AUROC": [], "P@1": []}
    for _ in range(n_boot):
        idx = [rng.randrange(len(data)) for _ in range(len(data))]
        a_, y = flat(idx, sa); b_, _ = flat(idx, sb)
        if sum(y) == 0 or sum(y) == len(y):
            continue
        diffs["AP"].append(average_precision(a_, y, rng, 3) - average_precision(b_, y, rng, 3))
        diffs["AUROC"].append(auroc(a_, y) - auroc(b_, y))
        diffs["P@1"].append(p1_diff(data, idx, sa, sb, rng))
    out = {}
    for m, d in diffs.items():
        d.sort()
        out[m] = ((st.mean(d), d[int(.025 * len(d))], d[min(int(.975 * len(d)), len(d) - 1)], len(d))
                  if d else (float("nan"),) * 3 + (0,))
    return out


def circuit_scores(data, fn):
    return [list(fn(i, ir, [(r.start, r.end) for r in rs])) for i, (ir, rs) in enumerate(data)]


def evaluate(data, scores, seed, repeats=20):
    """scores: per-circuit list of per-window scores. Returns dict of metrics.

    hits@1 / hits@1* are the expected number of circuits whose top-ranked window is reducible
    (P@1 times the number of circuits), so P@1 differences can be read as "about N circuits".
    """
    scored = [[(s, r.label) for s, r in zip(sc, rs)] for sc, (_, rs) in zip(scores, data)]
    pos_only = [c for c in scored if any(y == 1 for _, y in c)]
    n_known = sum(1 for c in scored if any(y is not None for _, y in c))
    out = {}
    for k in KS:
        for tag, pc in (("", scored), ("*", pos_only)):
            out[f"P@{k}{tag}"] = sum(precision_at_k(pc, k, random.Random(seed * 1000 + j))
                                     for j in range(repeats)) / repeats
    out["hits@1"] = out["P@1"] * n_known
    out["hits@1*"] = out["P@1*"] * len(pos_only)
    s = [x for c in scored for x, y in c if y is not None]
    y = [y for c in scored for _, y in c if y is not None]
    out["AP"] = average_precision(s, y, random.Random(seed))
    out["AUROC"] = auroc(s, y)
    return out


# ----------------------------------------------------------------------------- timeout probe
def probe_timeouts(data, n, timeout_s, max_qubits, max_gates):
    from qco.modules.smt_verifier.bounded_optimizer import BoundedResynthesizer, ResynthConfig
    from qco.ir.intermediate_representation import IntermediateRepresentation
    rs = BoundedResynthesizer(ResynthConfig(max_qubits=max_qubits, max_gates=max_gates, timeout_s=timeout_s))
    unk = [(ir, r) for ir, rows in data for r in rows if r.label is None and r.status == "timeout"]
    random.Random(0).shuffle(unk)
    print(f"probing {min(n, len(unk))} of {len(unk)} timed-out windows with timeout={timeout_s}s")
    outcomes = Counter()
    for ir, r in unk[:n]:
        w = IntermediateRepresentation(num_qubits=ir.num_qubits,
                                       gates=list(ir.gates[r.start:r.end]), name="probe")
        o = rs.resynthesize_outcome(w, timeout_s=timeout_s)
        outcomes[(r.end - r.start, o.optimality, bool(o.reduced))] += 1
        print(f"  len={r.end - r.start} optimality={o.optimality} reduced={o.reduced} solve_s={o.solve_s:.1f}")
    print("summary (len, optimality, reduced) -> count:")
    for k, v in sorted(outcomes.items()):
        print("  ", k, v)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=Path, default=ROOT / "benchmarks" / "circuits")
    ap.add_argument("--labels-out", type=Path, default=ROOT / "data" / "labels_resynth")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "gnn_eval" / "window_ranking.md")
    ap.add_argument("--max-qubits", type=int, default=3)
    ap.add_argument("--max-gates", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="number of CV repetitions")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--reuse-labels", action="store_true",
                    help="load existing JSON labels from --labels-out instead of re-running the solver")
    ap.add_argument("--timeouts-as-negative", action="store_true",
                    help="label timed-out windows 0 ('no reduction found within budget') instead of dropping them")
    ap.add_argument("--no-family-folds", action="store_true",
                    help="leave-workload-out (siblings grouped) instead of leave-one-family-out")
    ap.add_argument("--probe-timeouts", type=int, default=0,
                    help="re-solve N timed-out windows with --probe-timeout seconds, then exit")
    ap.add_argument("--probe-timeout", type=float, default=30.0)
    ap.add_argument("--no-rule-features", action="store_true",
                    help="train the window head without the 3 rule-evidence features")
    ap.add_argument("--ablate", action="store_true",
                    help="also run a GNN without rule features and compare it with the full model")
    ap.add_argument("--labels-only", action="store_true",
                    help="generate/load labels, print diagnostics, then exit (no training)")
    a = ap.parse_args()
    by_family = not a.no_family_folds
    rule_feats = not a.no_rule_features
    ablate = a.ablate and rule_feats

    # ---- labels ----
    if a.reuse_labels and a.labels_out.exists() and any(a.labels_out.glob("*.json")):
        data = load_resynth_labels(a.labels_out)
        print(f"loaded labels for {len(data)} circuits from {a.labels_out}")
    else:
        irs = []
        for p in sorted(a.circuits.glob("*.qasm")):
            ir = from_qasm2(p.read_text(encoding="utf-8")); ir.name = p.stem; irs.append(ir)
        data = dump_resynth_labels(irs, a.labels_out, timeout_s=a.timeout,
                                   max_qubits=a.max_qubits, max_gates=a.max_gates)

    if a.probe_timeouts:
        probe_timeouts(data, a.probe_timeouts, a.probe_timeout, a.max_qubits, a.max_gates)
        return

    if a.timeouts_as_negative:
        for _, rs_ in data:
            for r in rs_:
                if r.label is None and r.status == "timeout":
                    r.label = 0

    rows = [r for _, rs_ in data for r in rs_]
    n_pos = sum(r.label == 1 for r in rows)
    n_neg = sum(r.label == 0 for r in rows)
    n_unk = sum(r.label is None for r in rows)
    print(f"{len(rows)} windows: reducible={n_pos} not={n_neg} unknown={n_unk}"
          f"{'  (timeouts counted as not-reducible)' if a.timeouts_as_negative else ''}")

    # ---- diagnostics ----
    print("positives per circuit:",
          {ir.name: sum(r.label == 1 for r in rs_) for ir, rs_ in data if any(r.label == 1 for r in rs_)})
    fams = Counter(family(ir.name) for ir, _ in data)
    print(f"{len(fams)} families:", dict(fams))
    by_len = Counter((r.end - r.start, "unknown" if r.label is None else r.label) for r in rows)
    for L in sorted({l for l, _ in by_len}):
        tot = sum(v for (l, _), v in by_len.items() if l == L)
        print(f"  len {L}: n={tot} reducible={by_len[(L, 1)]} not={by_len[(L, 0)]} "
              f"unknown={by_len[(L, 'unknown')]}")
    rule_hit = [(r.label, rule_window_scorer(ir, [(r.start, r.end)])[0] > 0)
                for ir, rs_ in data for r in rs_ if r.label is not None]
    pos_cov = sum(1 for y, h in rule_hit if y == 1 and h)
    print(f"positives with a rule candidate inside: {pos_cov}/{n_pos}  "
          f"(solver-only positives: {n_pos - pos_cov})")
    print(f"windows with a rule candidate: {sum(h for _, h in rule_hit)} "
          f"of which reducible: {sum(1 for y, h in rule_hit if h and y == 1)}")
    if a.labels_only:
        return  

    # ---- fold setup ----
    n_groups = len(fams) if by_family else len({workload(ir.name) for ir, _ in data})
    k_folds = n_groups if by_family else min(a.folds, n_groups)   # family mode = true leave-one-family-out
    if k_folds < 2:
        raise SystemExit("need at least 2 groups for cross-validation")
    if n_pos < 20:
        print(f"WARNING: only {n_pos} positive windows - treat all numbers as exploratory.")

    # ---- CV over seeds ----
    baselines = {
        "Random": lambda i, ir, w: [0.0] * len(w),
        "Rule-based (candidates in window)": lambda i, ir, w: rule_window_scorer(ir, w),
        "Window length only": lambda i, ir, w: [float(e - s) for s, e in w],
    }
    names = [*baselines, "GNN window head", "Rule + GNN tie-break"]
    if ablate:
        names.append("GNN (no rule features)")
    boot_scores = {}
    res = {n: {} for n in names}
    cal_rows = []                    # per-seed calibration metrics
    temps, first_curves = [], None
    fam_auc = {}                     # family -> per-seed GNN AUROC (out-of-fold)
    fam_pos = {}
    base_scores = {n: circuit_scores(data, fn) for n, fn in baselines.items()}

    def add(name, metrics):
        for m, v in metrics.items():
            res[name].setdefault(m, []).append(v)

    for s in range(a.seeds):
        seed = a.seed + s
        oof, fold_of = out_of_fold_logits(data, a.epochs, k_folds, seed, by_family, rule_feats)

        # per-family diagnosis: separates data starvation (Grover only) from a bug (all families < 0.5)
        for fam in sorted(fams):
            idx = [i for i, (ir, _) in enumerate(data) if family(ir.name) == fam]
            zf = [zz for i in idx for zz, r in zip(oof[i], data[i][1]) if r.label is not None]
            yf = [r.label for i in idx for r in data[i][1] if r.label is not None]
            fam_pos[fam] = sum(yf)
            if sum(yf):
                a_f = auroc(zf, yf)
                fam_auc.setdefault(fam, []).append(a_f)
                print(f"  [seed {seed}] {fam}: pos={sum(yf)} auroc={a_f:.3f}")

        gnn_scores, Ts = crossfit_probs(data, oof, fold_of)
        z, y, cal = [], [], []
        for i, (ir, rs_) in enumerate(data):
            for zz, pp, r in zip(oof[i], gnn_scores[i], rs_):
                if r.label is not None:
                    z.append(zz); y.append(r.label); cal.append(pp)
        T = st.mean(Ts)
        if len(set(y)) < 2:
            print(f"[seed {seed}] WARNING: single-class labels, temperature fell back to 1.0")
        raw = [sigmoid(v) for v in z]
        cal_rows.append(dict(T=T, ece_raw=ece(raw, y), ece_cal=ece(cal, y),
                             nll_raw=nll(raw, y), nll_cal=nll(cal, y),
                             brier_raw=brier(raw, y), brier_cal=brier(cal, y)))
        temps.append(T)
        if first_curves is None:
            first_curves = {"raw": (raw, y), "temp-scaled (cross-fit)": (cal, y)}
        c = cal_rows[-1]
        print(f"[seed {seed}] T={T:.3f} NLL {c['nll_raw']:.4f}->{c['nll_cal']:.4f} "
              f"Brier {c['brier_raw']:.4f}->{c['brier_cal']:.4f} "
              f"ECE {c['ece_raw']:.4f}->{c['ece_cal']:.4f} ({len(y)} windows)")

        for n in baselines:
            add(n, evaluate(data, base_scores[n], seed))
        add("GNN window head", evaluate(data, gnn_scores, seed))
        # rule count dominates (GNN prob < 1), so the GNN only orders windows with equal rule counts
        rule_sc = base_scores["Rule-based (candidates in window)"]
        hybrid = [[r_ + p for r_, p in zip(rr, pp)] for rr, pp in zip(rule_sc, gnn_scores)]
        add("Rule + GNN tie-break", evaluate(data, hybrid, seed))
        nr_scores = None
        if ablate:
            oof2, fo2 = out_of_fold_logits(data, a.epochs, k_folds, seed, by_family, False)
            nr_scores, _ = crossfit_probs(data, oof2, fo2)
            add("GNN (no rule features)", evaluate(data, nr_scores, seed))
        for name_, sc_ in (("GNN window head", gnn_scores), ("Rule + GNN tie-break", hybrid),
                           ("GNN (no rule features)", nr_scores)):
            if sc_ is None:
                continue
            acc = boot_scores.setdefault(name_, [[0.0] * len(c) for c in sc_])
            for ci, c in enumerate(sc_):
                for wi, v in enumerate(c):
                    acc[ci][wi] += v / a.seeds
        boot_scores["Rule-based (candidates in window)"] = rule_sc
        boot_scores["GNN (no rule features)"] = nr_scores

    # ---- report ----
    def fmt(vals):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return "n/a"
        return f"{st.mean(vals):.3f} ± {st.pstdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"

    def table(cols, labels=None):
        labels = labels or cols
        t = "| System | " + " | ".join(labels) + " |\n|---|" + "---|" * len(cols) + "\n"
        for n in names:
            t += f"| {n} | " + " | ".join(fmt(res[n][c]) for c in cols) + " |\n"
        return t

    n_pos_circ = sum(1 for _, rs_ in data if any(r.label == 1 for r in rs_))
    cal_mean = {k: st.mean(r[k] for r in cal_rows) for k in cal_rows[0]}
    label_note = ("timeouts counted as not-reducible" if a.timeouts_as_negative
                  else f"timeouts ({n_unk} windows) excluded")
    mode = "family" if by_family else "workload"
    p0 = n_pos / max(n_pos + n_neg, 1)
    base_brier = p0 * (1 - p0)
    header = (
        f"## Window ranking + calibration ({k_folds}-fold leave-{mode}-out CV, "
        f"{a.seeds} seed(s) from {a.seed})\n\n"
        f"- Labels: {n_pos} reducible / {n_neg} not-reducible; {label_note}. 'Not reducible' means no "
        f"reduction found within the gate set, angle pool and time budget, not provably optimal.\n"
        f"- Positives come from {n_pos_circ} of {len(data)} circuits; rule candidates cover "
        f"{pos_cov}/{n_pos} positives.\n"
        f"- Folds: {'whole families are held out together' if by_family else 'sibling instances (e.g. grover_w5_i1/i2) share a fold, so near-duplicates cannot leak'}.\n"
        f"- ± is the std over CV seeds / tie-breaks only; it does not capture the small number of "
        f"positive circuits, so treat differences of a few points as noise.\n"
        f"- P@k pools hits over circuits (each circuit contributes min(k, #windows) slots). Rule-based "
        f"scores are heavily tied, so its P@1 is an expectation over random tie-breaks.\n"
        f"- Calibration (pooled out-of-fold; T cross-fitted, each fold's T fitted on the other folds' "
        f"logits): mean T = {fmt(temps)}; NLL {cal_mean['nll_raw']:.4f} -> {cal_mean['nll_cal']:.4f}; "
        f"Brier {cal_mean['brier_raw']:.4f} -> {cal_mean['brier_cal']:.4f} "
        f"(constant base-rate Brier = {base_brier:.4f}); "
        f"ECE {cal_mean['ece_raw']:.4f} -> {cal_mean['ece_cal']:.4f}. Temperature scaling is monotone, "
        f"so it does not change ranking metrics.\n"
        f"- ECE uses 10 equal-width bins and only {n_pos} positives, so most bins are nearly empty; "
        f"treat it as indicative, not conclusive.\n")
    fam_table = ("| Family | Positives | GNN AUROC (out-of-fold) |\n|---|---|---|\n"
                 + "".join(f"| {f} | {fam_pos[f]} | {fmt(v)} |\n" for f, v in sorted(fam_auc.items())))
    pairs = [("GNN window head", "Rule-based (candidates in window)"),
             ("Rule + GNN tie-break", "Rule-based (candidates in window)"),
             ("Rule + GNN tie-break", "GNN window head")]
    if ablate:
        pairs.append(("GNN window head", "GNN (no rule features)"))
    boot = ("\n### Paired bootstrap over circuits (seed-averaged scores, 1000 resamples, 95% CI)\n\n"
            "| A vs B | ΔAP | ΔAUROC | ΔP@1 |\n|---|---|---|---|\n")
    for A, B in pairs:
        r_ = paired_bootstrap(data, boot_scores[A], boot_scores[B], seed=a.seed)
        cell = lambda m: f"{r_[m][0]:+.3f} [{r_[m][1]:+.3f}, {r_[m][2]:+.3f}]"
        boot += f"| {A} vs {B} | {cell('AP')} | {cell('AUROC')} | {cell('P@1')} |\n"
    boot += "\nA CI that spans 0 means the difference is not distinguishable from noise at this circuit count.\n"
    body = (f"Base rate of reducible windows: {n_pos}/{n_pos + n_neg}\n\n"
            "### Ranking (all circuits; AP/AUROC pooled over windows)\n\n"
            + table([f"P@{k}" for k in KS] + ["AP", "AUROC"]) +
            "\n### P@k restricted to circuits with at least one positive window\n\n"
            + table([f"P@{k}*" for k in KS]) +
            f"\n### Top-1 hits (expected number of circuits whose top window is reducible)\n\n"
            f"Denominators: {len(data)} circuits (all), {n_pos_circ} circuits with a positive (*).\n\n"
            + table(["hits@1", "hits@1*"], ["hits@1 (all)", "hits@1* (positive circuits)"]) +
            "\n### Per-family GNN AUROC (families with positives)\n\n" + fam_table + boot)
    write_pr_table(a.out, body, header=header)
    try:
        plot_reliability(first_curves, a.out.with_suffix(".png"))
    except ImportError as e:
        print(f"skipping reliability plot ({e}); pip install matplotlib")
    print(header + "\n" + body)


if __name__ == "__main__":
    main()
    