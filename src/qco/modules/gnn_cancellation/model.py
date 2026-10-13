"""Module B — GraphSAGE / GAT cancellation predictor (Sprint 2 / issue #11).

A message-passing stack embeds each gate; a scoring head ranks overlapping
gate pairs.  Training labels come from the Sprint-1 rule-based generator.
Until ``fit`` (or ``load``) has run, :meth:`GNNCancellationPredictor.predict`
falls back to that generator so the pipeline stays usable without torch.

Issue #28 adds a window-level (graph-pooled) head trained on exact
re-synthesis labels, with temperature-scaled confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import math

from qco.ir.intermediate_representation import PARAMETRIC, IntermediateRepresentation
from qco.modules.gnn_cancellation.dataset import load_training_dataset, sample_training_pairs
from qco.modules.gnn_cancellation.eval import (
    fit_temperature,
    held_out_split,
    metrics_from_counts,
    sigmoid,
)
from qco.modules.gnn_cancellation.features import (
    EDGE_FEATURE_DIM,
    FEATURE_DIM,
    CancellationCandidate,
    are_inverses,
    build_pyg_data,
    gate_pair_features,
    overlapping_gate_pairs,
    rule_based_candidates,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.utils import add_self_loops
except ImportError:  # pragma: no cover - CI installs only ``.[test]``
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    GATConv = SAGEConv = add_self_loops = None  # type: ignore[assignment]


_GNN_EXTRA = "GNNCancellationPredictor requires the `gnn` extra: pip install -e '.[gnn]'"

WINDOW_EXTRA_DIM = 6


def _window_extras(ir, s, e, cands=(), use_rules=True) -> list[float]:
    """Hand features: size, active qubits, 2q fraction + rule-candidate evidence."""
    gs = ir.gates[s:e]
    n = max(len(gs), 1)
    nq = len({q for g in gs for q in g.qubits})
    base = [len(gs) / 12.0, nq / 4.0, sum(1 for g in gs if len(g.qubits) == 2) / n]
    if not use_rules:
        return base + [0.0, 0.0, 0.0]
    inside = [c for c in cands if s <= c.a and c.b < e]
    merge = any(c.kind != "inverse_pair" for c in inside)
    return base + [math.log1p(len(inside)) / math.log(4.0), float(bool(inside)), float(merge)]

@dataclass(slots=True)
class GNNConfig:
    conv: str = "sage"  # "sage" | "gat"
    hidden: int = 128
    layers: int = 3
    dropout: float = 0.1
    epochs: int = 100
    lr: float = 1e-3
    rule_features: bool = True


@dataclass(slots=True)
class FitResult:
    losses: list[float]
    threshold: float
    n_circuits: int
    n_epochs: int


@dataclass(slots=True)
class WindowFitResult:
    losses: list[float]
    temperature: float
    n_val_windows: int


def _require_torch() -> None:
    if torch is None:
        raise ImportError(_GNN_EXTRA)


def _config_dict(config: GNNConfig) -> dict[str, Any]:
    return {
        "conv": config.conv,
        "hidden": config.hidden,
        "layers": config.layers,
        "dropout": config.dropout,
        "epochs": config.epochs,
        "lr": config.lr,
        "rule_features": config.rule_features,
    }


def _prepare_edges(edge_index, num_nodes: int):
    """Bidirected DAG + self-loops so both endpoints of a pair see each other."""
    if num_nodes == 0:
        return edge_index
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


if torch is not None:

    class CancellationGNN(nn.Module):
        """GraphSAGE or GAT encoder plus MLP heads over gate pairs and windows."""

        def __init__(self, config: GNNConfig, in_dim: int = FEATURE_DIM):
            super().__init__()
            conv_name = config.conv.lower()
            if conv_name not in {"sage", "gat"}:
                raise ValueError(f"GNNConfig.conv must be 'sage' or 'gat', got {config.conv!r}")
            if config.layers < 1:
                raise ValueError("GNNConfig.layers must be >= 1")
            hidden = config.hidden
            self.hidden = hidden
            self.convs = nn.ModuleList()
            src = in_dim
            for _ in range(config.layers):
                if conv_name == "gat":
                    self.convs.append(
                        GATConv(src, hidden, heads=1, dropout=config.dropout, concat=False)
                    )
                else:
                    self.convs.append(SAGEConv(src, hidden))
                src = hidden
            self.dropout = nn.Dropout(config.dropout)
            pair_in = hidden * 4 + EDGE_FEATURE_DIM
            self.scorer = nn.Sequential(
                nn.Linear(pair_in, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )
            # Window head (#28): mean || max pooled gate embeddings + hand features.
            self.window_scorer = nn.Sequential(
                nn.Linear(hidden * 2 + WINDOW_EXTRA_DIM, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )

        def encode(self, x, edge_index):
            if x.size(0) == 0:
                return x.new_zeros((0, self.hidden))
            edge_index = _prepare_edges(edge_index, x.size(0))
            h = x
            for conv in self.convs:
                h = conv(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            return h

        def score_pairs(self, h, pair_index, pair_attr):
            if pair_index.numel() == 0:
                return h.new_zeros((0,))
            left = h[pair_index[0]]
            right = h[pair_index[1]]
            feat = torch.cat(
                [left, right, (left - right).abs(), left * right, pair_attr],
                dim=-1,
            )
            return self.scorer(feat).squeeze(-1)

        def score_windows(self, h, windows, extras):
            """Graph-pooled (mean || max over the window's gate embeddings) + extras -> logit."""
            if not windows:
                return h.new_zeros((0,))
            pooled = torch.stack(
                [torch.cat([h[s:e].mean(0), h[s:e].max(0).values]) for s, e in windows]
            )
            return self.window_scorer(torch.cat([pooled, extras], dim=-1)).squeeze(-1)

else:  # pragma: no cover
    CancellationGNN = None  # type: ignore[misc, assignment]


def _infer_kind(ir: IntermediateRepresentation, i: int, j: int) -> str:
    gi, gj = ir.gates[i], ir.gates[j]
    if are_inverses(gi, gj):
        return "inverse_pair"
    if gi.name == gj.name and gi.qubits == gj.qubits:
        if gi.name in PARAMETRIC:
            return "rotation_merge"
        return "identity_chain"
    return "inverse_pair"


def _expand_indices(ir: IntermediateRepresentation, i: int, j: int, kind: str) -> tuple[int, ...]:
    if kind not in {"rotation_merge", "identity_chain"}:
        return ()
    gi = ir.gates[i]
    idxs = [
        k
        for k in range(i, j + 1)
        if ir.gates[k].name == gi.name and ir.gates[k].qubits == gi.qubits
    ]
    return tuple(idxs) if len(idxs) >= 2 else ()


def _candidate_for_pair(
    ir: IntermediateRepresentation, i: int, j: int, score: float
) -> CancellationCandidate:
    kind = _infer_kind(ir, i, j)
    return CancellationCandidate(i, j, kind, float(score), _expand_indices(ir, i, j, kind))


class GNNCancellationPredictor:
    def __init__(self, config: GNNConfig | None = None):
        self.config = config or GNNConfig()
        self._net = None
        self._threshold = 0.5
        self._temperature = 1.0  # window-head calibration (#28)
        self._rule_cache: dict = {}

    def fit(
        self,
        dataset,
        *,
        negative_ratio: int = 4,
        seed: int = 0,
        val_fraction: float = 0.2,
    ) -> FitResult:
        """Train on Sprint-1 rule-based labels and calibrate the predict threshold."""
        _require_torch()
        labelled = load_training_dataset(dataset)
        if not labelled:
            raise ValueError("fit() got an empty dataset")

        rng = random.Random(seed)
        torch.manual_seed(seed)

        if val_fraction > 0 and len(labelled) >= 4:
            train, val = held_out_split(labelled, test_fraction=val_fraction, seed=seed)
            if not val:
                val = train
        else:
            train, val = labelled, labelled

        net = CancellationGNN(self.config)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=self.config.lr)
        self._net = net

        losses: list[float] = []
        for _epoch in range(self.config.epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            running = 0.0
            steps = 0
            for idx in order:
                ir, gold = train[idx]
                pairs, labels = sample_training_pairs(
                    ir, gold, negative_ratio=negative_ratio, rng=rng
                )
                if not pairs:
                    continue
                running += self._train_step(ir, pairs, labels, opt)
                steps += 1
            losses.append(running / max(steps, 1))

        net.eval()
        self._threshold = self._calibrate_threshold(val)
        return FitResult(
            losses=losses,
            threshold=self._threshold,
            n_circuits=len(train),
            n_epochs=self.config.epochs,
        )

    def predict(
        self, ir: IntermediateRepresentation, threshold: float | None = None
    ) -> list[CancellationCandidate]:
        """Ranked cancellation candidates (highest score first)."""
        thr = self._threshold if threshold is None else threshold
        if self._net is None:
            cands = rule_based_candidates(ir)
            return sorted((c for c in cands if c.score >= thr), key=lambda c: -c.score)

        _require_torch()
        self._net.eval()
        pairs, logits = self._pair_logits(ir)
        if not pairs:
            return []
        scores = torch.sigmoid(logits).tolist()
        cands = [
            _candidate_for_pair(ir, i, j, score)
            for (i, j), score in zip(pairs, scores)
            if score >= thr
        ]
        cands.sort(key=lambda c: -c.score)
        return cands

    # -- window head (#28) ---------------------------------------------------

    def fit_windows(
        self, train, val, *, epochs: int | None = None, seed: int = 0
    ) -> WindowFitResult:
        """Train the window head on re-synthesis labels, then fit the temperature on ``val``.

        ``train`` / ``val``: ``list[(ir, list[WindowLabel])]``, split BY CIRCUIT.
        Windows with ``label is None`` (timeouts etc.) are skipped.
        """
        _require_torch()
        rng = random.Random(seed)
        torch.manual_seed(seed)
        if self._net is None:
            self._net = CancellationGNN(self.config)
        net = self._net
        opt = torch.optim.Adam(net.parameters(), lr=self.config.lr)

        losses: list[float] = []
        for _ in range(epochs or self.config.epochs):
            net.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            running, steps = 0.0, 0
            for i in order:
                ir, rows = train[i]
                rows = [r for r in rows if r.label is not None]
                if not rows:
                    continue
                y = torch.tensor([float(r.label) for r in rows], dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                logits = self._window_forward(ir, [(r.start, r.end) for r in rows])
                # Plain BCE (no pos_weight): pos_weight shifts logits, which a
                # temperature cannot undo.
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                opt.step()
                running += float(loss.item())
                steps += 1
            losses.append(running / max(steps, 1))

        net.eval()
        z: list[float] = []
        y_val: list[int] = []
        for ir, rows in val:
            rows = [r for r in rows if r.label is not None]
            if not rows:
                continue
            z += self.window_logits(ir, [(r.start, r.end) for r in rows])
            y_val += [int(r.label) for r in rows]
        self._temperature = fit_temperature(z, y_val)
        return WindowFitResult(losses, self._temperature, len(y_val))

    def window_logits(self, ir: IntermediateRepresentation, windows) -> list[float]:
        """Raw (uncalibrated) window logits."""
        _require_torch()
        if self._net is None:
            raise RuntimeError("window head not trained; call fit_windows() or load() first")
        windows = list(windows)
        if not windows:
            return []
        self._net.eval()
        with torch.no_grad():
            return self._window_forward(ir, windows).tolist()

    def score_windows(self, ir: IntermediateRepresentation, windows) -> list[float]:
        """Calibrated P(window is reducible by exact re-synthesis)."""
        t = self._temperature
        return [sigmoid(z / t) for z in self.window_logits(ir, windows)]

    def rank_windows(self, ir: IntermediateRepresentation, windows, top_k: int | None = None):
        """``[((start, end), prob), ...]`` sorted by calibrated confidence, best first."""
        windows = list(windows)
        ranked = sorted(zip(windows, self.score_windows(ir, windows)), key=lambda t: -t[1])
        return ranked[:top_k] if top_k else ranked

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write config + weights + calibrated threshold/temperature to ``path``."""
        _require_torch()
        if self._net is None:
            raise RuntimeError("nothing to save; call fit() or load() first")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": _config_dict(self.config),
                "state_dict": self._net.state_dict(),
                "threshold": self._threshold,
                "temperature": self._temperature,
                "feature_dim": FEATURE_DIM,
            },
            dest,
        )
        return dest

    def load(self, path: str | Path) -> "GNNCancellationPredictor":
        """Restore a checkpoint produced by :meth:`save`. Returns ``self``."""
        _require_torch()
        blob = torch.load(Path(path), map_location="cpu")
        cfg = dict(blob["config"])
        self.config = GNNConfig(**{k: cfg[k] for k in _config_dict(GNNConfig()) if k in cfg})
        if blob.get("feature_dim") not in (None, FEATURE_DIM):
            raise ValueError(
                f"checkpoint feature_dim={blob['feature_dim']} != FEATURE_DIM={FEATURE_DIM}"
            )
        net = CancellationGNN(self.config)
        # strict=False so pre-#28 checkpoints (no window_scorer) still load.
        res = net.load_state_dict(blob["state_dict"], strict=False)
        bad = [k for k in res.missing_keys if not k.startswith("window_scorer")] + list(
            res.unexpected_keys
        )
        if bad:
            raise ValueError(f"checkpoint mismatch: {bad}")
        net.eval()
        self._net = net
        self._threshold = float(blob.get("threshold", 0.5))
        self._temperature = float(blob.get("temperature", 1.0))
        return self

    # -- internals -----------------------------------------------------------

    def _graph_tensors(self, ir: IntermediateRepresentation):
        data = build_pyg_data(ir)
        return data.x, data.edge_index

    def _pair_tensors(self, ir: IntermediateRepresentation, pairs: list[tuple[int, int]]):
        if not pairs:
            empty_idx = torch.zeros((2, 0), dtype=torch.long)
            empty_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
            return empty_idx, empty_attr
        pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        pair_attr = torch.tensor(
            [gate_pair_features(ir, i, j) for i, j in pairs],
            dtype=torch.float32,
        ).reshape(len(pairs), EDGE_FEATURE_DIM)
        return pair_index, pair_attr

    def _pair_logits(self, ir: IntermediateRepresentation):
        assert self._net is not None
        pairs = overlapping_gate_pairs(ir)
        x, edge_index = self._graph_tensors(ir)
        pair_index, pair_attr = self._pair_tensors(ir, pairs)
        with torch.no_grad():
            h = self._net.encode(x, edge_index)
            logits = self._net.score_pairs(h, pair_index, pair_attr)
        return pairs, logits

    def _rules(self, ir):
        if not self.config.rule_features:
            return []
        key = (ir.name, len(ir.gates))
        if key not in self._rule_cache:
            self._rule_cache[key] = rule_based_candidates(ir)
        return self._rule_cache[key]

    def _window_forward(self, ir, windows):
        """Window logits WITH grad (callers wrap in no_grad for inference)."""
        assert self._net is not None
        x, edge_index = self._graph_tensors(ir)
        cands = self._rules(ir)
        extras = torch.tensor(
            [_window_extras(ir, s, e, cands, self.config.rule_features) for s, e in windows],
            dtype=torch.float32,
        ).reshape(len(windows), WINDOW_EXTRA_DIM)
        h = self._net.encode(x, edge_index)
        return self._net.score_windows(h, windows, extras)

    def _train_step(
        self,
        ir: IntermediateRepresentation,
        pairs: list[tuple[int, int]],
        labels: list[float],
        opt,
    ) -> float:
        assert self._net is not None
        self._net.train()
        x, edge_index = self._graph_tensors(ir)
        pair_index, pair_attr = self._pair_tensors(ir, pairs)
        y = torch.tensor(labels, dtype=torch.float32)
        n_pos = float(y.sum().item())
        n_neg = float(len(labels) - n_pos)
        if n_pos == 0.0 or n_neg == 0.0:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        h = self._net.encode(x, edge_index)
        logits = self._net.score_pairs(h, pair_index, pair_attr)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
        return float(loss.item())

    def _calibrate_threshold(self, labelled) -> float:
        """Pick the threshold that maximises F1 vs rule labels on ``labelled``."""
        assert self._net is not None
        self._net.eval()
        scored: list[tuple[float, int]] = []
        for ir, gold in labelled:
            gold_set = {(c.a, c.b) for c in gold}
            pairs, logits = self._pair_logits(ir)
            probs = torch.sigmoid(logits).tolist() if pairs else []
            for pair, prob in zip(pairs, probs):
                scored.append((prob, 1 if pair in gold_set else 0))
        if not scored:
            return 0.5

        best_t, best_f1 = 0.5, -1.0
        for step in range(1, 20):
            t = step / 20.0
            tp = fp = fn = 0
            for prob, label in scored:
                pred = prob >= t
                if pred and label:
                    tp += 1
                elif pred and not label:
                    fp += 1
                elif (not pred) and label:
                    fn += 1
            f1 = metrics_from_counts(tp, fp, fn).f1
            if f1 > best_f1 or (f1 == best_f1 and abs(t - 0.5) < abs(best_t - 0.5)):
                best_t, best_f1 = t, f1
        return best_t