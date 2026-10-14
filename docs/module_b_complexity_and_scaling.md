# Module B: Complexity and Scaling

Time and memory costs of the GNN cancellation module (`src/qco/modules/gnn_cancellation/`), read from the code. Nothing here is measured. Mirrors `docs/module_c_complexity_and_scaling.md`.

## Notation

G gates, Q qubits, E DAG edges, E′ = 2E + G edges after `_prepare_edges`, P overlapping gate pairs (P ≤ G(G−1)/2), S sampled training pairs, W windows (lengths wᵢ), C rule candidates, L conv layers, d hidden width, F = `FEATURE_DIM` = 35, N_train training circuits, M scored validation items (pairs or windows). Gate arity is at most 3, so per-gate set operations are O(1).

Costs of the PyG layers (`SAGEConv`, `GATConv`) are the library's standard costs, not read from this repo.

## 1. Graph Construction

1. **ASAP layers.** `_asap_layers` makes one pass over the gates, touching at most 3 qubits each. O(G + Q).
2. **Node features.** `gate_node_features` builds a one-hot plus a fixed number of scalars: O(F) per gate, O(G·F) total. The `torch.tensor` call is also O(G·F).
3. **DAG.** `build_dag` is one pass over the gates, keeping the last gate seen on each qubit. A gate calls `add_edge(prev, idx)` once per qubit that has a previous writer, so there are at most 3G insertions. `GateDAG` stores `succ` and `pred` as `dict[int, set[int]]`, so each insertion is O(1) expected, and duplicate edges (gates sharing more than one qubit) are merged. Hence E ≤ 3G, with no self-loops since every edge points forward. Time O(G), memory O(E) = O(G).
4. **Edge loop.** For each node `u`, `sorted(dag.succ[u])` sorts a set of at most 3 elements, then `_edge_features` is called once per edge. Both are O(1) per edge, so O(E) = O(G) in total.
5. **Edge tensors.** `edge_index` is 2E integers, `edge_attr` is E × 4 floats.
6. **Edge preparation.** `_prepare_edges` concatenates the reversed edges and adds G self-loops: E′ = 2E + G ≤ 7G edges, built in O(G).

Total: O(G·F + Q) time and O(G·F) memory.

Two properties matter for scaling:

- `edge_attr` is built but never used by the encoder. `_graph_tensors` returns only `x` and `edge_index`, and the convolutions are called without edge features. The E calls to `_edge_features` are wasted on the model path.
- The graph is rebuilt on every call. `_pair_logits`, `_window_forward` and `_train_step` each call `build_pyg_data` with no caching.

## 2. Pair Scoring

1. **Enumeration.** `overlapping_gate_pairs` builds G support sets, then tests every i < j for a shared qubit. This is Θ(G²) regardless of P. Output memory is O(P).
2. **Pair features.** `_pair_tensors` calls `gate_pair_features` (which is `_edge_features`) once per pair. O(P) time and memory.
3. **Encoding.** `encode` runs L convolutions over G nodes and E′ ≤ 7G edges. The cost is O(G·(F·d + L·d²)), linear in G. The first layer takes F inputs and the rest take d.
4. **Pair head.** `score_pairs` concatenates [left, right, |left−right|, left·right, pair_attr], which is 4d + 4 floats per pair (516 at d = 128). The first MLP layer costs O(P·d²) and the output layer O(P·d). Peak memory for this step is O(P·d).
5. **Thresholding.** `predict` keeps pairs with score ≥ threshold. For each kept pair, `_candidate_for_pair` calls `_expand_indices`, which scans range(i, j+1): O(j − i) per kept pair. Then a sort, O(K log K) for K kept pairs.

Total for `predict`: Θ(G²) + O(G·(F·d + L·d²)) + O(P·d²). In the worst case P = Θ(G²), giving Θ(G²·d²). Because the encoder is linear in G, the pair path (enumeration and head) is the scaling bottleneck for large circuits. The enumeration loop runs before any model call, so a higher threshold does not avoid it.

## 3. Window Ranking

**Windows.** When windows come from `extract_windows` (defaults `max_gates=6`, `stride=3`, `max_qubits=3`), extraction is one pass costing O(G·max_gates/stride) = O(G). It yields W ≤ ⌈G/stride⌉ windows of at most 6 gates each, so Σwᵢ ≤ 6W = O(G). These bounds hold only for extractor-produced windows with these defaults.

**Rule candidates.** `rule_based_candidates` has two parts:

- The pair loop tests every i < j with `are_inverses`, Θ(G²). Each inverse pair also pays `_wire_clear` or `_can_commute_together`, an O(j − i) scan. Worst case O(G³), for example a long run of one self-inverse gate on one wire, where every pair is an inverse pair.
- The same-support run splitting adds at most O(G²) (the gaps checked per support group sum to at most G, over at most G groups).

The candidate count C is O(G²) in the worst case, because each inverse pair is a candidate, and is usually small.

`_rules` caches the result per circuit, keyed by gate contents. Building the key is O(G) on every call. The cache is cleared once it holds more than 256 entries. When `rule_features=False`, `_rules` returns an empty list without computing anything.

**Features.** `_window_extras` slices the window and counts active qubits, O(w). With `rule_features=True` it also scans all C candidates, so O(w + C) per window and O(Σw + W·C) overall.

**Forward pass.** `_window_forward` builds the graph (Section 1), gets rule candidates, builds extras, runs `encode` (Section 2), then `CancellationGNN.score_windows`. That function mean-pools and max-pools each window's embeddings in a Python loop, O(wᵢ·d) per window, then applies the window MLP, O(d²) per window. Total: O(G·(F·d + L·d²)) + O(Σw·d + W·d²) + O(Σw + W·C) for extras, plus W Python-level iterations. There is no pair enumeration, so the Θ(G²) term from Section 2 does not appear here. With `rule_features=True`, the rule-candidate cost appears instead.

**Entry points.**

- `window_logits` is one no-grad forward pass.
- `score_windows` is `window_logits` plus an O(W) sigmoid with temperature.
- `rank_windows` is `score_windows` plus a sort, O(W log W).
- `rank_windows_rule_first` calls `rule_window_scorer`, which runs `rule_based_candidates(ir)` fresh each time (not through the `_rules` cache) and then counts, for each window, the candidates fully inside it: cost of the generator + O(W·C). If the window head is trained, it also runs `score_windows` and sorts by (−rule, −prob), O(W log W). If not, it sorts by rule count only and never calls the network.
- `rank_candidates` always runs `predict` (the Section 2 pair path, Θ(G²) or more). When `include_rules` is set and a net exists, it calls `rule_based_candidates` again, uncached. It then calls `rank_windows_rule_first`, which runs the generator once more through `rule_window_scorer`. Ranking windows through this entry point therefore still pays the pair-path cost, and runs the rule generator up to twice more.

## 4. Training

Default `GNNConfig`: `conv="sage"`, `hidden=128`, `layers=3`, `dropout=0.1`, `epochs=100`, `lr=1e-3`, `rule_features=True`.

**Parameters.** The encoder is O(F·d + L·d²), the pair head O(d²) (input 4d + 4) and the window head O(d²) (input 2d + 6), so O(F·d + L·d²) in total. At the defaults (conv=sage, d=128, L=3, F=35) the model has 174,978 parameters, counted with `sum(p.numel() for p in CancellationGNN(GNNConfig()).parameters())`.

**`fit` (pair head).** One optimizer step per training circuit per epoch (not minibatched), so epochs × N_train steps. Each step costs:

- `rule_based_candidates(ir)` for the unknown set. It is called directly, not through the `_rules` cache, so it is recomputed every epoch: Θ(G²) to O(G³).
- `sample_training_pairs`, which calls `overlapping_gate_pairs` (Θ(G²), also recomputed every epoch), filters and shuffles the negative pool in O(P), and keeps at most `negative_ratio` × |pos| negatives (or 16 when there are no positives). The sampled pair count is S ≤ (1 + negative_ratio)·|pos| when |pos| > 0, and |pos| ≤ C.
- `build_pyg_data` (Section 1) and `_pair_tensors` over the S sampled pairs, O(S).
- `encode` over the whole graph, O(G·(F·d + L·d²)).
- `score_pairs` over S pairs, O(S·d²).
- Backward pass, of the same order as forward.

Because S is small, the pair head is cheap in training. The Θ(G²) pair pool and the rule generator dominate per-step cost on large circuits. Loss is BCE with `pos_weight = n_neg / n_pos` per circuit (1.0 when either class is empty).

**Threshold calibration.** `_calibrate_threshold` runs the full `_pair_logits` (Θ(G²) enumeration plus all P pairs through the head) on every validation circuit, then sweeps 19 thresholds over all M scored pairs, O(19·M).

**`fit_windows` (window head).** One step per training circuit per epoch. Each step is one `_window_forward` with gradients (Section 3 forward cost) plus backward. The rule candidates come from `_rules`, so after the first epoch they are cached. The loss is plain BCE, using only windows with a label. After training there is one no-grad pass per validation circuit, then `fit_temperature`.

**Temperature fit.** `fit_temperature` returns 1.0 if there are fewer than 2 labels or only one class. Otherwise it evaluates a grid of 201 temperatures (200 log-spaced from 0.05 to 20, plus 1.0). Each `_nll` call is one linear pass over the M validation windows (a sigmoid and two logs per window), so the fit is O(201·M), in pure Python.

If `self._net` already exists, `fit_windows` continues training that encoder with a fresh Adam optimizer. Otherwise it creates a new net.

**Memory.** Training is one circuit at a time. Activation memory is O(L·G·d) plus O(S·d) for the pair head (`fit`) or O(W·d) for the window head (`fit_windows`).

## 5. Verifier Routing

`EquivalenceChecker.check()` first rejects a qubit-count mismatch. With `backend` set to `auto` or `phase_poly`, if both circuits are phase-polynomial circuits it calls `check_phase_poly` at any width. Otherwise, for n ≤ 8 it builds dense unitaries and checks with SMT or numeric (under `auto`, SMT when n > 5 or the combined gate count exceeds 50, numeric otherwise). Beyond 8 qubits with non-phase-poly gates it returns `method="unverified"` with `equivalent=False`.

The pipeline calls `check()` through `_check`, after `_compact` of the pair. GNN extras beyond the rules-proven rewrites are verified with `cheap_checker` (`backend="numeric"`) when the lowest extra score is at least `confident_threshold`, and with the `auto` checker otherwise. Rules-proven rewrites are applied unverified. `revert_rewrite` keeps a rewrite only if `.equivalent` is true, so `unverified` results are rejected. `verify_labels` (label generation) maps `error` and `unverified` to `None`, so those windows are excluded from labels.

Two behaviours worth knowing:

1. `backend="numeric"` skips the phase-poly shortcut, so `cheap_checker` never proves anything past 8 qubits. A confident extra on a compacted circuit wider than 8 qubits is rejected as `unverified`, even if the `auto` checker could have proved it. This loses gain but never accepts a wrong rewrite.
2. Without Z3, `check_phase_poly` returns non-equivalent whenever the term maps differ. That wrongly rejects circuits that are equal up to phase (for example `Z q0; Z q1` versus `CX; Rz(π) q1; CX`). This is conservative too.

This section does not state how much `_compact` narrows a block, because that function has not been inspected.
