# IR, Front End & Graph Layer — Notes for the Report (Stages 1–3)

**Author:** Swaraj Rane (@ArmouredOre) · implemented in #4 and #9 (PR #49)
**For:** Prisha's graph-layer report section (#48) and the final report (#44)

These are working notes, not a finished section: facts, design decisions and
numbers that the graph-layer write-up can draw on. Everything below describes
`main` as of PR #49; `qco.ir` and `qco.graphs` have not changed since.

---

## 1. Stage 1 — Intermediate representation (`qco.ir`)

**Data model.** A circuit is an `IntermediateRepresentation`: `num_qubits` plus
an ordered list of immutable `Gate(name, qubits, params, tag)`. Every later
stage reads and rewrites this one structure. Arity is checked when a gate is
built, and qubit indices are bounds-checked on `add`.

**Gate set (29 gates).** The supported set is exactly the set Module C's
checker can build a matrix for, so **every IR circuit is verifiable** — this is
what lets novelty N2 promise "verified or explicitly unverified", never
"silently unchecked".

| Arity | Gates |
|---|---|
| 1 | `id x y z h s sdg t tdg rx ry rz p` |
| 2 | `cx cy cz ch cs csdg swap cp crx cry crz rzz rxx ryy` |
| 3 | `ccx cswap` |

Each gate knows its inverse (`Gate.inverse()`): self-inverse gates return
themselves, rotations negate their angle, `s↔sdg`, `t↔tdg`, `cs↔csdg`.
Module B's cancellation rules and the SMT verifier both rely on this.

**Front ends.**

| Entry point | What it does |
|---|---|
| `from_qasm2(text)` | OpenQASM 2.0, `qelib1.inc` subset. All `qreg`s are flattened into one index space in declaration order. Legacy spellings are mapped (`u1→p`, `cu1→cp`, `cnot→cx`, `i→id`, `toffoli→ccx`, `fredkin→cswap`). Custom `gate` definitions are skipped. |
| `from_qiskit(circuit, bindings)` | Walks `QuantumCircuit.data` directly; registers flattened via `find_bit`; `Parameter`s bound from `bindings`; any still-unbound parameter raises. |
| `from_cirq(circuit, resolver)` | Resolves sympy parameters, then goes through `cirq.qasm` → `from_qasm2`. Cirq packs gates into moments, so independent gates may come back reordered (still equivalent). |
| `parse(source)` | Dispatches on type: IR, QASM string, file path, Qiskit or Cirq object. |

Non-unitary instructions (measure, reset, barrier, delay) are dropped — the
optimizer works on the unitary part. A gate outside the set raises
`ValueError` naming the gate; the fix is to transpile to a supported basis first.

**Validation.** Every one of the 26 benchmark circuits round-trips
QASM → IR → QASM unchanged, and loads identically through Qiskit
(`tests/test_ir_and_graphs.py`).

## 2. Stage 2 — Graphs (`qco.graphs.dag`, `qco.graphs.hypergraph`)

**Gate-dependency DAG.** Node = gate index; edge `i → j` when `j` is the next
gate on a qubit last touched by `i` (last-writer per qubit). Provides roots,
topological order (Kahn's algorithm with a deque, O(V+E)) and critical-path
length, which equals `IntermediateRepresentation.depth()`.

**Commutation table — `commutes(a, b)`.** Each gate lists, per qubit, the Pauli
it commutes with: `cx` commutes with Z on the control and X on the target, `rzz`
with Z on both, `ccx` with Z, Z, X, and so on (`None` = no simple Pauli). Two
gates commute if, on every qubit they share, both commute with the same Pauli:
each is then block-diagonal in that Pauli's eigenbasis there, and the blocks act
on disjoint remaining qubits. The rule is **sufficient, not necessary** — a
`False` means "not proven", never "proven not to commute".

*Soundness check:* for every ordered pair of the 29 gates and every
qubit-overlap layout, whenever `commutes(a, b)` is `True` the test checks
`a·b == b·a` **exactly**, on dense unitaries. Exact comparison matters: the
verifier compares up to global phase, and anticommuting Paulis
(X·Z = −Z·X) would wrongly pass a phase-insensitive check.

**Qubit-interaction hypergraph.** Vertices = qubits; one hyperedge per
multi-qubit gate. Hyperedges are weighted by **gate cost in CX-equivalents**
from standard decompositions (`swap` 3, `rxx/ryy/rzz` 2, `ccx` 6, `cswap` 8,
other 2-qubit gates 1), and identical qubit sets are merged with summed weight.
An optional `window` groups the multi-qubit gates inside each window of that
many gates by shared qubits (connected components) and emits one hyperedge per
group — only gates that actually interact are merged. `to_kahypar_hmetis()`
serializes to the hMETIS format KaHyPar reads.

The same CX-cost weighting is reused by Stage 6 as the basis-independent
`cx_cost` metric (`qco.evaluation.baselines`).

## 3. Stage 3 — Partitioning (`qco.graphs.partitioning`)

**Contract (all backends, tested on every benchmark).** Every gate lands in
exactly one `Block`, and concatenating the blocks in list order preserves each
qubit's gate order — so the stitched circuit equals the input and per-block
rewrites compose. This is what makes novelty **N3** (block-level crossover)
equivalent by construction, and what lets per-block verification stand in for
whole-circuit verification on wide circuits. `Block.boundary_qubits` = the
qubits a block shares with other blocks (its cut wires).

**Backends.**

| Backend | Method | Blocks |
|---|---|---|
| `auto` / `greedy` (default, no dependencies) | Sweep gates in order; merge the open blocks a gate touches while the result stays within `max_block_qubits` (8) and `max_block_gates` (64); otherwise emit them and open a new block | ≤ 8 qubits — verifiable and re-synthesizable |
| `metis` / `kahypar` (opt-in) | Cut the qubit hypergraph into ⌈n / 8⌉ balanced parts (METIS via `pymetis` on the clique expansion; KaHyPar via the hMETIS file + a preset `.ini` in `$KAHYPAR_INI`), then the same sweep without cross-part merges | ≤ 8 qubits |
| `fallback` | Connected components of the DAG, then size-capped chunks | Unbounded width |

**Why the width cap.** Dense verification is limited to 8 qubits, and bounded
exact re-synthesis works on ≤ 4-qubit windows (carved from blocks with the same
sweep). A block wider than 8 qubits can be neither verified densely nor
re-synthesized, so a lower wire cut bought with wide blocks is of no use to
Modules C and D.

**Results** (`python benchmarks/bench_partition.py 16`; ≥ 16-qubit benchmarks,
64-gate blocks; *cut* = DAG edges crossing blocks, *sd* = block-size standard
deviation, *width* = widest block):

| circuit | n | gates | fallback cut / blocks / width | greedy cut / blocks / width | METIS cut / blocks / width |
|---|---|---|---|---|---|
| ghz_n20 | 20 | 20 | 0 / 1 / 20 | 2 / 3 / 8 | 2 / 3 / 8 |
| ghz_n40 | 40 | 40 | 0 / 1 / 40 | 5 / 6 / 8 | 5 / 6 / 8 |
| qaoa_n16_p1 | 16 | 104 | 16 / 2 / 16 | 25 / 17 / 8 | 23 / 16 / 7 |
| qaoa_n16_p2 | 16 | 192 | 32 / 3 / 16 | 42 / 15 / 8 | 44 / 22 / 8 |
| qft_n16 | 16 | 144 | 28 / 3 / 16 | 119 / 27 / 8 | 148 / 37 / 8 |
| rand_cliffordt_n20 | 20 | 662 | 200 / 11 / 20 | 255 / 60 / 8 | 393 / 195 / 6 |
| rand_cnotpauli_n20 | 20 | 625 | 180 / 10 / 20 | 272 / 54 / 8 | 433 / 212 / 8 |

Reading the table:

- The fallback has the lowest cut only because its blocks span the whole
  circuit. No width-capped partitioner can match that, which is why #9's
  acceptance criterion was amended (2026-10-13) from "beat the fallback" to
  "bounded width, cut no worse than the METIS-guided variant".
- Greedy beats METIS-guided on every ≥ 20-qubit circuit (ties on GHZ). The
  random Clifford+T / CNOT+Pauli circuits have no qubit locality for a
  hypergraph cut to exploit, so solver guidance adds fragmentation. METIS wins
  narrowly only on `qaoa_n16_p1` (23 vs 25), so the solvers stay opt-in.

**Known cost of the cap (input for #46).** With the trained RL agent, running
Module A per block on wide circuits gains less than running it on the whole
circuit (`rand_cliffordt_n20`: 579 vs 515 gates), because 20 qubits split into
about 60 blocks. Raising `max_block_gates` doesn't help there — width, not gate
count, fragments the circuit — and raising the width cap gives up dense
verification. This is the main tuning question for #46.

## 4. Where to find things

| What | Where |
|---|---|
| IR, gate set, aliases | `src/qco/ir/intermediate_representation.py` |
| Parsers | `src/qco/ir/parser.py` |
| DAG + commutation table | `src/qco/graphs/dag.py` |
| Hypergraph + CX-cost weights | `src/qco/graphs/hypergraph.py` |
| Partitioner + `cut_size` | `src/qco/graphs/partitioning.py` |
| Partitioner bench | `benchmarks/bench_partition.py` |
| Tests (round-trips, soundness, contract) | `tests/test_ir_and_graphs.py` |
