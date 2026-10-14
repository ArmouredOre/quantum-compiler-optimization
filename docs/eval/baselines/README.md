# Baseline compilers on the benchmark suite (final)

These numbers are independent of qco's modules, so they are final. Regenerate with
`make eval` (they are part of `results/`); tool versions used:
qiskit 2.5.2, pytket 2.18.5, pyzx 0.10.7, bqskit 1.2.1.

Calibration `default`. **CX-cost** = multi-qubit gates in CX-equivalents (ccx = 6,
swap = 3, ...). Use it to compare tools: Qiskit keeps `ccx`/`cp` in its output
basis while TKET, PyZX and BQSKit decompose them, so raw gate counts favor Qiskit.
BQSKit is capped at 8 qubits (instantiation-based synthesis is exponential in
width); PyZX fails on one circuit (`qft_n16`). Skipped circuits are listed, never counted as 0%.

### Suite means

Means are over the circuits each tool handled (`circuits` column), so BQSKit's row covers only the 11 circuits with ≤ 8 qubits.

| optimizer | circuits | gate red % | depth red % | CX-cost red % | exec (ns) | fidelity | opt (s) | solver (s) | skipped |
|---|---|---|---|---|---|---|---|---|---|
| qiskit-O0 | 26 | 0.0 | 0.0 | 0.0 | 7915 | 0.5750 | 0.00 | — | 0 |
| qiskit-O1 | 26 | 14.6 | 7.7 | 0.7 | 7844 | 0.5835 | 0.00 | — | 0 |
| qiskit-O2 | 26 | 20.0 | 16.6 | 11.7 | 7047 | 0.6240 | 0.01 | — | 0 |
| qiskit-O3 | 26 | 19.8 | 16.8 | 11.3 | 7041 | 0.6241 | 0.01 | — | 0 |
| tket | 26 | -126.2 | -198.1 | -1.6 | 12894 | 0.5190 | 1.08 | — | 0 |
| pyzx | 25 | -107.0 | -223.5 | -72.3 | 23013 | 0.4466 | 0.08 | — | 1 |
| bqskit | 11 | -725.2 | -623.5 | -1.7 | 12602 | 0.5576 | 8.80 | — | 15 |

### Gates / CX-cost after optimization, per circuit

| circuit | input | qiskit-O0 | qiskit-O1 | qiskit-O2 | qiskit-O3 | tket | pyzx | bqskit |
|---|---|---|---|---|---|---|---|---|
| ghz_n10 | 10 / 9 | 10 / 9 | 10 / 9 | 10 / 9 | 10 / 9 | 12 / 9 | 10 / 9 | n/a |
| ghz_n20 | 20 / 19 | 20 / 19 | 20 / 19 | 20 / 19 | 20 / 19 | 22 / 19 | 20 / 19 | n/a |
| ghz_n40 | 40 / 39 | 40 / 39 | 40 / 39 | 40 / 39 | 40 / 39 | 42 / 39 | 40 / 39 | n/a |
| ghz_n5 | 5 / 4 | 5 / 4 | 5 / 4 | 5 / 4 | 5 / 4 | 7 / 4 | 5 / 4 | 9 / 4 |
| grover_w3_i1 | 21 / 12 | 21 / 12 | 10 / 12 | 10 / 12 | 10 / 12 | 49 / 12 | 36 / 13 | 103 / 8 |
| grover_w3_i2 | 39 / 24 | 39 / 24 | 18 / 24 | 18 / 24 | 18 / 24 | 102 / 20 | 70 / 27 | 114 / 9 |
| grover_w5_i1 | 39 / 60 | 39 / 60 | 24 / 60 | 24 / 60 | 24 / 60 | 214 / 60 | 140 / 65 | 662 / 57 |
| grover_w5_i2 | 73 / 120 | 73 / 120 | 44 / 120 | 44 / 120 | 44 / 120 | 402 / 120 | 270 / 132 | 1223 / 108 |
| grover_w7_i1 | 57 / 108 | 57 / 108 | 38 / 108 | 38 / 108 | 38 / 108 | 388 / 108 | 257 / 137 | n/a |
| grover_w7_i2 | 107 / 216 | 107 / 216 | 70 / 216 | 70 / 216 | 70 / 216 | 707 / 216 | 434 / 209 | n/a |
| qaoa_n10_p1 | 65 / 30 | 65 / 30 | 65 / 30 | 66 / 18 | 71 / 18 | 85 / 30 | 151 / 96 | n/a |
| qaoa_n10_p2 | 120 / 60 | 120 / 60 | 120 / 60 | 88 / 54 | 88 / 54 | 140 / 60 | 265 / 169 | n/a |
| qaoa_n16_p1 | 104 / 48 | 104 / 48 | 104 / 48 | 118 / 34 | 118 / 36 | 136 / 48 | 274 / 186 | n/a |
| qaoa_n16_p2 | 192 / 96 | 192 / 96 | 192 / 96 | 173 / 81 | 163 / 82 | 224 / 96 | 521 / 363 | n/a |
| qaoa_n6_p1 | 39 / 18 | 39 / 18 | 39 / 18 | 36 / 11 | 38 / 11 | 51 / 18 | 70 / 39 | 63 / 18 |
| qaoa_n6_p2 | 72 / 36 | 72 / 36 | 72 / 36 | 57 / 30 | 57 / 30 | 84 / 36 | 133 / 73 | 96 / 36 |
| qft_n12 | 84 / 84 | 84 / 84 | 84 / 84 | 78 / 66 | 78 / 66 | 302 / 132 | 331 / 222 | n/a |
| qft_n16 | 144 / 144 | 144 / 144 | 144 / 144 | 136 / 120 | 136 / 120 | 530 / 240 | n/a | n/a |
| qft_n5 | 17 / 16 | 17 / 16 | 17 / 16 | 15 / 10 | 15 / 10 | 57 / 20 | 64 / 37 | 322 / 27 |
| qft_n8 | 40 / 40 | 40 / 40 | 40 / 40 | 36 / 28 | 36 / 28 | 138 / 56 | 156 / 102 | 799 / 69 |
| rand_cliffordt_n10 | 336 / 113 | 336 / 113 | 266 / 111 | 225 / 108 | 223 / 107 | 255 / 100 | 277 / 198 | n/a |
| rand_cliffordt_n20 | 662 / 228 | 662 / 228 | 562 / 228 | 490 / 221 | 491 / 222 | 572 / 214 | 812 / 650 | n/a |
| rand_cliffordt_n5 | 173 / 53 | 173 / 53 | 128 / 49 | 120 / 45 | 117 / 46 | 120 / 40 | 91 / 48 | 477 / 47 |
| rand_cnotpauli_n10 | 308 / 122 | 308 / 122 | 238 / 116 | 216 / 110 | 216 / 110 | 89 / 83 | 71 / 60 | n/a |
| rand_cnotpauli_n20 | 625 / 248 | 625 / 248 | 509 / 246 | 468 / 242 | 468 / 242 | 249 / 227 | 208 / 188 | n/a |
| rand_cnotpauli_n5 | 143 / 49 | 143 / 49 | 109 / 47 | 89 / 43 | 89 / 44 | 27 / 25 | 19 / 15 | 399 / 49 |

### Skipped circuits

- **pyzx**: qft_n16
- **bqskit**: ghz_n10, ghz_n20, ghz_n40, grover_w7_i1, grover_w7_i2, qaoa_n10_p1, qaoa_n10_p2, qaoa_n16_p1, qaoa_n16_p2, qft_n12, qft_n16, rand_cliffordt_n10, rand_cliffordt_n20, rand_cnotpauli_n10, rand_cnotpauli_n20

Per-tool five-metric tables: `qiskit-O0.md` ... `bqskit.md` (and `.csv`) in this folder.
