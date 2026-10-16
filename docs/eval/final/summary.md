# qco evaluation

Calibration `default`, solver budget 2.0s/circuit, RL policy `models/rl_ppo_5q.json`, GNN checkpoint `checkpoints/module_b.pt`. CX-cost = multi-qubit gates in CX-equivalents (basis-independent; tools emit different gate sets).

### qco vs baselines

| optimizer | circuits | correct | gate red % | depth red % | CX-cost red % | exec (ns) | fidelity | opt (s) | solver (s) | skipped |
|---|---|---|---|---|---|---|---|---|---|---|
| qco | 26 | 25/25 (1 n/a) | 7.1 | 7.9 | 0.9 | 7826 | 0.5812 | 2.99 | 52.0 | 0 |
| qiskit-O0 | 26 | 25/25 (1 n/a) | 0.0 | 0.0 | 0.0 | 7915 | 0.5750 | 0.00 | — | 0 |
| qiskit-O1 | 26 | 25/25 (1 n/a) | 14.6 | 7.7 | 0.7 | 7844 | 0.5835 | 0.00 | — | 0 |
| qiskit-O2 | 26 | 25/25 (1 n/a) | 18.9 | 16.0 | 7.4 | 7088 | 0.6204 | 0.01 | — | 0 |
| qiskit-O3 | 26 | 25/25 (1 n/a) | 18.5 | 15.9 | 7.0 | 7088 | 0.6202 | 0.01 | — | 0 |
| tket | 26 | 25/25 (1 n/a) | -131.1 | -204.5 | -8.4 | 13787 | 0.5034 | 1.05 | — | 0 |
| pyzx | 25 | 24/24 (1 n/a) | -107.0 | -223.5 | -72.3 | 23013 | 0.4466 | 0.08 | — | 1 |
| bqskit | 11 | 11/11 | -725.2 | -623.5 | -1.7 | 12602 | 0.5576 | 11.08 | — | 15 |

*correct* = outputs checked equivalent to the input (Module C's checker, else a Qiskit statevector check on random states up to 22 qubits; n/a = too wide to decide).

| circuit | input | qco | qiskit-O0 | qiskit-O1 | qiskit-O2 | qiskit-O3 | tket | pyzx | bqskit |
|---|---|---|---|---|---|---|---|---|---|
| ghz_n10 | 10 / 9 | 10 / 9 | 10 / 9 | 10 / 9 | 10 / 9 | 10 / 9 | 12 / 9 | 10 / 9 | n/a |
| ghz_n20 | 20 / 19 | 20 / 19 | 20 / 19 | 20 / 19 | 20 / 19 | 20 / 19 | 22 / 19 | 20 / 19 | n/a |
| ghz_n40 | 40 / 39 | 40 / 39 | 40 / 39 | 40 / 39 | 40 / 39 | 40 / 39 | 42 / 39 | 40 / 39 | n/a |
| ghz_n5 | 5 / 4 | 5 / 4 | 5 / 4 | 5 / 4 | 5 / 4 | 5 / 4 | 7 / 4 | 5 / 4 | 9 / 4 |
| grover_w3_i1 | 21 / 12 | 17 / 12 | 21 / 12 | 10 / 12 | 10 / 12 | 10 / 12 | 49 / 12 | 36 / 13 | 103 / 8 |
| grover_w3_i2 | 39 / 24 | 29 / 24 | 39 / 24 | 18 / 24 | 18 / 24 | 18 / 24 | 102 / 20 | 70 / 27 | 114 / 9 |
| grover_w5_i1 | 39 / 60 | 35 / 60 | 39 / 60 | 24 / 60 | 24 / 60 | 24 / 60 | 214 / 60 | 140 / 65 | 662 / 57 |
| grover_w5_i2 | 73 / 120 | 63 / 120 | 73 / 120 | 44 / 120 | 44 / 120 | 44 / 120 | 402 / 120 | 270 / 132 | 1223 / 108 |
| grover_w7_i1 | 57 / 108 | 53 / 108 | 57 / 108 | 38 / 108 | 38 / 108 | 38 / 108 | 388 / 108 | 257 / 137 | n/a |
| grover_w7_i2 | 107 / 216 | 99 / 216 | 107 / 216 | 70 / 216 | 70 / 216 | 70 / 216 | 707 / 216 | 434 / 209 | n/a |
| qaoa_n10_p1 | 65 / 30 | 65 / 30 | 65 / 30 | 65 / 30 | 66 / 18 | 70 / 18 | 85 / 30 | 151 / 96 | n/a |
| qaoa_n10_p2 | 120 / 60 | 120 / 60 | 120 / 60 | 120 / 60 | 88 / 54 | 88 / 54 | 140 / 60 | 265 / 169 | n/a |
| qaoa_n16_p1 | 104 / 48 | 104 / 48 | 104 / 48 | 104 / 48 | 121 / 34 | 120 / 35 | 136 / 48 | 274 / 186 | n/a |
| qaoa_n16_p2 | 192 / 96 | 192 / 96 | 192 / 96 | 192 / 96 | 168 / 85 | 166 / 86 | 224 / 96 | 521 / 363 | n/a |
| qaoa_n6_p1 | 39 / 18 | 39 / 18 | 39 / 18 | 39 / 18 | 36 / 12 | 36 / 13 | 51 / 18 | 70 / 39 | 63 / 18 |
| qaoa_n6_p2 | 72 / 36 | 72 / 36 | 72 / 36 | 72 / 36 | 55 / 30 | 57 / 30 | 84 / 36 | 133 / 73 | 96 / 36 |
| qft_n12 | 84 / 84 | 84 / 84 | 84 / 84 | 84 / 84 | 84 / 84 | 84 / 84 | 320 / 150 | 331 / 222 | n/a |
| qft_n16 | 144 / 144 | 144 / 144 | 144 / 144 | 144 / 144 | 144 / 144 | 144 / 144 | 554 / 264 | n/a | n/a |
| qft_n5 | 17 / 16 | 17 / 16 | 17 / 16 | 17 / 16 | 17 / 16 | 17 / 16 | 63 / 26 | 64 / 37 | 322 / 27 |
| qft_n8 | 40 / 40 | 40 / 40 | 40 / 40 | 40 / 40 | 40 / 40 | 40 / 40 | 150 / 68 | 156 / 102 | 799 / 69 |
| rand_cliffordt_n10 | 336 / 113 | 285 / 109 | 336 / 113 | 266 / 111 | 225 / 106 | 225 / 106 | 267 / 112 | 277 / 198 | n/a |
| rand_cliffordt_n20 | 662 / 228 | 579 / 222 | 662 / 228 | 562 / 228 | 488 / 220 | 491 / 222 | 584 / 226 | 812 / 650 | n/a |
| rand_cliffordt_n5 | 173 / 53 | 132 / 49 | 173 / 53 | 128 / 49 | 115 / 44 | 119 / 44 | 129 / 49 | 91 / 48 | 477 / 47 |
| rand_cnotpauli_n10 | 308 / 122 | 266 / 120 | 308 / 122 | 238 / 116 | 216 / 110 | 216 / 110 | 110 / 104 | 71 / 60 | n/a |
| rand_cnotpauli_n20 | 625 / 248 | 559 / 248 | 625 / 248 | 509 / 246 | 468 / 242 | 468 / 242 | 276 / 254 | 208 / 188 | n/a |
| rand_cnotpauli_n5 | 143 / 49 | 107 / 45 | 143 / 49 | 109 / 47 | 89 / 43 | 89 / 43 | 33 / 31 | 19 / 15 | 399 / 49 |

### Pareto fronts (qco, per circuit)

Plots + CSV in [`pareto/`](pareto/).

| circuit | front size | gates (min) | depth (min) | est. fidelity (max) |
|---|---|---|---|---|
| ghz_n10 | 1 | 10 | 10 | 0.9126 |
| ghz_n20 | 1 | 20 | 20 | 0.8253 |
| ghz_n40 | 1 | 40 | 40 | 0.6751 |
| ghz_n5 | 1 | 5 | 5 | 0.9596 |
| grover_w3_i1 | 1 | 17 | 8 | 0.9269 |
| grover_w3_i2 | 1 | 29 | 14 | 0.8634 |
| grover_w5_i1 | 1 | 35 | 15 | 0.7192 |
| grover_w5_i2 | 1 | 63 | 29 | 0.5209 |
| grover_w7_i1 | 1 | 53 | 23 | 0.5581 |
| grover_w7_i2 | 1 | 99 | 45 | 0.3136 |
| qaoa_n10_p1 | 1 | 65 | 21 | 0.7142 |
| qaoa_n10_p2 | 1 | 120 | 32 | 0.5153 |
| qaoa_n16_p1 | 1 | 104 | 18 | 0.5837 |
| qaoa_n16_p2 | 1 | 192 | 44 | 0.3462 |
| qaoa_n6_p1 | 1 | 39 | 16 | 0.8172 |
| qaoa_n6_p2 | 2 | 72 | 32 | 0.6718 |
| qft_n12 | 1 | 84 | 24 | 0.4792 |
| qft_n16 | 1 | 144 | 32 | 0.2719 |
| qft_n5 | 1 | 17 | 10 | 0.8820 |
| qft_n8 | 1 | 40 | 16 | 0.7192 |
| rand_cliffordt_n10 | 9 | 285 | 61 | 0.2804 |
| rand_cliffordt_n20 | 1 | 579 | 64 | 0.0751 |
| rand_cliffordt_n5 | 4 | 132 | 51 | 0.5624 |
| rand_cnotpauli_n10 | 3 | 266 | 64 | 0.2587 |
| rand_cnotpauli_n20 | 1 | 559 | 66 | 0.0606 |
| rand_cnotpauli_n5 | 3 | 107 | 45 | 0.5979 |

### Ablations (full pipeline minus one module)

| optimizer | circuits | correct | gate red % | depth red % | CX-cost red % | exec (ns) | fidelity | opt (s) | solver (s) | skipped |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 26 | 25/25 (1 n/a) | 7.1 | 7.9 | 0.9 | 7826 | 0.5812 | 2.99 | 52.0 | 0 |
| minus A (RL) | 26 | 25/25 (1 n/a) | 6.6 | 4.1 | 0.9 | 7828 | 0.5807 | 2.96 | 52.0 | 0 |
| minus B (GNN) | 26 | 25/25 (1 n/a) | 7.1 | 7.8 | 0.9 | 7825 | 0.5812 | 2.97 | 52.0 | 0 |
| minus C (SMT) | 26 | **16/25 (1 n/a) ✗** | 10.8 | 13.0 | 7.3 | 6669 | 0.6087 | 0.91 | 0.0 | 0 |
| minus D (NSGA-II) | 26 | 25/25 (1 n/a) | 7.1 | 7.1 | 0.9 | 7930 | 0.5812 | 2.15 | 52.0 | 0 |

> ⚠ **minus C (SMT)**: 9 output(s) are NOT equivalent to their input, so its reductions are not valid results.

*correct* = outputs checked equivalent to the input (Module C's checker, else a Qiskit statevector check on random states up to 22 qubits; n/a = too wide to decide).

### N1 window selection (circuits <= 6 qubits)

| optimizer | circuits | correct | gate red % | depth red % | CX-cost red % | exec (ns) | fidelity | opt (s) | solver (s) | skipped |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 8 | 8/8 | 11.7 | 15.8 | 2.0 | 4630 | 0.7851 | 2.40 | 16.0 | 0 |
| rule | 8 | 8/8 | 11.3 | 15.8 | 2.0 | 4634 | 0.7848 | 2.41 | 16.0 | 0 |
| random | 8 | 8/8 | 11.9 | 16.2 | 2.5 | 4597 | 0.7867 | 2.39 | 16.0 | 0 |
| exhaustive | 8 | 8/8 | 13.0 | 17.1 | 2.5 | 4592 | 0.7876 | 133.96 | 1067.8 | 0 |

*correct* = outputs checked equivalent to the input (Module C's checker, else a Qiskit statevector check on random states up to 22 qubits; n/a = too wide to decide).

### N5: confidence-gated verification

| gating | circuits | all verified | unverified blocks | gates after | checks run | checks skipped | verification (s) |
|---|---|---|---|---|---|---|---|
| on | 26 | 26/26 | 0 | 3176 | 388 | 0 | 1.12 |
| off | 26 | 26/26 | 0 | 3176 | 388 | 0 | 1.32 |
