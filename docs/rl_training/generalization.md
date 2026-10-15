# Zero-shot generalization: 5 -> 10 -> 20 qubits (issue #26)

Trained on generated 5-qubit circuits (families clifford_t / cnot_pauli / qaoa / qft, disjoint from `benchmarks/`), 100000 timesteps, `commutation=full`, reward = environment default.

Seeds: [0, 1, 2]. Values are mean % reduction vs the raw benchmark circuit (depth, then gate count); depth is mean ± std over seeds.

## Depth reduction % by qubit bucket

| Method | 5-6 qubits (n=6) | 8-12 qubits (n=7) | 16-20 qubits (n=6) |
|---|---|---|---|
| greedy `schedule()` stub | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| CR only | 4.6 ± 0.0 | 2.4 ± 0.0 | 3.1 ± 0.0 |
| greedy-depth + CR | 11.5 ± 0.0 | 8.0 ± 0.4 | 7.4 ± 0.5 |
| **PPO + CR** | 12.4 ± 0.7 | 9.2 ± 0.6 | 7.9 ± 0.2 |
| PPO + CR, scaled budget | 12.4 ± 0.7 | 9.2 ± 0.6 | 7.9 ± 0.2 |
| PPO, no CR (trained & run) | 7.9 ± 1.0 | 7.3 ± 0.8 | 9.2 ± 2.3 |
| CR-trained PPO, CR off at test | 7.2 ± 0.7 | 7.0 ± 0.6 | 4.9 ± 0.2 |

## Gate-count reduction %

| Method | 5-6 | 8-12 | 16-20 |
|---|---|---|---|
| greedy `schedule()` stub | 0.0 | 0.0 | 0.0 |
| CR only | 8.7 | 7.2 | 6.5 |
| greedy-depth + CR | 8.7 | 7.2 | 6.5 |
| **PPO + CR** | 8.7 | 7.2 | 6.5 |
| PPO + CR, scaled budget | 8.7 | 7.2 | 6.5 |
| PPO, no CR (trained & run) | 0.0 | 0.0 | 0.0 |
| CR-trained PPO, CR off at test | 0.0 | 0.0 | 0.0 |

## Per circuit (depth reduction %, mean over seeds)

| Circuit | qubits | gates | depth | greedy `schedule()` stub | CR only | greedy-depth + CR | **PPO + CR** | PPO + CR, scaled budget | PPO, no CR (trained & run) | CR-trained PPO, CR off at test |
|---|---|---|---|---|---|---|---|---|---|---|
| ghz_n10 | 10 | 10 | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ghz_n20 | 20 | 20 | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ghz_n5 | 5 | 5 | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qaoa_n10_p1 | 10 | 65 | 26 | 0.0 | 0.0 | 12.8 | 20.5 | 20.5 | 21.8 | 20.5 |
| qaoa_n10_p2 | 10 | 120 | 36 | 0.0 | 0.0 | 17.6 | 17.6 | 17.6 | 17.6 | 17.6 |
| qaoa_n16_p1 | 16 | 104 | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 16.7 | 0.0 |
| qaoa_n16_p2 | 16 | 192 | 48 | 0.0 | 0.0 | 18.8 | 18.8 | 18.8 | 23.6 | 18.8 |
| qaoa_n6_p1 | 6 | 39 | 20 | 0.0 | 0.0 | 20.0 | 23.3 | 23.3 | 23.3 | 23.3 |
| qaoa_n6_p2 | 6 | 72 | 36 | 0.0 | 0.0 | 11.1 | 13.0 | 13.0 | 13.0 | 13.0 |
| qft_n12 | 12 | 84 | 24 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qft_n16 | 16 | 144 | 32 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qft_n5 | 5 | 17 | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qft_n8 | 8 | 40 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rand_cliffordt_n10 | 10 | 336 | 66 | 0.0 | 13.6 | 15.2 | 15.7 | 15.7 | 4.5 | 4.5 |
| rand_cliffordt_n20 | 20 | 662 | 70 | 0.0 | 8.6 | 12.9 | 14.8 | 14.8 | 7.1 | 5.7 |
| rand_cliffordt_n5 | 5 | 173 | 60 | 0.0 | 8.3 | 15.0 | 15.6 | 15.6 | 5.6 | 3.3 |
| rand_cnotpauli_n10 | 10 | 308 | 67 | 0.0 | 3.0 | 10.4 | 10.4 | 10.4 | 7.5 | 6.5 |
| rand_cnotpauli_n20 | 20 | 625 | 71 | 0.0 | 9.9 | 12.7 | 13.6 | 13.6 | 8.0 | 5.2 |
| rand_cnotpauli_n5 | 5 | 143 | 57 | 0.0 | 19.3 | 22.8 | 22.8 | 22.8 | 5.8 | 3.5 |

Equivalence (numeric, <= 8 qubits): 147 checks, 0 failures. Circuits above 8 qubits are not equivalence-checked here (numeric backend cap; the SMT path is the follow-up).
