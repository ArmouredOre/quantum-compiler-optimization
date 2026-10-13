# Module C: Verification Scalability & Empirical Benchmarks

### 1. Verification Runtime vs Qubit Count

| Qubits ($n$) | Gate Count | Numeric (Dense) [s] | Z3-Dense SMT [s] | Phase-Poly SMT (N2) [s] | Scaling Status |
|:------------:|:----------:|:-------------------:|:----------------:|:-----------------------:|:---------------|
|  2 |   8 | 0.0101 | 0.1300 | 0.00014 | Verified (All) |
|  3 |  14 | 0.0023 | 0.0314 | 0.00019 | Verified (All) |
|  4 |  20 | 0.0045 | 0.0406 | 0.00036 | Verified (All) |
|  5 |  26 | 0.0066 | *Intractable (>4q)* | 0.00013 | Verified (All) |
|  6 |  32 | 0.0098 | *Intractable (>4q)* | 0.00026 | Verified (All) |
|  7 |  38 | 0.0310 | *Intractable (>4q)* | 0.00022 | Verified (All) |
|  8 |  44 | 1.1650 | *Intractable (>4q)* | 0.00049 | Verified (All) |
| 10 |  56 | *Capped (>8q)* | *Intractable (>4q)* | 0.00034 | Verified (Phase-Poly) |
| 12 |  68 | *Capped (>8q)* | *Intractable (>4q)* | 0.00031 | Verified (Phase-Poly) |
| 16 |  92 | *Capped (>8q)* | *Intractable (>4q)* | 0.00077 | Verified (Phase-Poly) |
| 20 | 116 | *Capped (>8q)* | *Intractable (>4q)* | 0.00041 | Verified (Phase-Poly) |
| 24 | 140 | *Capped (>8q)* | *Intractable (>4q)* | 0.00060 | Verified (Phase-Poly) |
| 32 | 188 | *Capped (>8q)* | *Intractable (>4q)* | 0.00063 | Verified (Phase-Poly) |
| 40 | 236 | *Capped (>8q)* | *Intractable (>4q)* | 0.00110 | Verified (Phase-Poly) |
| 48 | 284 | *Capped (>8q)* | *Intractable (>4q)* | 0.00164 | Verified (Phase-Poly) |
| 64 | 380 | *Capped (>8q)* | *Intractable (>4q)* | 0.00191 | Verified (Phase-Poly) |

### 2. Bounded Re-Synthesis Gains on Real Benchmark Windows

| Benchmark Circuit | Qubits | Original Gates | Evaluated Windows | Reduced Windows | Gates Saved | Reduction (%) | Cache Hits | Solve Time [s] |
|:------------------|:------:|:--------------:|:-----------------:|:---------------:|:-----------:|:-------------:|:----------:|:---------------:|
| `ghz_n5.qasm` | 5 | 5 | 0 | 0 | 0 | 0.0% | 0 | 0.0000 |
| `ghz_n10.qasm` | 10 | 10 | 0 | 0 | 0 | 0.0% | 0 | 0.0000 |
| `rand_cnotpauli_n5.qasm` | 5 | 34 | 8 | 1 | 1 | 2.9% | 0 | 1.6522 |
| `rand_cliffordt_n5.qasm` | 5 | 34 | 8 | 0 | 0 | 0.0% | 0 | 1.8883 |
| `qaoa_n6_p1.qasm` | 6 | 32 | 8 | 0 | 0 | 0.0% | 1 | 1.5023 |
| `qft_n5.qasm` | 5 | 17 | 2 | 0 | 0 | 0.0% | 1 | 0.6018 |
