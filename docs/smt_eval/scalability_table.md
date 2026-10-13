# Module C: Verification Scalability & Empirical Benchmarks

### 1. Verification Runtime vs Qubit Count

| Qubits ($n$) | Gate Count | Numeric (Dense) [s] | Z3-Dense SMT [s] | Phase-Poly SMT (N2) [s] | Scaling Status |
|:------------:|:----------:|:-------------------:|:----------------:|:-----------------------:|:---------------|
|  2 |   8 | 0.0104 | 0.2155 | 0.00015 | Verified (All) |
|  3 |  14 | 0.0047 | 0.1563 | 0.00029 | Verified (All) |
|  4 |  20 | 0.0061 | 0.0865 | 0.00014 | Verified (All) |
|  5 |  26 | 0.0083 | *Intractable (>4q)* | 0.00015 | Verified (All) |
|  6 |  32 | 0.0168 | *Intractable (>4q)* | 0.00024 | Verified (All) |
|  7 |  38 | 0.0502 | *Intractable (>4q)* | 0.00023 | Verified (All) |
|  8 |  44 | 3.2169 | *Intractable (>4q)* | 0.00031 | Verified (All) |
| 10 |  56 | *Capped (>8q)* | *Intractable (>4q)* | 0.00030 | Verified (Phase-Poly) |
| 12 |  68 | *Capped (>8q)* | *Intractable (>4q)* | 0.00031 | Verified (Phase-Poly) |
| 16 |  92 | *Capped (>8q)* | *Intractable (>4q)* | 0.00042 | Verified (Phase-Poly) |
| 20 | 116 | *Capped (>8q)* | *Intractable (>4q)* | 0.00052 | Verified (Phase-Poly) |
| 24 | 140 | *Capped (>8q)* | *Intractable (>4q)* | 0.00059 | Verified (Phase-Poly) |
| 32 | 188 | *Capped (>8q)* | *Intractable (>4q)* | 0.00227 | Verified (Phase-Poly) |
| 40 | 236 | *Capped (>8q)* | *Intractable (>4q)* | 0.00733 | Verified (Phase-Poly) |
| 48 | 284 | *Capped (>8q)* | *Intractable (>4q)* | 0.00303 | Verified (Phase-Poly) |
| 64 | 380 | *Capped (>8q)* | *Intractable (>4q)* | 0.00219 | Verified (Phase-Poly) |

### 2. Bounded Re-Synthesis Gains on Real Benchmark Windows

| Benchmark Circuit | Qubits | Original Gates | Evaluated Windows | Reduced Windows | Gates Saved | Reduction (%) | Cache Hits | Solve Time [s] |
|:------------------|:------:|:--------------:|:-----------------:|:---------------:|:-----------:|:-------------:|:----------:|:---------------:|
| `ghz_n5.qasm` | 5 | 5 | 0 | 0 | 0 | 0.0% | 0 | 0.0000 |
| `ghz_n10.qasm` | 10 | 10 | 0 | 0 | 0 | 0.0% | 0 | 0.0000 |
| `rand_cnotpauli_n5.qasm` | 5 | 34 | 8 | 1 | 1 | 2.9% | 0 | 1.5645 |
| `rand_cliffordt_n5.qasm` | 5 | 34 | 8 | 0 | 0 | 0.0% | 0 | 1.9688 |
| `qaoa_n6_p1.qasm` | 6 | 32 | 8 | 0 | 0 | 0.0% | 1 | 1.5053 |
| `qft_n5.qasm` | 5 | 17 | 2 | 0 | 0 | 0.0% | 1 | 0.6010 |
