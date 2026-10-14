# qco evaluation

Calibration `default`, solver budget 0.25s/circuit. CX-cost = multi-qubit gates in CX-equivalents (basis-independent; tools emit different gate sets).

### N1 window selection (circuits <= 5 qubits)

| optimizer | circuits | gate red % | depth red % | CX-cost red % | exec (ns) | fidelity | opt (s) | solver (s) | skipped |
|---|---|---|---|---|---|---|---|---|---|
| gnn | 6 | 13.2 | 14.1 | 2.6 | 4421 | 0.7970 | 0.40 | 1.5 | 0 |
| rule | 6 | 13.2 | 14.1 | 2.6 | 4421 | 0.7970 | 0.37 | 1.5 | 0 |
| random | 6 | 14.0 | 14.9 | 2.6 | 4415 | 0.7973 | 0.37 | 1.5 | 0 |
| exhaustive | 6 | 16.1 | 15.8 | 3.3 | 4409 | 0.8009 | 144.45 | 866.1 | 0 |
