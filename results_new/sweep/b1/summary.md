# qco evaluation

Calibration `default`, solver budget 1.0s/circuit. CX-cost = multi-qubit gates in CX-equivalents (basis-independent; tools emit different gate sets).

### N1 window selection (circuits <= 5 qubits)

| optimizer | circuits | gate red % | depth red % | CX-cost red % | exec (ns) | fidelity | opt (s) | solver (s) | skipped |
|---|---|---|---|---|---|---|---|---|---|
| gnn | 6 | 13.6 | 14.1 | 2.6 | 4421 | 0.7974 | 1.17 | 6.0 | 0 |
| rule | 6 | 13.2 | 14.1 | 2.6 | 4421 | 0.7970 | 1.13 | 6.0 | 0 |
| random | 6 | 14.0 | 14.9 | 2.6 | 4415 | 0.7973 | 1.14 | 6.0 | 0 |
| exhaustive | 6 | 16.2 | 15.8 | 3.3 | 4409 | 0.8010 | 139.89 | 838.8 | 0 |
