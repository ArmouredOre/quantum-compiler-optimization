# Benchmark suite

Selection and rationale: [`../docs/phase2_environment_setup.md`](../docs/phase2_environment_setup.md) §(c).

## Regenerate

```bash
python generate_circuits.py            # writes circuits/*.qasm
python generate_circuits.py --list     # dry run
```

The generator uses only the Python standard library (PyYAML is optional and only
used to read `suite.yaml`; without it the baked-in defaults are used). All
circuits use the `qelib1.inc` gate subset supported by `qco.ir` so they load
straight into the pipeline:

```python
from qco.pipeline import optimize
res = optimize("circuits/ghz_n5.qasm")
print(res.reward, len(res.pareto_front))
```

## Families

| Prefix | Family | Notes |
|--------|--------|-------|
| `ghz_n*` | GHZ state | linear CX chain |
| `qft_n*` | Quantum Fourier Transform | `h` + `cp` ladder + terminal swaps |
| `grover_w*_i*` | Grover search | `w` = working qubits; extra ancillas added for the mcx ladder; `i` = iterations |
| `qaoa_n*_p*` | QAOA on a random 3-regular graph | cost layer = `cx · rz · cx`, mixer = `rx` |
| `rand_cliffordt_n*` | Random Clifford+T | matches Tao et al. 2026 training distribution |
| `rand_cnotpauli_n*` | Random CNOT+Pauli | isolates two-qubit-gate reduction |

External corpora (MQT Bench, QASMBench) are added in Phase 3 once the `compiler`
extra is installed.
