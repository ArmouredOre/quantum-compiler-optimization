"""Stage 6 evaluation tests.

Validates the analytic fidelity surrogate
(``qco.modules.evolutionary.nsga2.structure_aware_fidelity_surrogate``) against
a real (if toy) noisy-sim baseline (``qco.evaluation.noisy_sim``), per issue
#13's acceptance criterion: Spearman correlation > 0.8 on small circuits.

Every circuit here comes from a seeded generator and both fidelity functions
are pure/deterministic (no RNG on this path) - the resulting rho is exactly
reproducible, unlike the NSGA-II genetic-search tests in
test_modules_and_pipeline.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
scipy_stats = pytest.importorskip("scipy.stats")

import benchmarks.generate_circuits as gen
from qco.evaluation.noisy_sim import noisy_sim_fidelity
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.evolutionary.nsga2 import structure_aware_fidelity_surrogate


def _small_benchmark_circuits() -> list[IntermediateRepresentation]:
    """A range of <= 8-qubit circuits, from near-trivial to noticeably lossy,
    so the correlation check isn't just comparing values clustered near 1.0.
    """
    qasm_circuits = [
        gen.ghz(2), gen.ghz(3), gen.ghz(5), gen.ghz(8),
        gen.qft(4), gen.qft(5), gen.qft(6),
        gen.grover(3, 1),
        gen.qaoa(6, 1, seed=1),
        gen.random_clifford_t(5, layers=10, t_fraction=0.15, seed=1),
        gen.random_cnot_pauli(5, layers=10, seed=1),
    ]
    circuits = [from_qasm2(c.text()) for c in qasm_circuits]

    shallow = IntermediateRepresentation(2, name="shallow")
    shallow.add("h", [0])
    circuits.append(shallow)

    deep = IntermediateRepresentation(3, name="deep_chain")
    for _ in range(15):
        deep.add("h", [0]).add("cx", [0, 1]).add("cx", [1, 2])
    circuits.append(deep)

    return circuits


def test_fidelity_surrogate_correlates_with_noisy_sim():
    circuits = _small_benchmark_circuits()
    surrogate = [structure_aware_fidelity_surrogate(c) for c in circuits]
    noisy = [noisy_sim_fidelity(c) for c in circuits]

    rho, p_value = scipy_stats.spearmanr(surrogate, noisy)
    assert rho > 0.8, f"surrogate vs noisy-sim Spearman rho={rho:.3f} (want > 0.8), p={p_value:.4g}"


def test_noisy_sim_fidelity_decreases_with_more_noisy_gates():
    empty = IntermediateRepresentation(2)
    short = IntermediateRepresentation(2)
    short.add("h", [0]).add("cx", [0, 1])
    long = IntermediateRepresentation(2)
    for _ in range(20):
        long.add("h", [0]).add("cx", [0, 1])

    assert noisy_sim_fidelity(empty) == 1.0
    assert noisy_sim_fidelity(empty) > noisy_sim_fidelity(short) > noisy_sim_fidelity(long)


def test_noisy_sim_fidelity_rejects_over_8_qubits():
    big = IntermediateRepresentation(9)
    big.add("h", [0])
    with pytest.raises(ValueError):
        noisy_sim_fidelity(big)
