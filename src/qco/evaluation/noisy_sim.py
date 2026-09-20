"""Validation-only noisy density-matrix simulator (<= 8 qubits).

Not on any pipeline hot path. Exists to sanity-check
``qco.modules.evolutionary.nsga2.structure_aware_fidelity_surrogate`` against a
real (if toy) noise model, per issue #13's acceptance criterion: Spearman
correlation > 0.8 between the surrogate and a noisy-sim fidelity estimate on
small circuits. See ``tests/test_evaluation.py``.

Reuses Module C's dependency-free per-gate unitary builder
(``qco.modules.smt_verifier.equivalence._gate_on_register``) for the ideal gate
operators, so this doesn't re-derive a second gate-matrix table that could
drift from the equivalence checker's. Local depolarizing noise (one qubit at a
time, Kraus decomposition) is applied to every qubit a gate touches, at the
same per-arity error rate ``qco.evaluation.metrics`` already uses elsewhere -
a standard approximation when no full multi-qubit error channel is modeled.
"""

from __future__ import annotations

import numpy as np

from qco.evaluation.metrics import DEFAULT_ERROR_RATE
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.smt_verifier.equivalence import _gate_on_register

_I2 = np.eye(2, dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Single-qubit depolarizing channel Kraus operators for total error prob ``p``."""
    p = max(0.0, min(1.0, p))
    return [
        np.sqrt(max(0.0, 1 - 3 * p / 4)) * _I2,
        np.sqrt(p / 4) * _PAULI_X,
        np.sqrt(p / 4) * _PAULI_Y,
        np.sqrt(p / 4) * _PAULI_Z,
    ]


def _embed_1q(op: np.ndarray, target: int, n: int) -> np.ndarray:
    """Embed a 2x2 single-qubit operator into the full n-qubit space.

    Matches the little-endian convention used throughout the IR (qubit ``q``
    is bit ``q`` of the basis index): qubit ``n-1`` is the outermost kron
    factor, qubit 0 the innermost.
    """
    mats = [_I2] * n
    mats[target] = op
    full = mats[n - 1]
    for q in range(n - 2, -1, -1):
        full = np.kron(full, mats[q])
    return full


def _apply_local_depolarizing(rho: np.ndarray, target: int, n: int, p: float) -> np.ndarray:
    out = np.zeros_like(rho)
    for k in _depolarizing_kraus(p):
        kn = _embed_1q(k, target, n)
        out += kn @ rho @ kn.conj().T
    return out


def noisy_sim_fidelity(
    circuit: IntermediateRepresentation,
    error_rate: dict[str, float] | None = None,
) -> float:
    """State fidelity <psi_ideal| rho_noisy |psi_ideal> from an exact per-gate
    local-depolarizing density-matrix simulation, starting from |0...0>.

    Deterministic: Kraus sums are applied exactly, no Monte Carlo sampling.
    """
    n = circuit.num_qubits
    if n > 8:
        raise ValueError("noisy_sim_fidelity limited to 8 qubits (dense 2^n x 2^n density matrix)")
    e = error_rate or DEFAULT_ERROR_RATE
    dim = 1 << n

    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0

    for g in circuit.gates:
        u = np.array(_gate_on_register(g, n), dtype=complex)
        psi = u @ psi
        rho = u @ rho @ u.conj().T
        p = e.get(f"{len(g.qubits)}q", e["2q"])
        for q in g.qubits:
            rho = _apply_local_depolarizing(rho, q, n, p)

    fidelity = np.real(psi.conj() @ rho @ psi)
    return float(np.clip(fidelity, 0.0, 1.0))
