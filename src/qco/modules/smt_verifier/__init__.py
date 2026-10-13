"""Module C — SAT/SMT-based verifier and bounded optimizer (Sahib Singh)."""

from qco.modules.smt_verifier.equivalence import (
    EquivalenceChecker,
    EquivalenceResult,
    apply_candidate_to_ir,
    check_smt,
    circuit_unitary,
    verify_candidates,
    verify_candidates_detailed,
)
from qco.modules.smt_verifier.bounded_optimizer import (
    BoundedResynthesizer,
    ResynthConfig,
    ResynthOutcome,
    ResynthResult,
    canonical_window_form,
    resynthesize_window,
)
from qco.modules.smt_verifier.phase_poly import (
    PhasePolynomial,
    check_phase_poly,
    extract_phase_polynomial,
    is_phase_poly_circuit,
)

__all__ = [
    "EquivalenceChecker",
    "EquivalenceResult",
    "BoundedResynthesizer",
    "ResynthConfig",
    "ResynthOutcome",
    "ResynthResult",
    "canonical_window_form",
    "resynthesize_window",
    "PhasePolynomial",
    "check_phase_poly",
    "extract_phase_polynomial",
    "is_phase_poly_circuit",
    "check_smt",
    "circuit_unitary",
    "apply_candidate_to_ir",
    "verify_candidates",
    "verify_candidates_detailed",
]

