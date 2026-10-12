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
from qco.modules.smt_verifier.bounded_optimizer import BoundedResynthesizer

__all__ = [
    "EquivalenceChecker",
    "EquivalenceResult",
    "BoundedResynthesizer",
    "check_smt",
    "circuit_unitary",
    "apply_candidate_to_ir",
    "verify_candidates",
    "verify_candidates_detailed",
]
