"""Module C — SAT/SMT-based verifier and bounded optimizer (Sahib Singh)."""

from qco.modules.smt_verifier.equivalence import EquivalenceChecker, EquivalenceResult
from qco.modules.smt_verifier.bounded_optimizer import BoundedResynthesizer

__all__ = ["EquivalenceChecker", "EquivalenceResult", "BoundedResynthesizer"]
