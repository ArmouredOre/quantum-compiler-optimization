"""Stage 6 — named device calibration profiles.

Each profile supplies per-arity gate durations (ns) and error rates used by
``qco.evaluation.metrics.execution_time`` / ``estimated_fidelity``. Profiles are
looked up by name so the benchmark runner and reward signal can be pointed at a
different backend without touching call sites.

``"default"`` is a coarse, hand-picked superconducting-style profile (kept for
backward compatibility with Phase 2 numbers). ``"superconducting_ibm_like"`` and
``"trapped_ion_like"`` are illustrative, order-of-magnitude profiles built from
publicly reported ranges for the two dominant NISQ hardware families — they are
**not** pulled from a specific live backend's calibration data. Real per-backend
calibration (e.g. via Qiskit's `BackendProperties`) is wired in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Calibration:
    """Per-arity gate duration (ns) and error-rate table for one device profile."""

    name: str
    description: str
    gate_time_ns: dict[str, float] = field(default_factory=dict)
    error_rate: dict[str, float] = field(default_factory=dict)

    def duration(self, arity: int) -> float:
        key = f"{arity}q"
        return self.gate_time_ns.get(key, self.gate_time_ns["2q"])

    def error(self, arity: int) -> float:
        key = f"{arity}q"
        return self.error_rate.get(key, self.error_rate["2q"])


CALIBRATIONS: dict[str, Calibration] = {
    "default": Calibration(
        name="default",
        description="Coarse superconducting-style placeholder (Phase 2 default).",
        gate_time_ns={"1q": 35.0, "2q": 300.0, "3q": 600.0},
        error_rate={"1q": 1e-3, "2q": 1e-2, "3q": 3e-2},
    ),
    "superconducting_ibm_like": Calibration(
        name="superconducting_ibm_like",
        description=(
            "Illustrative fixed-frequency transmon profile: fast gates, "
            "moderate two-qubit error. Order-of-magnitude only."
        ),
        gate_time_ns={"1q": 30.0, "2q": 350.0, "3q": 700.0},
        error_rate={"1q": 3e-4, "2q": 8e-3, "3q": 2.4e-2},
    ),
    "trapped_ion_like": Calibration(
        name="trapped_ion_like",
        description=(
            "Illustrative trapped-ion profile: slow gates (microseconds), "
            "low error rates. Order-of-magnitude only."
        ),
        gate_time_ns={"1q": 3_000.0, "2q": 150_000.0, "3q": 300_000.0},
        error_rate={"1q": 1e-4, "2q": 1e-3, "3q": 3e-3},
    ),
}


def get_calibration(profile: "str | Calibration" = "default") -> Calibration:
    """Resolve a calibration by name, or pass one through unchanged."""
    if isinstance(profile, Calibration):
        return profile
    try:
        return CALIBRATIONS[profile]
    except KeyError as exc:
        available = ", ".join(sorted(CALIBRATIONS))
        raise KeyError(f"unknown calibration profile {profile!r}; available: {available}") from exc


def list_calibrations() -> list[str]:
    return sorted(CALIBRATIONS)
