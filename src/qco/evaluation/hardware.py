"""Stage 6 — opt-in hardware-aware cost model (``HardwareProfile``).

``Calibration`` (``calibration.py``) only knows per-arity gate time/error, so two
profiles can differ by *scale* but never in *structure*. This module adds the
structure that makes backends genuinely want different schedules, **without
changing** ``Calibration`` or any existing metric (everything here is new and
opt-in; nothing imports it by default):

* **topology** — ``all_to_all`` | ``line`` | ``ring`` | ``grid`` coupling graph;
* **SWAP routing cost** — a deterministic greedy router (identity initial layout,
  layout evolves as SWAPs are inserted) so cost depends on gate *order*;
* **native-gate cost** — per-gate-name time / error multipliers on top of a base
  ``Calibration`` (e.g. ``rzz`` is ~2 CX on a transmon chip, ~1 MS gate on ions);
* **parallelism limit** — ``max_parallel_2q`` (a single ion chain runs one
  entangling gate at a time).

The routed circuit is only *costed*, never returned: the optimizer's circuits stay
in logical form, so the IR, the SMT verifier and the pipeline are unaffected.

Profiles below are illustrative and order-of-magnitude, like the ``Calibration``
ones; they are **not** taken from a live backend's calibration data.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

from qco.evaluation.calibration import get_calibration
from qco.ir.intermediate_representation import IntermediateRepresentation

TOPOLOGIES = ("all_to_all", "line", "ring", "grid")


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    name: str
    description: str = ""
    calibration: str = "default"          # base per-arity durations / errors
    topology: str = "all_to_all"
    gate_time_factor: dict[str, float] = field(default_factory=dict)
    gate_error_factor: dict[str, float] = field(default_factory=dict)
    swap_two_qubit_gates: int = 3         # a SWAP costs this many two-qubit gates
    max_parallel_2q: int | None = None    # None = unlimited

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}, got {self.topology!r}")
        if self.max_parallel_2q is not None and self.max_parallel_2q < 1:
            raise ValueError("max_parallel_2q must be None or >= 1")
        get_calibration(self.calibration)  # raises KeyError for unknown base profile


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "ibm_grid_like": HardwareProfile(
        name="ibm_grid_like",
        description="Transmon-style: 2D nearest-neighbour grid, CX-native (rzz/rxx/cp ~ 2 CX), parallel gates.",
        calibration="superconducting_ibm_like",
        topology="grid",
        gate_time_factor={"rzz": 2.0, "rxx": 2.0, "ryy": 2.0, "cp": 2.0, "crz": 2.0, "crx": 2.0, "cry": 2.0, "swap": 3.0},
        gate_error_factor={"rzz": 2.0, "rxx": 2.0, "ryy": 2.0, "cp": 2.0, "crz": 2.0, "crx": 2.0, "cry": 2.0, "swap": 3.0},
    ),
    "line_superconducting_like": HardwareProfile(
        name="line_superconducting_like",
        description="Same gate costs as ibm_grid_like but a 1D chain: routing is expensive.",
        calibration="superconducting_ibm_like",
        topology="line",
        gate_time_factor={"rzz": 2.0, "rxx": 2.0, "ryy": 2.0, "cp": 2.0, "crz": 2.0, "crx": 2.0, "cry": 2.0, "swap": 3.0},
        gate_error_factor={"rzz": 2.0, "rxx": 2.0, "ryy": 2.0, "cp": 2.0, "crz": 2.0, "crx": 2.0, "cry": 2.0, "swap": 3.0},
    ),
    "ion_all_to_all_like": HardwareProfile(
        name="ion_all_to_all_like",
        description="Trapped-ion chain: all-to-all (no routing), MS-native (rzz/rxx cheap, cx slightly dearer), one entangling gate at a time.",
        calibration="trapped_ion_like",
        topology="all_to_all",
        gate_time_factor={"cx": 1.2, "cz": 1.2, "cp": 2.0, "crz": 2.0, "crx": 2.0, "cry": 2.0, "swap": 3.0},
        gate_error_factor={"cx": 1.2, "cz": 1.2, "cp": 2.0, "crz": 2.0, "crx": 2.0, "cry": 2.0, "swap": 3.0},
        max_parallel_2q=1,
    ),
}


def get_hardware(profile: "str | HardwareProfile") -> HardwareProfile:
    if isinstance(profile, HardwareProfile):
        return profile
    try:
        return HARDWARE_PROFILES[profile]
    except KeyError as exc:
        raise KeyError(f"unknown hardware profile {profile!r}; available: {', '.join(sorted(HARDWARE_PROFILES))}") from exc


def list_hardware() -> list[str]:
    return sorted(HARDWARE_PROFILES)


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _adjacency(topology: str, n: int) -> tuple[tuple[int, ...], ...]:
    nbrs: list[set[int]] = [set() for _ in range(n)]
    if topology == "all_to_all":
        for a in range(n):
            nbrs[a] = set(range(n)) - {a}
    elif topology in ("line", "ring"):
        for a in range(n - 1):
            nbrs[a].add(a + 1)
            nbrs[a + 1].add(a)
        if topology == "ring" and n > 2:
            nbrs[0].add(n - 1)
            nbrs[n - 1].add(0)
    else:  # grid
        cols = max(1, math.ceil(math.sqrt(n)))
        for q in range(n):
            r, c = divmod(q, cols)
            if c + 1 < cols and q + 1 < n:
                nbrs[q].add(q + 1)
                nbrs[q + 1].add(q)
            if q + cols < n:
                nbrs[q].add(q + cols)
                nbrs[q + cols].add(q)
    return tuple(tuple(sorted(s)) for s in nbrs)


def coupling_edges(profile: "str | HardwareProfile", n: int) -> list[tuple[int, int]]:
    adj = _adjacency(get_hardware(profile).topology, n)
    return [(a, b) for a in range(n) for b in adj[a] if a < b]


@lru_cache(maxsize=None)
def _shortest_path(topology: str, n: int, src: int, dst: int) -> tuple[int, ...]:
    adj = _adjacency(topology, n)
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    path = []
    u = dst
    while u is not None:
        path.append(u)
        u = prev[u]
    return tuple(reversed(path))


# --------------------------------------------------------------------------
# routing (cost model only)
# --------------------------------------------------------------------------

def route(circuit: IntermediateRepresentation, profile: "str | HardwareProfile") -> list[tuple[str, str, tuple[int, ...]]]:
    """Greedy SWAP routing of ``circuit`` in program order.

    Returns ops ``(kind, name, physical_qubits)`` with ``kind`` in ``{"gate", "swap"}``.
    Layout starts as the identity; to run a multi-qubit gate every operand is moved
    next to the first one along a shortest path (SWAPs update the layout). Because
    the layout evolves, the cost depends on the *order* of gates.
    """
    hw = get_hardware(profile)
    n = circuit.num_qubits
    phys = list(range(n))                # logical -> physical
    log_at = list(range(n))              # physical -> logical
    ops: list[tuple[str, str, tuple[int, ...]]] = []
    adj = _adjacency(hw.topology, n)
    for g in circuit.gates:
        qs = list(g.qubits)
        if len(qs) >= 2 and hw.topology != "all_to_all":
            anchor = qs[0]
            for other in qs[1:]:
                while phys[other] not in adj[phys[anchor]]:
                    path = _shortest_path(hw.topology, n, phys[other], phys[anchor])
                    a, b = path[0], path[1]          # move `other` one step toward the anchor
                    ops.append(("swap", "swap", (a, b)))
                    la, lb = log_at[a], log_at[b]
                    phys[la], phys[lb] = b, a
                    log_at[a], log_at[b] = lb, la
        ops.append(("gate", g.name, tuple(phys[q] for q in qs)))
    return ops


def swap_count(circuit: IntermediateRepresentation, profile: "str | HardwareProfile") -> int:
    return sum(1 for k, _, _ in route(circuit, profile) if k == "swap")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _op_time(hw: HardwareProfile, cal, kind: str, name: str, arity: int) -> float:
    if kind == "swap":
        return hw.swap_two_qubit_gates * cal.duration(2) * hw.gate_time_factor.get("cx", 1.0)
    return cal.duration(arity) * hw.gate_time_factor.get(name, 1.0)


def hardware_execution_time(circuit: IntermediateRepresentation, profile: "str | HardwareProfile") -> float:
    """Critical-path time (ns) of the *routed* circuit under the profile (ASAP, with the 2q-parallelism cap)."""
    hw = get_hardware(profile)
    cal = get_calibration(hw.calibration)
    frontier = [0.0] * circuit.num_qubits
    intervals: list[tuple[float, float]] = []          # running entangling ops
    end = 0.0
    for kind, name, qs in route(circuit, hw):
        dur = _op_time(hw, cal, kind, name, len(qs))
        start = max(frontier[q] for q in qs)
        if hw.max_parallel_2q and len(qs) >= 2:
            while True:
                overl = [e for s, e in intervals if s < start + dur and e > start]
                if len(overl) < hw.max_parallel_2q:
                    break
                start = min(overl)
            intervals.append((start, start + dur))
        for q in qs:
            frontier[q] = start + dur
        end = max(end, start + dur)
    return end


def hardware_fidelity(circuit: IntermediateRepresentation, profile: "str | HardwareProfile") -> float:
    """Product of per-op success probabilities of the routed circuit (depolarizing proxy)."""
    hw = get_hardware(profile)
    cal = get_calibration(hw.calibration)
    f = 1.0
    for kind, name, qs in route(circuit, hw):
        if kind == "swap":
            e = min(0.999, cal.error(2) * hw.gate_error_factor.get("cx", 1.0))
            f *= (1.0 - e) ** hw.swap_two_qubit_gates
        else:
            f *= 1.0 - min(0.999, cal.error(len(qs)) * hw.gate_error_factor.get(name, 1.0))
    return f
