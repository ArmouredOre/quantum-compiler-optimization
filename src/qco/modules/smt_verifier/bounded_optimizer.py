import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from qco.ir.intermediate_representation import Gate, IntermediateRepresentation
from qco.modules.smt_verifier.equivalence import EquivalenceChecker, circuit_unitary


@dataclass(slots=True)
class ResynthConfig:
    max_qubits: int = 4
    max_gates: int = 12
    solver: str = "z3"          # "z3" | "cvc5"
    timeout_s: float = 30.0
    gate_set: tuple[str, ...] = ("cx", "cz", "h", "x", "z", "s", "t", "rz", "rx")


@dataclass(slots=True)
class ResynthResult:
    """Stable re-synthesis result handed to M5 (Issue #33).

    Supports tuple unpacking:
        ``ir, status, solve_s = resynthesizer.resynthesize(window)``
    as well as attribute access:
        ``result.ir``, ``result.status``, ``result.solve_s``, ``result.reduced``, ``result.gates_saved``, ``result.from_cache``
    """
    ir: IntermediateRepresentation
    status: str            # "optimal" | "reduced" | "timeout" | "unreduced" | "not_applicable"
    solve_s: float
    reduced: bool = False
    gates_saved: int = 0
    from_cache: bool = False

    def __iter__(self):
        """Allow unpacking as (ir, status, solve_s)."""
        return iter((self.ir, self.status, self.solve_s))

    def __getitem__(self, idx: int) -> Any:
        return (self.ir, self.status, self.solve_s)[idx]

    def __len__(self) -> int:
        return 3

    @property
    def circuit(self) -> IntermediateRepresentation:
        return self.ir


@dataclass(slots=True)
class ResynthOutcome:
    """Outcome of exact bounded re-synthesis (Novelty N1 / M3 Label Source).

    Supports attribute access:
        ``outcome.reduced``, ``outcome.gates_saved``, ``outcome.solve_s``
    as well as tuple unpacking:
        ``window, reduced, gates_saved, solve_s = resynthesize_window(window)``
    """
    window: IntermediateRepresentation
    reduced: bool
    gates_saved: int
    solve_s: float
    circuit: IntermediateRepresentation
    optimality: str = "optimal"  # "optimal" | "reduced" | "timeout" | "unreduced" | "not_applicable"
    from_cache: bool = False

    def __iter__(self):
        """Allow unpacking as (window, reduced, gates_saved, solve_s)."""
        return iter((self.window, self.reduced, self.gates_saved, self.solve_s))


@dataclass(slots=True)
class _CachedResynth:
    status: str
    reduced: bool
    gates_saved: int
    canonical_gates: list[Gate]
    timeout_s: float


def canonical_window_form(
    window: IntermediateRepresentation,
) -> tuple[tuple[int, tuple[Any, ...]], list[int]]:
    """Return the canonical key for a window and the ordered list of active qubits.

    Active qubits are sorted and mapped to 0..k-1.
    Parametric angles are rounded to 8 decimal places for canonical hash stability.
    """
    active = sorted(set(q for g in window.gates for q in g.qubits))
    qmap = {q: i for i, q in enumerate(active)}
    gate_tuples = tuple(
        (g.name, tuple(qmap[q] for q in g.qubits), tuple(round(p, 8) for p in g.params))
        for g in window.gates
    )
    return (len(active), gate_tuples), active


class BoundedResynthesizer:
    def __init__(self, config: ResynthConfig | None = None):
        self.config = config or ResynthConfig()
        self._checker = EquivalenceChecker(backend="numeric")
        self._cache: dict[tuple[int, tuple[Any, ...]], _CachedResynth] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def cache_info(self) -> dict[str, int]:
        """Return cache hits, misses, and current cached entry count."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear cached results and reset statistics."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def applicable(self, window: IntermediateRepresentation) -> bool:
        """Check if window size is within synthesis bounds.

        Evaluates effective active qubits (qubits touched by gates in the window)
        so that sub-circuit blocks carved from larger registers are handled properly.
        """
        num_active = len(set(q for g in window.gates for q in g.qubits))
        effective_qubits = num_active if window.gates else window.num_qubits
        return (
            effective_qubits <= self.config.max_qubits
            and window.gate_count() <= self.config.max_gates
        )

    def resynthesize(
        self,
        window: IntermediateRepresentation,
        timeout_s: float | None = None,
    ) -> ResynthResult:
        """Hand M5 a stable resynthesize(window) -> (ir, status, solve_s) interface (Issue #33).

        Parameters
        ----------
        window : IntermediateRepresentation
            The input block/window to re-synthesize.
        timeout_s : float | None
            Per-call solver timeout override. If None, uses ``config.timeout_s``.

        Returns
        -------
        ResynthResult
            Dataclass unpackable as ``(ir, status, solve_s)`` with full metadata.
        """
        timeout = timeout_s if timeout_s is not None else self.config.timeout_s

        if not self.applicable(window):
            return ResynthResult(
                ir=window.copy(),
                status="not_applicable",
                solve_s=0.0,
                reduced=False,
                gates_saved=0,
                from_cache=False,
            )

        key, active = canonical_window_form(window)

        # Cache lookup: keyed by canonical window form
        if key in self._cache:
            entry = self._cache[key]
            # If the entry was a timeout and current call provides more time, re-solve
            if entry.status == "timeout" and timeout > entry.timeout_s:
                pass
            else:
                self.cache_hits += 1
                inv_qmap = {i: active[i] for i in range(len(active))}
                remapped_gates = [
                    Gate(g.name, tuple(inv_qmap[q] for q in g.qubits), tuple(g.params))
                    for g in entry.canonical_gates
                ]
                remapped_ir = IntermediateRepresentation(
                    num_qubits=window.num_qubits,
                    gates=remapped_gates,
                    name=f"{window.name}_opt" if entry.reduced else window.name,
                )
                return ResynthResult(
                    ir=remapped_ir,
                    status=entry.status,
                    solve_s=0.0,
                    reduced=entry.reduced,
                    gates_saved=entry.gates_saved,
                    from_cache=True,
                )

        # Cache miss: solve normalized canonical window (active qubits mapped to 0..k-1)
        self.cache_misses += 1
        qmap = {q: i for i, q in enumerate(active)}
        norm_window = IntermediateRepresentation(
            num_qubits=len(active),
            gates=[
                Gate(g.name, tuple(qmap[q] for q in g.qubits), tuple(g.params))
                for g in window.gates
            ],
            name=window.name,
        )

        norm_ir, status, reduced, gates_saved, solve_s = self._solve(norm_window, timeout)

        # Store canonical result in cache
        entry = _CachedResynth(
            status=status,
            reduced=reduced,
            gates_saved=gates_saved,
            canonical_gates=[Gate(g.name, tuple(g.qubits), tuple(g.params)) for g in norm_ir.gates],
            timeout_s=timeout,
        )
        self._cache[key] = entry

        # Remap canonical gates back to original window qubits
        inv_qmap = {i: active[i] for i in range(len(active))}
        remapped_gates = [
            Gate(g.name, tuple(inv_qmap[q] for q in g.qubits), tuple(g.params))
            for g in norm_ir.gates
        ]
        remapped_ir = IntermediateRepresentation(
            num_qubits=window.num_qubits,
            gates=remapped_gates,
            name=f"{window.name}_opt" if reduced else window.name,
        )

        return ResynthResult(
            ir=remapped_ir,
            status=status,
            solve_s=solve_s,
            reduced=reduced,
            gates_saved=gates_saved,
            from_cache=False,
        )

    def resynthesize_outcome(
        self,
        window: IntermediateRepresentation,
        timeout_s: float | None = None,
    ) -> ResynthOutcome:
        """Run re-synthesis and return the full outcome record for Module B (M3)."""
        res = self.resynthesize(window, timeout_s=timeout_s)
        return ResynthOutcome(
            window=window,
            reduced=res.reduced,
            gates_saved=res.gates_saved,
            solve_s=res.solve_s,
            circuit=res.ir,
            optimality=res.status,
            from_cache=res.from_cache,
        )

    def _get_candidate_pool(self, window: IntermediateRepresentation) -> list[Gate]:
        """Generate single-gate candidates from gate_set on active qubits."""
        n = window.num_qubits
        pool: list[Gate] = []
        gate_set = set(self.config.gate_set)

        # Collect angles from existing parametric gates plus canonical fractions of pi
        param_angles = {math.pi / 4, math.pi / 2, math.pi, -math.pi / 4, -math.pi / 2}
        for g in window.gates:
            if g.params:
                param_angles.add(g.params[0])
                param_angles.add(-g.params[0])

        # 1-qubit gates
        for q in range(n):
            for name in ("h", "x", "z", "s", "t", "sdg", "tdg", "y"):
                if name in gate_set:
                    pool.append(Gate(name, (q,)))
            for name in ("rz", "rx", "ry", "p"):
                if name in gate_set:
                    for angle in sorted(param_angles):
                        pool.append(Gate(name, (q,), (angle,)))

        # 2-qubit gates
        for c, t in itertools.permutations(range(n), 2):
            if "cx" in gate_set:
                pool.append(Gate("cx", (c, t)))

        for c, t in itertools.combinations(range(n), 2):
            if "cz" in gate_set:
                pool.append(Gate("cz", (c, t)))
            if "swap" in gate_set:
                pool.append(Gate("swap", (c, t)))

        return pool

    def _solve(
        self,
        window: IntermediateRepresentation,
        timeout_s: float,
    ) -> tuple[IntermediateRepresentation, str, bool, int, float]:
        """Iterative deepening exact re-synthesis over gate_set up to max_gates.

        Returns:
            (synthesized_ir, status, reduced, gates_saved, solve_s)
        """
        start_t = time.perf_counter()
        k = window.gate_count()
        n = window.num_qubits

        # Base case: 0-gate circuit
        if k == 0:
            solve_s = time.perf_counter() - start_t
            return (window.copy(), "optimal", False, 0, solve_s)

        u_target = circuit_unitary(window)
        dim = 1 << n

        def is_identity(u) -> bool:
            flat_diag = [u[i][i] for i in range(dim)]
            p0 = flat_diag[0]
            if abs(p0) < 1e-6:
                return False
            ph = p0 / abs(p0)
            for i in range(dim):
                for j in range(dim):
                    expected = ph if i == j else 0j
                    if abs(u[i][j] - expected) > 1e-5:
                        return False
            return True

        # Check depth 0 (identity)
        if is_identity(u_target):
            empty_circuit = IntermediateRepresentation(num_qubits=n, name=f"{window.name}_opt")
            solve_s = time.perf_counter() - start_t
            return (empty_circuit, "optimal", True, k, solve_s)

        # Depth 1 .. k - 1: Iterative deepening search with strict per-call timeout
        candidate_pool = self._get_candidate_pool(window)
        timeout = timeout_s
        timed_out = False

        for depth in range(1, k):
            if time.perf_counter() - start_t >= timeout:
                timed_out = True
                break

            found_circuit, depth_timed_out = self._search_depth(
                n=n,
                u_target=u_target,
                depth=depth,
                pool=candidate_pool,
                start_t=start_t,
                timeout=timeout,
            )
            if depth_timed_out:
                timed_out = True
                break

            if found_circuit is not None:
                solve_s = time.perf_counter() - start_t
                return (found_circuit, "optimal", True, k - depth, solve_s)

        solve_s = time.perf_counter() - start_t
        if timed_out or solve_s >= timeout:
            return (window.copy(), "timeout", False, 0, solve_s)

        # If no depth < k matches, the window is already optimal
        return (window.copy(), "optimal", False, 0, solve_s)

    def _search_depth(
        self,
        n: int,
        u_target: list[list[complex]],
        depth: int,
        pool: list[Gate],
        start_t: float,
        timeout: float,
    ) -> tuple[IntermediateRepresentation | None, bool]:
        """Search for an equivalent circuit of exactly ``depth`` gates.

        Returns:
            (circuit_or_none, timed_out_bool)
        """
        prefix_gates: list[Gate] = []

        def backtrack(d_remaining: int) -> tuple[IntermediateRepresentation | None, bool]:
            if time.perf_counter() - start_t >= timeout:
                return None, True

            if d_remaining == 0:
                trial = IntermediateRepresentation(num_qubits=n, gates=list(prefix_gates))
                u_trial = circuit_unitary(trial)
                if self._checker.check_unitaries(u_target, u_trial).equivalent:
                    return trial, False
                return None, False

            prev_gate = prefix_gates[-1] if prefix_gates else None

            for g in pool:
                if time.perf_counter() - start_t >= timeout:
                    return None, True

                # Symmetry pruning: avoid consecutive identical self-inverse gates
                if prev_gate is not None:
                    if g.name == prev_gate.name and g.qubits == prev_gate.qubits:
                        if g.name in ("h", "x", "z", "cx", "cz", "swap"):
                            continue

                prefix_gates.append(g)
                res, is_to = backtrack(d_remaining - 1)
                prefix_gates.pop()

                if is_to:
                    return None, True
                if res is not None:
                    return res, False

            return None, False

        return backtrack(depth)


def resynthesize_window(
    window: IntermediateRepresentation,
    config: ResynthConfig | None = None,
    timeout_s: float | None = None,
) -> ResynthOutcome:
    """Re-synthesis outcome API for Module B (M3) and pipeline window optimization.

    Parameters
    ----------
    window : IntermediateRepresentation
        The input circuit window (<= 4 qubits, <= 12 gates by default).
    config : ResynthConfig | None
        Optional re-synthesis configuration.
    timeout_s : float | None
        Optional per-call solver timeout override.

    Returns
    -------
    ResynthOutcome
        The re-synthesis result, unpackable as ``(window, reduced, gates_saved, solve_s)``.
    """
    resynthesizer = BoundedResynthesizer(config=config)
    return resynthesizer.resynthesize_outcome(window, timeout_s=timeout_s)
