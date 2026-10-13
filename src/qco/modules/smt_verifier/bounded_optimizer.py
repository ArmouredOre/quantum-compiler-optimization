import itertools
import math
import time
from dataclasses import dataclass
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
    optimality: str = "optimal"  # "optimal" | "reduced" | "timeout" | "unreduced"

    def __iter__(self):
        """Allow unpacking as (window, reduced, gates_saved, solve_s)."""
        return iter((self.window, self.reduced, self.gates_saved, self.solve_s))


class BoundedResynthesizer:
    def __init__(self, config: ResynthConfig | None = None):
        self.config = config or ResynthConfig()
        self._checker = EquivalenceChecker(backend="numeric")

    def applicable(self, window: IntermediateRepresentation) -> bool:
        return (window.num_qubits <= self.config.max_qubits
                and window.gate_count() <= self.config.max_gates)

    def resynthesize(self, window: IntermediateRepresentation) -> IntermediateRepresentation:
        if not self.applicable(window):
            return window
        return self._solve(window).circuit

    def resynthesize_outcome(self, window: IntermediateRepresentation) -> ResynthOutcome:
        """Run re-synthesis and return the full outcome record."""
        if not self.applicable(window):
            return ResynthOutcome(
                window=window,
                reduced=False,
                gates_saved=0,
                solve_s=0.0,
                circuit=window.copy(),
                optimality="unreduced",
            )
        return self._solve(window)

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
                    pool.append(Gate(name, [q]))
            for name in ("rz", "rx", "ry", "p"):
                if name in gate_set:
                    for angle in sorted(param_angles):
                        pool.append(Gate(name, [q], (angle,)))

        # 2-qubit gates
        for c, t in itertools.permutations(range(n), 2):
            if "cx" in gate_set:
                pool.append(Gate("cx", [c, t]))

        for c, t in itertools.combinations(range(n), 2):
            if "cz" in gate_set:
                pool.append(Gate("cz", [c, t]))
            if "swap" in gate_set:
                pool.append(Gate("swap", [c, t]))

        return pool

    def _solve(self, window: IntermediateRepresentation) -> ResynthOutcome:
        """Iterative deepening exact re-synthesis over gate_set up to max_gates."""
        start_t = time.perf_counter()
        k = window.gate_count()
        n = window.num_qubits

        # Base case: 0-gate circuit
        if k == 0:
            return ResynthOutcome(
                window=window,
                reduced=False,
                gates_saved=0,
                solve_s=time.perf_counter() - start_t,
                circuit=window.copy(),
                optimality="optimal",
            )

        u_target = circuit_unitary(window)
        dim = 1 << n

        def is_identity(u) -> bool:
            # Check if u == e^{i*phi} * I
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
            return ResynthOutcome(
                window=window,
                reduced=True,
                gates_saved=k,
                solve_s=time.perf_counter() - start_t,
                circuit=empty_circuit,
                optimality="optimal",
            )

        # Depth 1 .. k - 1: Iterative deepening search
        candidate_pool = self._get_candidate_pool(window)
        timeout = self.config.timeout_s

        for depth in range(1, k):
            now = time.perf_counter()
            if now - start_t > timeout:
                return ResynthOutcome(
                    window=window,
                    reduced=False,
                    gates_saved=0,
                    solve_s=now - start_t,
                    circuit=window.copy(),
                    optimality="timeout",
                )

            found_circuit = self._search_depth(
                n=n,
                u_target=u_target,
                depth=depth,
                pool=candidate_pool,
                start_t=start_t,
                timeout=timeout,
            )
            if found_circuit is not None:
                solve_s = time.perf_counter() - start_t
                return ResynthOutcome(
                    window=window,
                    reduced=True,
                    gates_saved=k - depth,
                    solve_s=solve_s,
                    circuit=found_circuit,
                    optimality="optimal",
                )

        # If no depth < k matches, the window is already optimal
        solve_s = time.perf_counter() - start_t
        return ResynthOutcome(
            window=window,
            reduced=False,
            gates_saved=0,
            solve_s=solve_s,
            circuit=window.copy(),
            optimality="optimal",
        )

    def _search_depth(
        self,
        n: int,
        u_target: list[list[complex]],
        depth: int,
        pool: list[Gate],
        start_t: float,
        timeout: float,
    ) -> IntermediateRepresentation | None:
        """Search for an equivalent circuit of exactly ``depth`` gates."""
        prefix_gates: list[Gate] = []

        def backtrack(d_remaining: int) -> IntermediateRepresentation | None:
            if time.perf_counter() - start_t > timeout:
                return None

            if d_remaining == 0:
                trial = IntermediateRepresentation(num_qubits=n, gates=list(prefix_gates))
                u_trial = circuit_unitary(trial)
                if self._checker.check_unitaries(u_target, u_trial).equivalent:
                    return trial
                return None

            prev_gate = prefix_gates[-1] if prefix_gates else None

            for g in pool:
                # Symmetry pruning: avoid consecutive identical self-inverse gates
                if prev_gate is not None:
                    if g.name == prev_gate.name and g.qubits == prev_gate.qubits:
                        if g.name in ("h", "x", "z", "cx", "cz", "swap"):
                            continue

                prefix_gates.append(g)
                res = backtrack(d_remaining - 1)
                prefix_gates.pop()
                if res is not None:
                    return res

            return None

        return backtrack(depth)


def resynthesize_window(
    window: IntermediateRepresentation,
    config: ResynthConfig | None = None,
) -> ResynthOutcome:
    """Re-synthesis outcome API for Module B (M3) and pipeline window optimization.

    Parameters
    ----------
    window : IntermediateRepresentation
        The input circuit window (<= 4 qubits, <= 12 gates by default).
    config : ResynthConfig | None
        Optional re-synthesis configuration.

    Returns
    -------
    ResynthOutcome
        The re-synthesis result, unpackable as ``(window, reduced, gates_saved, solve_s)``.
    """
    resynthesizer = BoundedResynthesizer(config=config)
    return resynthesizer.resynthesize_outcome(window)
