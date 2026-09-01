"""The internal IR that every pipeline stage reads and rewrites.

Fully implemented in Phase 2 because Stages 2-6 all depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

# Single-qubit self-inverse gates (g . g = I) and 2-qubit self-inverse gates.
SELF_INVERSE = {"x", "y", "z", "h", "cx", "cz", "swap"}
# Rotation gates: g(theta) . g(-theta) = I  (used by cancellation heuristics).
PARAMETRIC = {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz", "rxx"}
# Number of qubits each supported gate acts on.
ARITY = {
    "x": 1, "y": 1, "z": 1, "h": 1, "s": 1, "sdg": 1, "t": 1, "tdg": 1,
    "rx": 1, "ry": 1, "rz": 1, "p": 1,
    "cx": 2, "cz": 2, "swap": 2, "cp": 2, "crx": 2, "cry": 2, "crz": 2, "rzz": 2, "rxx": 2,
    "ccx": 3,
}


@dataclass(frozen=True, slots=True)
class Gate:
    """A single gate application.

    Attributes
    ----------
    name : canonical lower-case gate name (see ``ARITY``).
    qubits : tuple of qubit indices the gate acts on, control(s) first.
    params : tuple of real parameters (radians) for parametric gates.
    tag : optional free-form marker used by modules to annotate provenance.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    tag: str | None = None

    def __post_init__(self) -> None:
        expected = ARITY.get(self.name)
        if expected is not None and len(self.qubits) != expected:
            raise ValueError(
                f"gate {self.name!r} expects {expected} qubits, got {self.qubits!r}"
            )

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

    @property
    def is_multi_qubit(self) -> bool:
        return len(self.qubits) >= 2

    def inverse(self) -> "Gate":
        """Return the inverse gate (used by the SMT verifier and cancellation)."""
        if self.name in SELF_INVERSE:
            return self
        if self.name in PARAMETRIC:
            return replace(self, params=tuple(-p for p in self.params))
        inv = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        if self.name in inv:
            return replace(self, name=inv[self.name])
        raise NotImplementedError(f"no known inverse for gate {self.name!r}")


@dataclass(slots=True)
class IntermediateRepresentation:
    """Ordered list of gates over ``num_qubits`` qubits."""

    num_qubits: int
    gates: list[Gate] = field(default_factory=list)
    name: str = "circuit"

    # -- construction -----------------------------------------------------
    def add(self, name: str, qubits: Iterable[int], params: Iterable[float] = ()) -> "IntermediateRepresentation":
        g = Gate(name=name, qubits=tuple(qubits), params=tuple(params))
        for q in g.qubits:
            if not 0 <= q < self.num_qubits:
                raise IndexError(f"qubit {q} out of range for {self.num_qubits}-qubit circuit")
        self.gates.append(g)
        return self

    def copy(self) -> "IntermediateRepresentation":
        return IntermediateRepresentation(self.num_qubits, list(self.gates), self.name)

    # -- metrics (consumed by qco.evaluation.metrics) --------------------
    def gate_count(self) -> int:
        return len(self.gates)

    def two_qubit_gate_count(self) -> int:
        return sum(1 for g in self.gates if g.is_two_qubit)

    def depth(self) -> int:
        """Critical-path length: greedy as-soon-as-possible layering."""
        frontier = [0] * self.num_qubits
        for g in self.gates:
            layer = max(frontier[q] for q in g.qubits) + 1
            for q in g.qubits:
                frontier[q] = layer
        return max(frontier, default=0)

    # -- iteration / dunder --------------------------------------------------
    def __iter__(self) -> Iterator[Gate]:
        return iter(self.gates)

    def __len__(self) -> int:
        return len(self.gates)

    # -- serialization -------------------------------------------------------
    def to_qasm2(self) -> str:
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.num_qubits}];"]
        for g in self.gates:
            p = f"({','.join(f'{x:.10g}' for x in g.params)})" if g.params else ""
            args = ",".join(f"q[{q}]" for q in g.qubits)
            lines.append(f"{g.name}{p} {args};")
        return "\n".join(lines) + "\n"
