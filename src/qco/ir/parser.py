"""Front-end parsers: OpenQASM 2.0 / Qiskit / Cirq  ->  internal IR.

Phase 2 status:
* ``from_qasm2`` : implemented for the ``qelib1.inc`` gate subset the benchmark
  suite uses (enough to load every circuit in ``benchmarks/circuits/``).
* ``from_qiskit`` / ``from_cirq`` : signatures frozen; bodies land in Phase 3
  (Member 1) once the ``compiler`` extra is installed.
"""

from __future__ import annotations

import math
import re
from typing import Any

from qco.ir.intermediate_representation import ARITY, IntermediateRepresentation

_QREG = re.compile(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_GATE = re.compile(r"^\s*(\w+)\s*(\([^)]*\))?\s+(.+?)\s*;\s*$")
_QUBIT = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")

_CONSTS = {"pi": math.pi, "tau": math.tau, "euler": math.e}


def _eval_param(expr: str) -> float:
    expr = expr.strip()
    for name, val in _CONSTS.items():
        expr = re.sub(rf"\b{name}\b", repr(val), expr)
    # Only arithmetic on numbers/operators is permitted.
    if not re.fullmatch(r"[0-9eE+\-*/(). ]+", expr):
        raise ValueError(f"unsupported parameter expression: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - sandboxed above


def from_qasm2(text: str) -> IntermediateRepresentation:
    """Parse an OpenQASM 2.0 string into an :class:`IntermediateRepresentation`."""
    m = _QREG.search(text)
    if not m:
        raise ValueError("no `qreg` declaration found")
    reg_name, nq = m.group(1), int(m.group(2))
    ir = IntermediateRepresentation(num_qubits=nq)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("//", "OPENQASM", "include", "qreg", "creg", "barrier", "measure")):
            continue
        gm = _GATE.match(line)
        if not gm:
            continue
        name = gm.group(1).lower()
        if name not in ARITY:
            raise ValueError(f"gate {name!r} not supported by the Phase 2 IR")
        params = ()
        if gm.group(2):
            inner = gm.group(2)[1:-1].strip()
            if inner:
                params = tuple(_eval_param(p) for p in inner.split(","))
        qubits = tuple(int(q) for reg, q in _QUBIT.findall(gm.group(3)) if reg == reg_name)
        ir.add(name, qubits, params)
    return ir


def from_qiskit(circuit: Any) -> IntermediateRepresentation:  # pragma: no cover - Phase 3
    """Convert a ``qiskit.QuantumCircuit`` to the internal IR.

    Phase 3 (Member 1): iterate ``circuit.data``, map standard gate names,
    resolve ``Parameter`` bindings, and flatten registers to a single index
    space.
    """
    raise NotImplementedError("from_qiskit lands in Phase 3 (needs the `compiler` extra)")


def from_cirq(circuit: Any) -> IntermediateRepresentation:  # pragma: no cover - Phase 3
    """Convert a ``cirq.Circuit`` to the internal IR (Phase 3, Member 1)."""
    raise NotImplementedError("from_cirq lands in Phase 3 (needs the `compiler` extra)")


def parse(source: Any) -> IntermediateRepresentation:
    """Dispatch on the source type."""
    if isinstance(source, IntermediateRepresentation):
        return source
    if isinstance(source, str):
        if "OPENQASM" in source:
            return from_qasm2(source)
        with open(source, "r", encoding="utf-8") as fh:  # treat as a path
            return from_qasm2(fh.read())
    mod = type(source).__module__.split(".")[0]
    if mod == "qiskit":
        return from_qiskit(source)
    if mod == "cirq":
        return from_cirq(source)
    raise TypeError(f"cannot parse object of type {type(source)!r}")
