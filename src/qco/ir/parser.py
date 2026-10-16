"""Front-end parsers: OpenQASM 2.0 / Qiskit / Cirq  ->  internal IR.

* ``from_qasm2`` : the ``qelib1.inc`` subset in ``ARITY`` (+ ``ALIASES``); all
  ``qreg`` declarations are flattened into one index space in declaration order.
* ``from_qiskit`` : walks ``QuantumCircuit.data`` directly.
* ``from_cirq`` : goes through ``cirq.qasm`` and ``from_qasm2``.

Non-unitary instructions (measure, reset, barrier) are dropped: the optimizer
works on the unitary part only. Gates outside ``ARITY`` raise ``ValueError`` -
transpile to a supported basis first.
"""

from __future__ import annotations

import math
import re
from typing import Any

from qco.ir.intermediate_representation import ALIASES, ARITY, IntermediateRepresentation

_QREG = re.compile(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_GATE = re.compile(r"^\s*(\w+)\s*(\([^)]*\))?\s+(.+?)\s*;\s*$")
_QUBIT = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")

_CONSTS = {"pi": math.pi, "tau": math.tau, "euler": math.e}
_SKIP = ("//", "OPENQASM", "include", "qreg", "creg", "barrier", "measure", "reset", "opaque")


def _canonical(name: str) -> str:
    name = name.lower()
    name = ALIASES.get(name, name)
    if name not in ARITY:
        raise ValueError(f"gate {name!r} not supported by the IR (see ARITY)")
    return name


def _eval_param(expr: str) -> float:
    expr = expr.strip()
    for name, val in _CONSTS.items():
        expr = re.sub(rf"\b{name}\b", repr(val), expr)
    # Only arithmetic on numbers/operators is permitted.
    if "**" in expr or "//" in expr or not re.fullmatch(r"[0-9eE+\-*/(). ]+", expr):
        raise ValueError(f"unsupported parameter expression: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - sandboxed above


def from_qasm2(text: str) -> IntermediateRepresentation:
    """Parse an OpenQASM 2.0 string into an :class:`IntermediateRepresentation`."""
    text = re.sub(r"//[^\n]*", "", text)  # comments must not reach the qreg scan or the gate regex
    offset: dict[str, int] = {}
    nq = 0
    for m in _QREG.finditer(text):
        offset[m.group(1)] = nq
        nq += int(m.group(2))
    if not offset:
        raise ValueError("no `qreg` declaration found")
    ir = IntermediateRepresentation(num_qubits=nq)

    in_gate_def = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("gate "):  # custom gate definition: skip header + body
            in_gate_def = "}" not in line
            continue
        if in_gate_def:
            in_gate_def = "}" not in line
            continue
        if not line or line.startswith(_SKIP):
            continue
        gm = _GATE.match(line)
        if not gm:
            raise ValueError(f"cannot parse line: {line!r}")
        name = _canonical(gm.group(1))
        params = ()
        if gm.group(2):
            inner = gm.group(2)[1:-1].strip()
            if inner:
                params = tuple(_eval_param(p) for p in inner.split(","))
        qubits = tuple(offset[reg] + int(q) for reg, q in _QUBIT.findall(gm.group(3)))
        ir.add(name, qubits, params)
    return ir


def from_qiskit(circuit: Any, bindings: dict | None = None) -> IntermediateRepresentation:
    """Convert a ``qiskit.QuantumCircuit`` to the internal IR.

    ``bindings`` maps ``Parameter`` -> value; every parameter must end up bound.
    Registers are flattened via ``circuit.find_bit`` (declaration order).
    """
    if bindings:
        circuit = circuit.assign_parameters(bindings)
    if circuit.parameters:
        raise ValueError(f"unbound parameters: {sorted(p.name for p in circuit.parameters)}")
    ir = IntermediateRepresentation(num_qubits=circuit.num_qubits, name=circuit.name)
    for inst in circuit.data:
        op = inst.operation
        if op.name in ("measure", "reset", "barrier", "delay"):
            continue
        if getattr(op, "condition", None) is not None:
            raise ValueError(f"classically conditioned gate {op.name!r} is not supported")
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        ir.add(_canonical(op.name), qubits, tuple(float(p) for p in op.params))
    return ir


def from_cirq(circuit: Any, resolver: Any = None) -> IntermediateRepresentation:
    """Convert a ``cirq.Circuit`` to the internal IR.

    ``resolver`` is anything ``cirq.resolve_parameters`` accepts (e.g. a dict of
    sympy symbol -> value). Qubits are numbered in cirq's sorted order.
    """
    import cirq

    if resolver is not None:
        circuit = cirq.resolve_parameters(circuit, resolver)
    if cirq.is_parameterized(circuit):
        raise ValueError("unbound parameters in cirq circuit; pass `resolver`")
    return from_qasm2(cirq.qasm(circuit))


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
