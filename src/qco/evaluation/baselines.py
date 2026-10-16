"""Baseline compilers for Stage 6 comparison (#23 Qiskit O0-O3, #32 TKET / PyZX / BQSKit).

Each factory returns an optimizer ``IntermediateRepresentation -> IntermediateRepresentation``
that round-trips through the tool via OpenQASM 2.0. Tools emit different gate
sets (Qiskit keeps ``ccx``; TKET / PyZX / BQSKit decompose it), so compare
``cx_cost`` - multi-qubit gates in CX-equivalents - rather than raw gate count.

Missing packages raise ``ImportError`` from the factory; a circuit the tool
can't handle raises ``Unsupported`` from the optimizer, and the runner skips
that row instead of reporting a fake 0% reduction.
"""

from __future__ import annotations

import functools
from typing import Callable

from qco.graphs.hypergraph import gate_cost
from qco.ir.intermediate_representation import ARITY, IntermediateRepresentation
from qco.ir.parser import from_qasm2

Optimizer = Callable[[IntermediateRepresentation], IntermediateRepresentation]


class Unsupported(ValueError):
    """The baseline tool can't process this circuit (gate set, size, ...)."""


def cx_cost(ir: IntermediateRepresentation) -> int:
    """Multi-qubit gate cost in CX-equivalents (basis-independent comparison)."""
    return sum(gate_cost(g.name) for g in ir.gates if g.is_multi_qubit)


def equivalence_status(before: IntermediateRepresentation, after: IntermediateRepresentation,
                       max_sim_qubits: int = 22) -> bool | None:
    """Is ``after`` equivalent to ``before`` (up to global phase)? None = could not decide.

    Module C's checker first (dense <= 8 qubits, phase-polynomial beyond). Otherwise, with
    qiskit installed, a statevector check on two random input states up to ``max_sim_qubits``
    (a counterexample is conclusive; agreement is strong but probabilistic evidence).
    """
    from qco.modules.smt_verifier.equivalence import EquivalenceChecker

    res = EquivalenceChecker().check(before, after)
    if res.method != "unverified":
        return res.equivalent
    if before.num_qubits > max_sim_qubits:
        return None
    try:
        import numpy as np
        from qiskit.quantum_info import random_statevector
    except ImportError:
        return None
    qa, qb = _to_qiskit(before), _to_qiskit(after)
    for seed in range(2):
        psi = random_statevector(2 ** before.num_qubits, seed=seed)
        if abs(abs(np.vdot(psi.evolve(qa).data, psi.evolve(qb).data)) - 1) > 1e-6:
            return False
    return True


def _to_qiskit(ir: IntermediateRepresentation):
    """IR -> ``QuantumCircuit`` via Qiskit's standard-gate mapping (covers every IR gate,
    unlike the legacy QASM 2 loader, which lacks e.g. ``ryy``)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import get_standard_gate_name_mapping

    gates = get_standard_gate_name_mapping()
    qc = QuantumCircuit(ir.num_qubits)
    for g in ir.gates:
        std = gates[g.name]
        qc.append(type(std)(*g.params) if g.params else std, list(g.qubits))
    return qc


def _guard(tool: str):
    """Turn a tool crash on one circuit into ``Unsupported`` (row skipped, run continues)."""

    def wrap(fn: Optimizer) -> Optimizer:
        @functools.wraps(fn)
        def run(ir: IntermediateRepresentation) -> IntermediateRepresentation:
            try:
                return fn(ir)
            except Unsupported:
                raise
            except Exception as exc:  # noqa: BLE001 - third-party tools raise anything
                raise Unsupported(f"{tool}: {type(exc).__name__}: {exc}") from exc

        return run

    return wrap


def _back(qasm: str, ir: IntermediateRepresentation) -> IntermediateRepresentation:
    try:
        out = from_qasm2(qasm)
    except ValueError as exc:  # tool emitted a gate outside the IR
        raise Unsupported(str(exc)) from exc
    out.name = ir.name
    return out


def qiskit_baseline(level: int) -> Optimizer:
    """``transpile(optimization_level=level)`` restricted to the IR gate set."""
    from qiskit import qasm2, transpile

    @_guard(f"qiskit-O{level}")
    def run(ir: IntermediateRepresentation) -> IntermediateRepresentation:
        qc = qasm2.loads(ir.to_qasm2(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        out = transpile(qc, basis_gates=list(ARITY), optimization_level=level, seed_transpiler=0)
        return _back(qasm2.dumps(out), ir)

    return run


def tket_baseline() -> Optimizer:
    """TKET ``FullPeepholeOptimise``, rebased to {CX, Rz, Rx, H}."""
    from pytket.circuit import OpType
    from pytket.passes import AutoRebase, DecomposeBoxes, FullPeepholeOptimise
    from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

    @_guard("tket")
    def run(ir: IntermediateRepresentation) -> IntermediateRepresentation:
        c = circuit_from_qasm_str(ir.to_qasm2())
        DecomposeBoxes().apply(c)  # pytket parses some qelib1 gates (e.g. cp) as CustomGate
        FullPeepholeOptimise().apply(c)
        AutoRebase({OpType.CX, OpType.Rz, OpType.Rx, OpType.H}).apply(c)
        return _back(circuit_to_qasm_str(c), ir)

    return run


def pyzx_baseline() -> Optimizer:
    """PyZX ``full_reduce`` + circuit extraction + ``basic_optimization``."""
    import pyzx as zx

    @_guard("pyzx")
    def run(ir: IntermediateRepresentation) -> IntermediateRepresentation:
        g = zx.Circuit.from_qasm(ir.to_qasm2()).to_graph()
        zx.full_reduce(g)
        out = zx.basic_optimization(zx.extract_circuit(g.copy()).to_basic_gates())
        return _back(out.to_qasm(), ir)

    return run


def bqskit_baseline(level: int = 1, max_qubits: int = 8) -> Optimizer:
    """BQSKit ``compile`` to {CNOT, RZ, RX}. Instantiation-based synthesis is
    exponential in width, so circuits above ``max_qubits`` are skipped."""
    from bqskit import MachineModel, compile
    from bqskit.ir.gates import CNOTGate, RXGate, RZGate
    from bqskit.ir.lang.qasm2 import OPENQASM2Language

    lang = OPENQASM2Language()

    @_guard("bqskit")
    def run(ir: IntermediateRepresentation) -> IntermediateRepresentation:
        if ir.num_qubits > max_qubits:
            raise Unsupported(f"bqskit: {ir.num_qubits} qubits > max_qubits={max_qubits}")
        circ = lang.decode(ir.to_qasm2())
        model = MachineModel(circ.num_qudits, gate_set={CNOTGate(), RZGate(), RXGate()})
        out = compile(circ, model=model, optimization_level=level, seed=0)
        return _back(lang.encode(out), ir)

    return run


def available_baselines() -> dict[str, Optimizer]:
    """Every baseline whose package is installed, in report order."""
    factories: dict[str, Callable[[], Optimizer]] = {
        **{f"qiskit-O{lvl}": (lambda lvl=lvl: qiskit_baseline(lvl)) for lvl in range(4)},
        "tket": tket_baseline,
        "pyzx": pyzx_baseline,
        "bqskit": bqskit_baseline,
    }
    out: dict[str, Optimizer] = {}
    for name, make in factories.items():
        try:
            out[name] = make()
        except ImportError:
            continue
    return out
