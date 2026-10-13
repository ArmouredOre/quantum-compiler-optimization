"""Run every installed baseline compiler on one circuit; QASM on stdin, JSON rows on stdout.

The Streamlit app calls this in a subprocess: Qiskit's native QASM parser
segfaults when called from Streamlit's script thread, and a separate process
also keeps any third-party tool crash from taking the app down.

Usage: python app/run_baselines.py <calibration> < circuit.qasm
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qco.evaluation.baselines import Unsupported, available_baselines, cx_cost  # noqa: E402
from qco.evaluation.metrics import estimated_fidelity, execution_time  # noqa: E402
from qco.ir.parser import from_qasm2  # noqa: E402


def main() -> None:
    calibration = sys.argv[1] if len(sys.argv) > 1 else "default"
    circuit = from_qasm2(sys.stdin.read())
    rows = []
    for name, opt in available_baselines().items():
        t0 = time.perf_counter()
        try:
            out = opt(circuit)
        except Unsupported as exc:
            rows.append({"Optimizer": name, "Note": str(exc)[:80]})
            continue
        rows.append({
            "Optimizer": name,
            "Gates": out.gate_count(),
            "2-qubit gates": out.two_qubit_gate_count(),
            "CX-cost": cx_cost(out),
            "Depth": out.depth(),
            "Est. fidelity": estimated_fidelity(out, calibration).value,
            "Exec. time (ns)": execution_time(out, calibration).value,
            "Runtime (s)": time.perf_counter() - t0,
            "Note": "",
        })
    json.dump(rows, sys.stdout)


if __name__ == "__main__":
    main()
