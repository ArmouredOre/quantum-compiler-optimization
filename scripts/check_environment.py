#!/usr/bin/env python3
"""Phase 2 environment check: report which optional research stacks are present.

    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import os
import sys

# Allow running before `pip install -e .`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

STACKS = {
    "core": ["yaml"],
    "compiler (Ajay M / Swaraj Rane)": ["qiskit", "qiskit_aer", "cirq", "pytket"],
    "graphs (Ajay M)": ["networkx"],
    "rl / Module A (Maanas Nair)": ["torch", "stable_baselines3", "gymnasium"],
    "gnn / Module B (Prisha)": ["torch", "torch_geometric"],
    "smt / Module C (Sahib Singh)": ["z3"],
    "ea / Module D (Swaraj Rane)": ["pymoo", "deap"],
    "viz / reporting": ["matplotlib", "pandas"],
    "test": ["pytest"],
}


def _probe(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
    except Exception:
        return "MISSING"
    return getattr(m, "__version__", "ok")


def main() -> int:
    print(f"Python {sys.version.split()[0]}\n")
    width = max(len(m) for mods in STACKS.values() for m in mods) + 2
    missing_core = False
    for label, mods in STACKS.items():
        print(f"[{label}]")
        for mod in mods:
            status = _probe(mod)
            print(f"  {mod:<{width}} {status}")
            if label == "core" and status == "MISSING":
                missing_core = True
        print()

    # Always-available in-repo pieces.
    try:
        importlib.import_module("qco")
        from qco.pipeline import optimize  # noqa: F401
        print("qco package importable ...... ok")
    except Exception as exc:  # pragma: no cover
        print(f"qco package importable ...... FAILED: {exc}")
        return 1

    if missing_core:
        print("\nInstall core deps:  pip install pyyaml")
        return 1
    print("\nCore ready. Install per-module extras as you start Phase 3, e.g.:")
    print("  pip install -e \".[rl]\"   # Maanas Nair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
