"""#41 - the Module A report figures build from the committed result files (needs matplotlib)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGS = ["fig_generalization", "fig_generalization_per_circuit", "fig_ablations", "fig_closed_loop", "fig_backend_policies", "fig_scaling"]


def _script():
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("plot_module_a_figures", ROOT / "scripts" / "plot_module_a_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_figures_are_written(tmp_path):
    mod = _script()
    assert mod.main(["--out", str(tmp_path)]) == 0
    for f in FIGS:
        assert (tmp_path / f"{f}.png").stat().st_size > 5000
        assert (tmp_path / f"{f}.svg").read_text(encoding="utf-8").lstrip().startswith(("<?xml", "<svg"))


def test_scaling_table_parses_from_the_complexity_doc():
    mod = _script()
    rows = mod.parse_scaling((ROOT / "docs" / "module_a_complexity_and_scaling.md").read_text(encoding="utf-8"))
    assert len(rows) >= 4 and all(len(r) >= 9 for r in rows)
    gates = [float(r[1]) for r in rows]
    assert gates == sorted(gates)


def test_committed_figures_exist():
    for f in FIGS:
        assert (ROOT / "docs" / "rl_training" / "figures" / f"{f}.png").exists(), f"run scripts/plot_module_a_figures.py ({f})"
