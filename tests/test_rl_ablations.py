"""#37 - Module A ablation / generalization helpers. Pure Python (no numpy, torch or z3)."""

from __future__ import annotations

import importlib.util
import warnings
import xml.dom.minidom
from pathlib import Path

import pytest

from qco.ir.parser import from_qasm2
from qco.modules.rl_scheduler.ablation import (
    BASELINE, BUCKETS, VARIANTS, bucket_name, evaluate_policy, slug, summarize, table, train_variant, variant_config,
)
from qco.modules.rl_scheduler.config import RLConfig
from qco.modules.rl_scheduler.generalization import BUCKETS as GEN_BUCKETS
from qco.modules.rl_scheduler.generalization import bucket_of, to_markdown, summarize as gen_summarize
from qco.modules.rl_scheduler.plots import SERIES_COLORS, grouped_bars_svg
from qco.modules.rl_scheduler.ppo import load_policy

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
TINY = dict(total_timesteps=200, n_steps=100, epochs=1, minibatch=50, n_train_circuits=2, n_eval_circuits=2,
            circuit_layers=5, eval_every=0, checkpoint_every=0, numeric_backend="python")


def tiny_base() -> RLConfig:
    return RLConfig(**TINY).validate()


class TestVariants:
    def test_every_variant_builds_a_valid_config(self):
        for name in VARIANTS:
            variant_config(tiny_base(), name, 3)

    def test_slugs_are_unique_filesystem_safe(self):
        slugs = [slug(v) for v in VARIANTS]
        assert len(set(slugs)) == len(slugs)
        assert all(s.replace("_", "").isalnum() for s in slugs)

    def test_cr_and_weight_overrides(self):
        base = tiny_base()
        assert variant_config(base, BASELINE, 0).deterministic_reduction is True
        assert variant_config(base, "CR off (train + run)", 0).deterministic_reduction is False
        w = variant_config(base, "depth only", 1).weights
        assert (w.depth, w.gate_count, w.fidelity) == (1.0, 0.0, 0.0) and variant_config(base, "depth only", 1).seed == 1
        assert variant_config(base, "no terminal bonus", 0).weights.terminal == 0.0
        assert base.weights.gate_count == 1.0  # the base config is not mutated

    def test_weight_variants_differ_from_baseline_only_in_weights(self):
        base = variant_config(tiny_base(), BASELINE, 0).to_dict()
        for name, (group, _) in VARIANTS.items():
            if group == "weights":
                d = variant_config(tiny_base(), name, 0).to_dict()
                assert {k for k in d if d[k] != base[k]} == {"weights"}


class TestTrainAndEvaluate:
    def test_train_variant_is_resumable_and_evaluates(self, tmp_path):
        f = train_variant(tiny_base(), BASELINE, 0, tmp_path)
        assert f.exists()
        mtime = f.stat().st_mtime_ns
        assert train_variant(tiny_base(), BASELINE, 0, tmp_path) == f
        assert f.stat().st_mtime_ns == mtime  # not retrained
        policy, payload = load_policy(f)
        ir = from_qasm2("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nrz(0.3) q[0];\nrz(0.4) q[0];\nh q[1];\ncx q[0],q[1];\nh q[2];\n")
        ir.name = "toy"
        res = evaluate_policy(policy, RLConfig.from_dict(payload["config"]), [ir])
        assert set(res["toy"]) == {"depth_red_pct", "gate_red_pct", "reward", "seconds"}
        assert res["toy"]["gate_red_pct"] > 0  # CR merges the two rz
        off = evaluate_policy(policy, RLConfig.from_dict(payload["config"]), [ir], cr_at_test=False)
        assert off["toy"]["gate_red_pct"] == 0.0


class TestSummary:
    @staticmethod
    def fake():
        row = lambda d, g, r: {"depth_red_pct": d, "gate_red_pct": g, "reward": r, "seconds": 0.1}  # noqa: E731
        results = {"A": {0: {"c5": row(10, 5, 0.1), "c40": row(4, 2, 0.05)}, 1: {"c5": row(12, 5, 0.1), "c40": row(6, 2, 0.05)}}}
        return results, {"c5": 5, "c40": 40}

    def test_bucket_means_and_std_over_seeds(self):
        s = summarize(*self.fake())
        assert s["A"]["5-6q"]["depth_red_pct"] == (11.0, 1.0)
        assert s["A"]["40q"]["depth_red_pct"] == (5.0, 1.0)
        assert "8-12q" not in s["A"]  # empty buckets are skipped

    def test_table_renders(self):
        t = table(summarize(*self.fake()), "depth_red_pct", ["A", "missing"])
        assert "11.00 ± 1.00" in t and "n/a" in t

    def test_bucket_names(self):
        assert [bucket_name(n) for n in (5, 6, 7, 12, 13, 20, 21, 40)] == ["5-6q", "5-6q", "8-12q", "8-12q", "16-20q", "16-20q", "40q", "40q"]
        assert len(BUCKETS) == 4


class TestGeneralizationBucket40:
    def test_40_qubit_bucket_exists(self):
        assert bucket_of(40) == "40" and GEN_BUCKETS[-1][0] == "40"
        assert bucket_of(20) == "16-20"

    def test_markdown_title_is_configurable(self):
        run = {m: {"depth_red_pct": 1.0, "gate_red_pct": 1.0, "seconds": 0.0} for m in ("stub", "agent_cr")}
        res = {"circuits": [{"name": "x", "qubits": 40, "gates": 10, "depth": 5, "bucket": "40"}], "runs": {0: {"x": run}}}
        md = to_markdown(res, gen_summarize(res), "info", title="Custom title")
        assert md.startswith("# Custom title") and "40 qubits (n=1)" in md


class TestPlots:
    def test_valid_svg_with_legend_and_errors(self):
        svg = grouped_bars_svg("t", ["g1", "g2"], {"a": [1.0, 2.0], "b": [3.0, -1.0]}, "y", {"a": [0.1, 0.2], "b": [0, 0]})
        xml.dom.minidom.parseString(svg)
        assert svg.count("<rect") >= 1 + 4 + 2 and ">a<" in svg and ">b<" in svg

    def test_writes_file_and_rejects_too_many_series(self, tmp_path):
        p = tmp_path / "x" / "c.svg"
        grouped_bars_svg("t", ["g"], {"a": [1.0]}, "y", path=p)
        assert p.read_text().startswith("<svg")
        with pytest.raises(ValueError):
            grouped_bars_svg("t", ["g"], {str(i): [1.0] for i in range(len(SERIES_COLORS) + 1)}, "y")

    def test_escapes_labels(self):
        assert "&lt;b&gt;" in grouped_bars_svg("<b>", ["g"], {"a": [1.0]}, "y")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGen40qCircuits:
    def test_generator_builds_five_40_qubit_circuits(self):
        mod = _load_script("gen_circuits_40q")
        built = mod.build()
        assert sorted(built) == ["qaoa_n40_p1", "qaoa_n40_p2", "qft_n40", "rand_cliffordt_n40", "rand_cnotpauli_n40"]
        for name, c in built.items():
            ir = from_qasm2(c.text())
            assert ir.num_qubits == 40 and ir.gate_count() > 100, name

    def test_committed_files_match_the_generator(self):
        mod = _load_script("gen_circuits_40q")
        for name, c in mod.build().items():
            f = ROOT / "benchmarks" / "circuits_40q" / f"{name}.qasm"
            assert f.exists() and f.read_text() == c.text(), f"{f} is stale: rerun scripts/gen_circuits_40q.py"
