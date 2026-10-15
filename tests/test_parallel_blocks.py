"""#46 — parallel block processing must reproduce the serial run exactly (pure Python)."""

from __future__ import annotations

import multiprocessing
import warnings

import pytest

from qco.graphs.partitioning import partition_circuit, schedule_blocks
from qco.ir.parser import parse
from qco.pipeline import HybridOptimizer, PipelineConfig, resolve_workers

warnings.simplefilter("ignore")
FORK = "fork" in multiprocessing.get_all_start_methods()
CIRCUIT = "benchmarks/circuits/rand_cliffordt_n20.qasm"


def cfg(**kw) -> PipelineConfig:
    base = dict(use_ea=False, gnn_checkpoint=None, enable_resynthesis=False, rl_policy_path=None)
    return PipelineConfig(**{**base, **kw})


def signature(res):
    return [pt.circuit.to_qasm2() for pt in res.pareto_front]


def test_resolve_workers():
    assert resolve_workers(1) == 1
    assert resolve_workers(3) == 3
    assert resolve_workers(0) >= 1
    assert resolve_workers(-4) == 1


def test_serial_default_reports_one_worker():
    res = HybridOptimizer(cfg()).run(parse(CIRCUIT))
    assert res.parallel_workers == 1
    assert res.block_stage_s >= 0.0


@pytest.mark.skipif(not FORK, reason="needs the fork start method")
class TestParallelEqualsSerial:
    def test_identical_circuits_and_counters(self):
        ir = parse(CIRCUIT)
        serial = HybridOptimizer(cfg()).run(ir)
        par = HybridOptimizer(cfg(parallel_blocks=2)).run(ir)
        assert par.parallel_workers == 2
        assert signature(par) == signature(serial)
        assert par.blocks == serial.blocks
        assert par.verified == serial.verified
        assert par.reverts == serial.reverts
        assert par.unverified_blocks == serial.unverified_blocks
        assert par.verification_calls == serial.verification_calls
        assert par.verifications_skipped == serial.verifications_skipped

    def test_optimize_blocks_order_and_variants(self):
        ir = parse(CIRCUIT)
        blocks = schedule_blocks(ir, partition_circuit(ir))
        a = HybridOptimizer(cfg()).optimize_blocks(blocks, workers=1)
        b = HybridOptimizer(cfg()).optimize_blocks(blocks, workers=2)
        assert [[v.to_qasm2() for v in vs] for vs in a] == [[v.to_qasm2() for v in vs] for vs in b]

    def test_exhaustive_resynthesis_matches_on_small_circuit(self):
        ir = parse("benchmarks/circuits/ghz_n20.qasm")
        kw = dict(enable_resynthesis=True, window_ranking="exhaustive", max_block_gates=4)
        s = HybridOptimizer(cfg(**kw)).run(ir)
        p = HybridOptimizer(cfg(parallel_blocks=2, **kw)).run(ir)
        assert signature(p) == signature(s)

    def test_one_block_stays_serial(self):
        ir = parse("benchmarks/circuits/ghz_n20.qasm")
        res = HybridOptimizer(cfg(parallel_blocks=2, max_block_gates=500, partition_backend="fallback")).run(ir)
        assert res.blocks == 1 and res.parallel_workers == 1


def test_no_fork_falls_back_with_warning(monkeypatch):
    real = multiprocessing.get_context

    def fake(method=None):
        if method == "fork":
            raise ValueError("no fork")
        return real(method)

    monkeypatch.setattr(multiprocessing, "get_context", fake)
    ir = parse(CIRCUIT)
    serial = HybridOptimizer(cfg()).run(ir)
    with pytest.warns(RuntimeWarning, match="fork"):
        par = HybridOptimizer(cfg(parallel_blocks=2)).run(ir)
    assert par.parallel_workers == 1
    assert signature(par) == signature(serial)
