"""Opt-in ``HardwareProfile`` cost model (topology, SWAP routing, native-gate cost,
parallelism cap) and its use by Module A. Pure Python."""

from __future__ import annotations

import pytest

from qco.evaluation.hardware import (
    HARDWARE_PROFILES,
    HardwareProfile,
    _adjacency,
    coupling_edges,
    get_hardware,
    hardware_execution_time,
    hardware_fidelity,
    list_hardware,
    route,
    swap_count,
)
from qco.evaluation.metrics import estimated_fidelity, execution_time
from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.rl_scheduler import RLConfig
from qco.modules.rl_scheduler.backend_matrix import backend_config, retarget, time_of
from qco.modules.rl_scheduler.circuits import sample_circuits
from qco.modules.rl_scheduler.environment import RewardWeights, SchedulerEnv


def circ(n, *gates):
    c = IntermediateRepresentation(n)
    for name, qs, *p in gates:
        c.add(name, qs, p[0] if p else [])
    return c


class TestTopology:
    @pytest.mark.parametrize("topo,n,edges", [("all_to_all", 4, 6), ("line", 4, 3), ("ring", 4, 4), ("grid", 4, 4), ("grid", 6, 7)])
    def test_edge_counts(self, topo, n, edges):
        assert len(coupling_edges(HardwareProfile("t", topology=topo), n)) == edges

    def test_adjacency_is_symmetric(self):
        for topo in ("line", "ring", "grid"):
            adj = _adjacency(topo, 7)
            assert all(a in adj[b] for a in range(7) for b in adj[a])

    def test_validation(self):
        with pytest.raises(ValueError):
            HardwareProfile("x", topology="torus")
        with pytest.raises(ValueError):
            HardwareProfile("x", max_parallel_2q=0)
        with pytest.raises(KeyError):
            HardwareProfile("x", calibration="nope")
        with pytest.raises(KeyError):
            get_hardware("nope")

    def test_builtin_profiles_listed(self):
        assert set(list_hardware()) == set(HARDWARE_PROFILES) and len(HARDWARE_PROFILES) >= 3


class TestRouting:
    def test_all_to_all_never_swaps(self):
        c = circ(5, ("cx", [0, 4]), ("cx", [1, 3]))
        assert swap_count(c, "ion_all_to_all_like") == 0

    def test_line_distance_costs_swaps(self):
        assert swap_count(circ(4, ("cx", [0, 3])), "line_superconducting_like") == 2
        assert swap_count(circ(4, ("cx", [0, 1])), "line_superconducting_like") == 0

    def test_swap_cost_depends_on_gate_order(self):
        a = circ(4, ("cx", [0, 3]), ("cx", [0, 1]))
        b = circ(4, ("cx", [0, 1]), ("cx", [0, 3]))
        assert swap_count(a, "ibm_grid_like") != swap_count(b, "ibm_grid_like")

    def test_routed_two_qubit_gates_are_adjacent(self):
        for c in sample_circuits(6, 6, 3, layers=5):
            for prof in ("ibm_grid_like", "line_superconducting_like"):
                hw = get_hardware(prof)
                adj = _adjacency(hw.topology, c.num_qubits)
                ops = route(c, hw)
                assert sum(1 for k, *_ in ops if k == "gate") == len(c.gates)
                for kind, _, qs in ops:
                    if len(qs) == 2:
                        assert qs[1] in adj[qs[0]]


class TestMetrics:
    def test_matches_existing_metrics_when_unstructured(self):
        plain = HardwareProfile("plain", calibration="default", topology="all_to_all")
        for c in sample_circuits(4, 4, 1, layers=4):
            assert hardware_execution_time(c, plain) == pytest.approx(execution_time(c, "default").value)
            assert hardware_fidelity(c, plain) == pytest.approx(estimated_fidelity(c, "default").value)

    def test_parallel_cap_serializes_entangling_gates(self):
        c = circ(4, ("cx", [0, 1]), ("cx", [2, 3]))
        free = HardwareProfile("f", calibration="default")
        capped = HardwareProfile("c", calibration="default", max_parallel_2q=1)
        assert hardware_execution_time(c, capped) == pytest.approx(2 * hardware_execution_time(c, free))

    def test_native_gate_factor_changes_time(self):
        c = circ(2, ("rzz", [0, 1], [0.3]))
        base = HardwareProfile("b", calibration="default")
        dear = HardwareProfile("d", calibration="default", gate_time_factor={"rzz": 2.0})
        assert hardware_execution_time(c, dear) == pytest.approx(2 * hardware_execution_time(c, base))

    def test_swaps_lower_fidelity_and_raise_time(self):
        c = circ(4, ("cx", [0, 3]))
        near = circ(4, ("cx", [0, 1]))
        p = "line_superconducting_like"
        assert hardware_fidelity(c, p) < hardware_fidelity(near, p) <= 1.0
        assert hardware_execution_time(c, p) > hardware_execution_time(near, p)


class TestModuleAIntegration:
    def test_env_default_unchanged(self):
        env = SchedulerEnv(circ(3, ("cx", [0, 2])), calibration="trapped_ion_like")
        assert env.hardware is None
        assert env.time_of(env.circuit) == pytest.approx(execution_time(env.circuit, "trapped_ion_like").value)

    def test_env_uses_hardware_costs(self):
        c = circ(4, ("cx", [0, 3]))
        env = SchedulerEnv(c, RewardWeights(time=1.0), hardware="line_superconducting_like")
        assert env.calibration == "superconducting_ibm_like"
        assert env.time_of(c) == pytest.approx(hardware_execution_time(c, "line_superconducting_like"))
        assert env.fidelity_of(c) == pytest.approx(hardware_fidelity(c, "line_superconducting_like"))

    def test_config_hardware_validated_and_round_trips(self):
        with pytest.raises(KeyError):
            RLConfig(hardware="nope").validate()
        c = RLConfig(hardware="ibm_grid_like")
        assert RLConfig.from_dict(c.to_dict()) == c

    def test_backend_config_and_retarget_accept_hardware_names(self):
        c = backend_config(RLConfig(), "ion_all_to_all_like", seed=2)
        assert c.hardware == "ion_all_to_all_like" and c.calibration == "trapped_ion_like"
        back = retarget(c, "default")
        assert back.hardware is None and back.calibration == "default"
        x = circ(4, ("cx", [0, 3]))
        assert time_of(x, "line_superconducting_like") == pytest.approx(hardware_execution_time(x, "line_superconducting_like"))
        assert time_of(x, "default") == pytest.approx(execution_time(x, "default").value)
