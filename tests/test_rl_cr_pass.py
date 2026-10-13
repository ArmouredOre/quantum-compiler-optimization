"""Sprint 2 — Module A: the deterministic Commutation-and-Reduction (CR) pass,
the gate-pair algebra, the commutation relation and the DAG-edge flips.

Pure Python (no numpy / gymnasium / torch), so these run in the minimal CI.
Soundness is checked against exact unitaries (``EquivalenceChecker``) on <= 4 qubits.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.ir.parser import from_qasm2
from qco.modules.rl_scheduler.circuits import FAMILIES, sample_circuits
from qco.modules.rl_scheduler.commutation import commutes
from qco.modules.rl_scheduler.cr_pass import is_reduced, reduce_circuit, reduce_with_stats
from qco.modules.rl_scheduler.gate_algebra import NOT_REDUCIBLE, reduce_pair
from qco.modules.rl_scheduler.rewrites import (
    IllegalActionError,
    apply_action,
    commute,
    flippable_edges,
    merge_rotation,
)
from qco.modules.smt_verifier.equivalence import EquivalenceChecker

CIRCUITS = Path(__file__).resolve().parents[1] / "benchmarks" / "circuits"
CHECK = EquivalenceChecker(backend="numeric")


def equivalent(a, b) -> bool:
    return bool(CHECK.check(a, b).equivalent)


def ir(n, *gates):
    c = IntermediateRepresentation(n)
    for name, qs, *p in gates:
        c.add(name, qs, p[0] if p else [])
    return c


# --------------------------------------------------------------------------
# gate algebra
# --------------------------------------------------------------------------

class TestReducePair:
    def test_self_inverse_cancels(self):
        for name in ("h", "x", "y", "z"):
            c = ir(1, (name, [0]), (name, [0]))
            assert reduce_pair(*c.gates) is None
        c = ir(2, ("cx", [0, 1]), ("cx", [0, 1]))
        assert reduce_pair(*c.gates) is None

    def test_cx_with_swapped_operands_is_not_reducible(self):
        c = ir(2, ("cx", [0, 1]), ("cx", [1, 0]))
        assert reduce_pair(*c.gates) is NOT_REDUCIBLE

    def test_symmetric_gate_cancels_either_order(self):
        c = ir(2, ("cz", [0, 1]), ("cz", [1, 0]))
        assert reduce_pair(*c.gates) is None

    def test_s_sdg_and_t_tdg_cancel(self):
        assert reduce_pair(*ir(1, ("s", [0]), ("sdg", [0])).gates) is None
        assert reduce_pair(*ir(1, ("tdg", [0]), ("t", [0])).gates) is None

    def test_phase_squares_merge(self):
        assert reduce_pair(*ir(1, ("t", [0]), ("t", [0])).gates).name == "s"
        assert reduce_pair(*ir(1, ("s", [0]), ("s", [0])).gates).name == "z"

    def test_rotations_sum_and_vanish_at_2pi(self):
        g = reduce_pair(*ir(1, ("rz", [0], [0.3]), ("rz", [0], [0.4])).gates)
        assert g.name == "rz" and g.params[0] == pytest.approx(0.7)
        assert reduce_pair(*ir(1, ("rz", [0], [math.pi]), ("rz", [0], [-math.pi])).gates) is None

    def test_controlled_rotation_period_is_4pi(self):
        # crz(pi)^2 = crz(2 pi) = diag(1,1,-1... ) is NOT the identity (it is Z on the control).
        out = reduce_pair(*ir(2, ("crz", [0, 1], [math.pi]), ("crz", [0, 1], [math.pi])).gates)
        assert out is not None
        assert reduce_pair(*ir(2, ("crz", [0, 1], [2 * math.pi]), ("crz", [0, 1], [2 * math.pi])).gates) is None

    def test_different_gates_not_reducible(self):
        assert reduce_pair(*ir(1, ("h", [0]), ("x", [0])).gates) is NOT_REDUCIBLE


# --------------------------------------------------------------------------
# CR pass
# --------------------------------------------------------------------------

class TestCRPass:
    def test_adjacent_cancel(self):
        out = reduce_circuit(ir(2, ("h", [0]), ("h", [0]), ("cx", [0, 1])))
        assert [g.name for g in out.gates] == ["cx"]

    def test_cancel_across_disjoint_gate(self):
        out = reduce_circuit(ir(3, ("h", [0]), ("x", [2]), ("h", [0])))
        assert [g.name for g in out.gates] == ["x"]

    def test_cancel_across_commuting_gate(self):
        # rz on the CX control commutes with the CX, so the two rz merge through it.
        c = ir(2, ("rz", [0], [0.2]), ("cx", [0, 1]), ("rz", [0], [0.3]))
        out = reduce_circuit(c)
        assert len(out.gates) == 2 and equivalent(c, out)

    def test_t_on_cx_target_blocks(self):
        c = ir(2, ("t", [1]), ("cx", [0, 1]), ("tdg", [1]))
        out = reduce_circuit(c)
        assert len(out.gates) == 3  # t does not commute with the CX target
        assert equivalent(c, out)

    def test_trivial_identity_removed(self):
        out = reduce_circuit(ir(1, ("rz", [0], [0.0]), ("h", [0])))
        assert [g.name for g in out.gates] == ["h"]

    def test_cascade_to_empty(self):
        c = ir(1, ("h", [0]), ("t", [0]), ("tdg", [0]), ("h", [0]))
        assert len(reduce_circuit(c).gates) == 0

    def test_input_not_mutated(self):
        c = ir(1, ("h", [0]), ("h", [0]))
        reduce_circuit(c)
        assert len(c.gates) == 2

    def test_stats_account_for_removed_gates(self):
        c = ir(2, ("h", [0]), ("h", [0]), ("t", [1]), ("t", [1]), ("rz", [0], [0.0]))
        out, stats = reduce_with_stats(c)
        assert stats.gates_removed == len(c.gates) - len(out.gates)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_sound_monotone_idempotent_on_random_circuits(self, family):
        for n in (3, 4):
            for c in sample_circuits(n, 6, seed=11, families=(family,), layers=6):
                out = reduce_circuit(c)
                assert len(out.gates) <= len(c.gates)
                assert equivalent(c, out), f"CR changed semantics of {c.name}"
                assert is_reduced(out)
                assert reduce_circuit(out).gates == out.gates

    def test_window_limits_scan(self):
        c = ir(3, ("h", [0]), ("x", [1]), ("x", [2]), ("h", [0]))
        assert len(reduce_circuit(c, window=1).gates) == 4  # partner is 3 gates away
        assert len(reduce_circuit(c).gates) == 2

    def test_reduces_gate_count_on_random_clifford_t_benchmark(self):
        """Sprint 2 acceptance: CR alone already reduces gate count on Clifford+T."""
        files = sorted(CIRCUITS.glob("rand_cliffordt*.qasm")) or sorted(CIRCUITS.glob("*cliff*.qasm"))
        if not files:
            pytest.skip("no random Clifford+T benchmark files found")
        total_before = total_after = 0
        for f in files:
            c = from_qasm2(f.read_text(encoding="utf-8"))
            out = reduce_circuit(c)
            total_before += len(c.gates)
            total_after += len(out.gates)
            assert len(out.gates) <= len(c.gates)
        assert total_after < total_before

    def test_commuted_reordering_does_not_change_reduced_size(self):
        """Documents the docstring claim: reordering commuting gates exposes no new cancellations."""
        rng = random.Random(5)
        for c in sample_circuits(4, 8, seed=3, families=("clifford_t", "cnot_pauli"), layers=6):
            base = len(reduce_circuit(c).gates)
            cur = c
            for _ in range(10):
                edges = flippable_edges(cur.gates, "full")
                if not edges:
                    break
                i, j = rng.choice(edges)
                cur = commute(cur, i, j, "full")
            assert equivalent(c, cur)
            assert len(reduce_circuit(cur).gates) == base


# --------------------------------------------------------------------------
# commutation relation & rewrites
# --------------------------------------------------------------------------

class TestCommutation:
    def test_disjoint_always_commute(self):
        a, b = ir(2, ("h", [0]), ("x", [1])).gates
        assert commutes(a, b, "disjoint") and commutes(a, b, "full")

    def test_shared_qubit_only_in_full_mode(self):
        a, b = ir(2, ("rz", [0], [0.4]), ("cx", [0, 1])).gates
        assert not commutes(a, b, "disjoint")
        assert commutes(a, b, "full")

    def test_h_does_not_commute_with_cx(self):
        a, b = ir(2, ("h", [0]), ("cx", [0, 1])).gates
        assert not commutes(a, b, "full")

    def test_full_mode_relation_is_sound_for_all_pairs(self):
        """Every pair the relation accepts must really commute as unitaries."""
        names1 = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
        gates = []
        for q in (0, 1):
            for n in names1:
                gates.append((n, [q], [0.37] if n.startswith("r") else []))
        for n in ("cx", "cz", "swap", "rzz", "crz"):
            for qs in ([0, 1], [1, 0]):
                gates.append((n, qs, [0.37] if n in ("rzz", "crz") else []))
        for n1, q1, p1 in gates:
            for n2, q2, p2 in gates:
                a = ir(2, (n1, q1, p1), (n2, q2, p2))
                b = ir(2, (n2, q2, p2), (n1, q1, p1))
                x, y = a.gates
                if commutes(x, y, "full"):
                    assert equivalent(a, b), (n1, q1, n2, q2)


class TestRewritesSprint2:
    def test_non_adjacent_commute_rejected(self):
        c = ir(3, ("h", [0]), ("x", [1]), ("x", [2]))
        with pytest.raises(IllegalActionError):
            commute(c, 0, 2)

    def test_non_adjacent_merge_rejected(self):
        c = ir(2, ("rz", [0], [0.1]), ("x", [1]), ("rz", [0], [0.2]))
        with pytest.raises(IllegalActionError):
            merge_rotation(c, 0, 2)

    def test_full_flip_preserves_semantics(self):
        rng = random.Random(1)
        for c in sample_circuits(4, 10, seed=21, families=("clifford_t", "cnot_pauli", "qaoa"), layers=6):
            edges = flippable_edges(c.gates, "full")
            for i, j in rng.sample(edges, min(5, len(edges))):
                assert equivalent(c, commute(c, i, j, "full"))

    def test_apply_action_threads_commutation_mode(self):
        c = ir(2, ("rz", [0], [0.4]), ("cx", [0, 1]))
        with pytest.raises(IllegalActionError):
            apply_action(c, "COMMUTE", 0, 1, "disjoint")
        out = apply_action(c, "COMMUTE", 0, 1, "full")
        assert [g.name for g in out.gates] == ["cx", "rz"]
