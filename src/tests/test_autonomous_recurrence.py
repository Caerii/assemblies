"""
Tests for autonomous recurrence (Q21): pure self-projection without stimulus.

Verifies that the assembly calculus supports attractor dynamics:
- Single-area: trained assembly is a fixed-point attractor under self-projection
- Multi-area: A->B->A cycling maintains coherent assemblies
- FiberCircuit: autonomous_step() convenience method works correctly

Theory: After Hebbian training strengthens recurrent connections, the
assembly becomes a stable attractor. Self-projection project({}, {A: [A]})
should converge to the same assembly within a few rounds.

Reference:
    Papadimitriou et al. "Brain Computation by Assemblies of Neurons."
    PNAS 117.25 (2020): 14464-14472.
"""

import time

import numpy as np
import pytest

from src.core.brain import Brain
from src.assembly_calculus import (
    Assembly, overlap, chance_overlap, project,
    reciprocal_project, FiberCircuit,
)
from src.assembly_calculus.ops import _snap


N = 10000
K = 100
P = 0.05
BETA = 0.1
ROUNDS = 10
SEED = 42


@pytest.fixture(autouse=True)
def _timer(request):
    t0 = time.perf_counter()
    yield
    print(f"  [{time.perf_counter() - t0:.3f}s]")


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


# ======================================================================
# 1. Single-area attractor dynamics
# ======================================================================

class TestSingleAreaAttractor:
    """Trained assembly acts as fixed-point attractor under autonomous
    recurrence (self-projection without stimulus)."""

    def test_self_projection_preserves_trained_assembly(self):
        """After training, project({}, {A: [A]}) maintains the assembly.

        Protocol: train assembly with stimulus + recurrence for ROUNDS
        steps, remove stimulus, run 10 steps of pure self-projection,
        verify overlap > 0.85 with trained assembly.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm_trained = project(b, "stim", "A", rounds=ROUNDS)

        for _ in range(10):
            b.project({}, {"A": ["A"]})
        asm_autonomous = _snap(b, "A")

        ov = overlap(asm_trained, asm_autonomous)
        assert ov > 0.85, (
            f"Autonomous recurrence should maintain trained assembly, "
            f"overlap={ov:.3f}"
        )

    def test_self_projection_converges_to_fixed_point(self):
        """Consecutive autonomous steps converge: overlap between step t
        and step t+1 approaches 1.0.

        Theory: attractor dynamics produce a fixed point under repeated
        self-projection with Hebbian-strengthened recurrent weights.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)

        overlaps = []
        for _ in range(8):
            snap_before = _snap(b, "A")
            b.project({}, {"A": ["A"]})
            snap_after = _snap(b, "A")
            overlaps.append(overlap(snap_before, snap_after))

        assert overlaps[-1] > 0.9, (
            f"Should converge: final step overlap={overlaps[-1]:.3f}, "
            f"trajectory={[f'{o:.2f}' for o in overlaps]}"
        )

    def test_training_strengthens_attractor(self):
        """More training rounds produce more stable attractors.

        Compares overlap after autonomous steps for 1-round vs 10-round
        training. The well-trained assembly should be more stable.
        """
        results = {}
        for train_rounds in [1, ROUNDS]:
            b = _make_brain()
            b.add_stimulus("stim", K)
            b.add_area("A", N, K, BETA)

            asm_trained = project(b, "stim", "A", rounds=train_rounds)

            # 15 autonomous steps (enough for drift to show)
            for _ in range(15):
                b.project({}, {"A": ["A"]})
            asm_after = _snap(b, "A")

            results[train_rounds] = overlap(asm_trained, asm_after)

        assert results[ROUNDS] >= results[1] - 0.05, (
            f"More training should give better stability: "
            f"1-round={results[1]:.3f}, {ROUNDS}-round={results[ROUNDS]:.3f}"
        )


# ======================================================================
# 2. Multi-area autonomous loops
# ======================================================================

class TestMultiAreaLoop:
    """Multi-area autonomous loops: A -> B -> A cycling."""

    def test_two_area_autonomous_cycle(self):
        """A->B and B->A with no stimulus maintain coherent assemblies.

        Protocol:
        1. Train A from stimulus, project A->B (reciprocal)
        2. Remove stimulus, cycle A->B->A autonomously
        3. Verify assemblies in both areas remain above chance
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)

        # Train assembly in A
        asm_a = project(b, "stim", "A", rounds=ROUNDS)

        # Project A -> B (reciprocal with recurrence)
        asm_b = reciprocal_project(b, "A", "B", rounds=ROUNDS)

        # Autonomous cycling: A->B, B->A simultaneously
        for _ in range(10):
            b.project({}, {"A": ["B"], "B": ["A"]})

        asm_a_after = _snap(b, "A")
        asm_b_after = _snap(b, "B")

        chance = chance_overlap(K, N)
        assert overlap(asm_a, asm_a_after) > chance * 5, (
            f"A should maintain structure: overlap={overlap(asm_a, asm_a_after):.3f}, "
            f"chance={chance:.3f}"
        )
        assert overlap(asm_b, asm_b_after) > chance * 5, (
            f"B should maintain structure: overlap={overlap(asm_b, asm_b_after):.3f}, "
            f"chance={chance:.3f}"
        )

    def test_three_area_ring(self):
        """A->B->C->A ring sustains activity without stimulus.

        Verifies that multi-area autonomous loops maintain
        non-trivial activity across all participating areas.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        b.add_area("C", N, K, BETA)

        # Train A, project chain A->B->C
        project(b, "stim", "A", rounds=ROUNDS)
        reciprocal_project(b, "A", "B", rounds=ROUNDS)
        reciprocal_project(b, "B", "C", rounds=ROUNDS)

        # Autonomous ring
        for _ in range(10):
            b.project({}, {"A": ["B"], "B": ["C"], "C": ["A"]})

        # All areas should still have K winners
        for area_name in ["A", "B", "C"]:
            assert len(b.areas[area_name].winners) == K, (
                f"Area {area_name} should have {K} winners"
            )


# ======================================================================
# 3. FiberCircuit autonomous convenience
# ======================================================================

class TestFiberCircuitAutonomous:
    """FiberCircuit convenience for autonomous (stimulus-free) steps."""

    def test_autonomous_step_via_fiber_circuit(self):
        """FiberCircuit with only area->area fibers (no stim fibers)
        executes autonomous recurrence via step()."""
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm_trained = project(b, "stim", "A", rounds=ROUNDS)

        circuit = FiberCircuit(b)
        circuit.add("A", "A")  # self-recurrence fiber only

        for _ in range(5):
            circuit.step()

        asm_after = _snap(b, "A")
        ov = overlap(asm_trained, asm_after)
        assert ov > 0.85, (
            f"FiberCircuit self-recurrence should maintain assembly: "
            f"overlap={ov:.3f}"
        )

    def test_autonomous_step_method(self):
        """autonomous_step(n) runs n steps using only area-to-area fibers,
        temporarily inhibiting all stimulus fibers."""
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm_trained = project(b, "stim", "A", rounds=ROUNDS)

        circuit = FiberCircuit(b)
        circuit.add("A", "A")
        circuit.add_stim("stim", "A")  # declare stim fiber too

        # autonomous_step should inhibit stim fiber temporarily
        circuit.autonomous_step(5)

        asm_after = _snap(b, "A")
        ov = overlap(asm_trained, asm_after)
        assert ov > 0.85, (
            f"autonomous_step should maintain assembly: overlap={ov:.3f}"
        )
        # Stim fiber should be restored after autonomous_step
        assert circuit.is_active("stim", "A"), (
            "Stim fiber should be restored after autonomous_step"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
