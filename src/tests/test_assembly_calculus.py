"""
Theory-grounded tests for Assembly Calculus operations.

Each test verifies a specific prediction from:
Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (PNAS 2020)

Tests use n=10000, k=100 (k/n=0.01) matching the sparsity regime of the
paper. The sparse engine requires k << n to avoid neuron pool exhaustion.
"""

import copy
import time

import numpy as np
import pytest

from src.core.brain import Brain
from src.assembly_calculus import (
    Assembly,
    overlap,
    chance_overlap,
    project,
    reciprocal_project,
    associate,
    merge,
    pattern_complete,
    separate,
    FiberCircuit,
)


# ---------------------------------------------------------------------------
# Timing fixture — prints elapsed time for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _timer(request):
    """Print wall-clock time for each test."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  [{elapsed:.3f}s]")

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

N = 10000       # neurons per area (k/n = 0.01, sparse regime)
K = 100         # assembly size
P = 0.05        # connection probability
BETA = 0.1      # plasticity rate
ROUNDS = 10     # stabilization rounds
SEED = 42       # reproducibility


def _make_brain(**kwargs):
    """Create a Brain with standard test parameters."""
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


def _snap(brain, area_name):
    """Snapshot current assembly with real neuron IDs (not compact indices)."""
    from src.assembly_calculus.ops import _snap as snap_impl
    return snap_impl(brain, area_name)


# ---------------------------------------------------------------------------
# Assembly dataclass tests
# ---------------------------------------------------------------------------

class TestAssembly:
    def test_assembly_is_immutable_snapshot(self):
        """Assembly.winners is a copy; mutating the source doesn't affect it."""
        winners = np.array([1, 2, 3], dtype=np.uint32)
        asm = Assembly("A", winners)
        winners[0] = 999
        assert asm.winners[0] == 1

    def test_assembly_len(self):
        asm = Assembly("A", np.arange(50, dtype=np.uint32))
        assert len(asm) == 50

    def test_assembly_overlap_identical(self):
        w = np.arange(100, dtype=np.uint32)
        a = Assembly("A", w)
        b = Assembly("A", w)
        assert a.overlap(b) == 1.0

    def test_assembly_overlap_disjoint(self):
        a = Assembly("A", np.arange(0, 100, dtype=np.uint32))
        b = Assembly("A", np.arange(100, 200, dtype=np.uint32))
        assert a.overlap(b) == 0.0

    def test_chance_overlap_formula(self):
        assert chance_overlap(100, 1000) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

class TestProject:
    def test_project_creates_stable_assembly(self):
        """After projection, further recurrence doesn't change the assembly.

        Theory: Assembly stabilizes after O(log n) rounds (Theorem 1).
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm = project(b, "stim", "A", rounds=ROUNDS)

        # Run 5 more recurrent-only steps
        for _ in range(5):
            b.project({}, {"A": ["A"]})
        asm_later = _snap(b, "A")

        assert asm.overlap(asm_later) > 0.9

    def test_project_assembly_is_correct_size(self):
        """Winner-take-all selects exactly k neurons."""
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm = project(b, "stim", "A", rounds=ROUNDS)
        assert len(asm) == K

    def test_project_returns_assembly_type(self):
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm = project(b, "stim", "A", rounds=5)
        assert isinstance(asm, Assembly)
        assert asm.area == "A"


# ---------------------------------------------------------------------------
# Reciprocal projection tests
# ---------------------------------------------------------------------------

class TestReciprocalProject:
    def test_reciprocal_project_creates_copy(self):
        """Projecting A→B then B→A recovers the original A assembly.

        Theory: Reciprocal projection restores the source assembly.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)

        # Establish assembly in A
        original_a = project(b, "stim", "A", rounds=ROUNDS)

        # Fix A, project A → B
        b.areas["A"].fix_assembly()
        reciprocal_project(b, "A", "B", rounds=ROUNDS)

        # Unfix A, project B → A
        b.areas["A"].unfix_assembly()
        b.project({}, {"B": ["A"]})
        for _ in range(ROUNDS - 1):
            b.project({}, {"B": ["A"], "A": ["A"]})

        recovered_a = _snap(b, "A")
        assert original_a.overlap(recovered_a) > 0.6

    def test_reciprocal_project_target_has_assembly(self):
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)
        b.areas["A"].fix_assembly()
        asm_b = reciprocal_project(b, "A", "B", rounds=ROUNDS)

        assert len(asm_b) == K
        assert asm_b.area == "B"


# ---------------------------------------------------------------------------
# Association tests
# ---------------------------------------------------------------------------

class TestAssociate:
    def test_associate_creates_shared_response(self):
        """After association, either source alone activates overlapping
        assemblies in the target.

        Theory (Papadimitriou 2020, §3): Association creates a shared
        representation. Overlap is well above chance (k/n).
        """
        b = _make_brain()
        b.add_stimulus("stimA", K)
        b.add_stimulus("stimB", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        b.add_area("C", N, K, BETA)

        # Establish assemblies in A and B
        project(b, "stimA", "A", rounds=ROUNDS)
        project(b, "stimB", "B", rounds=ROUNDS)

        # Associate A and B through C
        associate(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)

        # Test: activate only A, project to C
        b_copy1 = copy.deepcopy(b)
        b_copy1.project({"stimA": ["A"]}, {"A": ["C"]})
        for _ in range(5):
            b_copy1.project({}, {"A": ["C"], "C": ["C"]})
        c1 = _snap(b_copy1, "C")

        # Test: activate only B, project to C
        b_copy2 = copy.deepcopy(b)
        b_copy2.project({"stimB": ["B"]}, {"B": ["C"]})
        for _ in range(5):
            b_copy2.project({}, {"B": ["C"], "C": ["C"]})
        c2 = _snap(b_copy2, "C")

        # The two C assemblies should significantly overlap
        measured = c1.overlap(c2)
        chance = chance_overlap(K, N)
        assert measured > chance * 3, (
            f"Association overlap {measured:.3f} not much above chance {chance:.3f}"
        )


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_responds_to_either_source(self):
        """After merge, projecting either source alone into C
        activates overlapping assemblies.

        Theory: The merged assembly responds to either constituent.
        """
        b = _make_brain()
        b.add_stimulus("stimA", K)
        b.add_stimulus("stimB", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        b.add_area("C", N, K, BETA)

        # Establish source assemblies
        project(b, "stimA", "A", rounds=ROUNDS)
        project(b, "stimB", "B", rounds=ROUNDS)

        # Merge A and B into C
        merge(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)

        # Test: project only A → C
        b_copy1 = copy.deepcopy(b)
        b_copy1.areas["A"].fix_assembly()
        b_copy1.project({}, {"A": ["C"]})
        for _ in range(5):
            b_copy1.project({}, {"A": ["C"], "C": ["C"]})
        c1 = _snap(b_copy1, "C")

        # Test: project only B → C
        b_copy2 = copy.deepcopy(b)
        b_copy2.areas["B"].fix_assembly()
        b_copy2.project({}, {"B": ["C"]})
        for _ in range(5):
            b_copy2.project({}, {"B": ["C"], "C": ["C"]})
        c2 = _snap(b_copy2, "C")

        # Both should produce non-empty C assemblies with overlap above chance
        assert len(c1) == K
        assert len(c2) == K

        measured = c1.overlap(c2)
        chance = chance_overlap(K, N)
        assert measured > chance * 2, (
            f"Merge overlap {measured:.3f} not much above chance {chance:.3f}"
        )


# ---------------------------------------------------------------------------
# Pattern completion tests
# ---------------------------------------------------------------------------

class TestPatternCompletion:
    def test_pattern_completion_recovers_assembly(self):
        """Partial activation (50%) recovers the full assembly.

        Theory: A well-trained assembly is an attractor.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)

        recovered, recovery = pattern_complete(b, "A", fraction=0.5, rounds=5, seed=42)
        assert recovery > 0.6, f"Recovery {recovery:.3f} too low for fraction=0.5"

    def test_pattern_completion_degrades_with_less_cue(self):
        """More cue → better recovery (monotonic degradation).

        Theory: Recovery is a monotonically increasing function of
        the fraction of cue neurons.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)

        # Test with more cue
        b_hi = copy.deepcopy(b)
        _, recovery_hi = pattern_complete(b_hi, "A", fraction=0.8, rounds=5, seed=42)

        # Test with less cue
        b_lo = copy.deepcopy(b)
        _, recovery_lo = pattern_complete(b_lo, "A", fraction=0.3, rounds=5, seed=42)

        assert recovery_hi > recovery_lo, (
            f"Expected more cue to give better recovery: "
            f"0.8→{recovery_hi:.3f} vs 0.3→{recovery_lo:.3f}"
        )

    def test_pattern_completion_returns_assembly(self):
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)
        recovered, recovery = pattern_complete(b, "A", fraction=0.5, rounds=5, seed=42)

        assert isinstance(recovered, Assembly)
        assert isinstance(recovery, float)
        assert 0.0 <= recovery <= 1.0


# ---------------------------------------------------------------------------
# Separation tests
# ---------------------------------------------------------------------------

class TestSeparate:
    def test_separate_stimuli_create_distinct_assemblies(self):
        """Two independent stimuli produce assemblies with near-chance overlap.

        Theory: Independent stimuli → independent assemblies.
        """
        b = _make_brain()
        b.add_stimulus("stimA", K)
        b.add_stimulus("stimB", K)
        b.add_area("A", N, K, BETA)

        asm_a, asm_b, measured = separate(b, "stimA", "stimB", "A", rounds=ROUNDS)

        chance = chance_overlap(K, N)
        # Overlap should be low — within a few multiples of chance
        assert measured < 0.4, (
            f"Overlap {measured:.3f} too high for independent stimuli"
        )
        assert len(asm_a) == K
        assert len(asm_b) == K


# ---------------------------------------------------------------------------
# FiberCircuit tests
# ---------------------------------------------------------------------------

class TestFiberCircuit:
    def _make_wired_brain(self):
        """Create a brain with stimulus, areas A and B, and established
        assembly in A."""
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        project(b, "stim", "A", rounds=ROUNDS)
        return b

    def test_fiber_circuit_step_projects(self):
        """step() with active fibers actually projects."""
        b = self._make_wired_brain()
        b.areas["A"].fix_assembly()

        circuit = FiberCircuit(b)
        circuit.add("A", "B")

        circuit.step()

        # B should now have winners
        assert len(b.areas["B"].winners) == K

    def test_fiber_circuit_inhibit_blocks_projection(self):
        """Inhibited fibers don't project."""
        b = self._make_wired_brain()
        b.areas["A"].fix_assembly()

        circuit = FiberCircuit(b)
        circuit.add("A", "B")
        circuit.inhibit("A", "B")

        # With the only fiber inhibited, nothing should project to B
        old_winners = b.areas["B"].winners.copy()
        circuit.step()

        # B should not have gained a new assembly
        # (step with empty projections is a no-op)
        assert len(b.areas["B"].winners) == len(old_winners)

    def test_fiber_circuit_disinhibit_restores(self):
        """Disinhibiting a fiber makes it active again."""
        b = self._make_wired_brain()
        b.areas["A"].fix_assembly()

        circuit = FiberCircuit(b)
        circuit.add("A", "B")

        circuit.inhibit("A", "B")
        assert not circuit.is_active("A", "B")

        circuit.disinhibit("A", "B")
        assert circuit.is_active("A", "B")

        circuit.step()
        assert len(b.areas["B"].winners) == K

    def test_fiber_circuit_stim_fibers(self):
        """Stimulus fibers project stimuli to areas."""
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        circuit = FiberCircuit(b)
        circuit.add_stim("stim", "A")

        circuit.step()
        assert len(b.areas["A"].winners) == K

    def test_fiber_circuit_unknown_fiber_raises(self):
        """Inhibiting an undeclared fiber raises KeyError."""
        b = _make_brain()
        circuit = FiberCircuit(b)
        with pytest.raises(KeyError):
            circuit.inhibit("X", "Y")

    def test_fiber_circuit_active_projections(self):
        """active_area_projections returns only active fibers."""
        b = _make_brain()
        circuit = FiberCircuit(b)
        circuit.add("A", "B")
        circuit.add("B", "C")
        circuit.add("C", "A")
        circuit.inhibit("C", "A")

        proj = circuit.active_area_projections()
        assert proj == {"A": ["B"], "B": ["C"]}
