"""
Theory-grounded tests for Assembly Calculus operations.

Each test verifies a specific prediction from:
Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (PNAS 2020)

Tests use n=10000, k=100 (k/n=0.01) matching the sparsity regime of the
paper. The sparse engine requires k << n to avoid neuron pool exhaustion.
"""

import copy
import os
import time

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import (
    Assembly,
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
# Overridable so the seed-dependence audit can re-run this file at many seeds
# without editing it (research/experiments/lucky_seed_audit.py). Default 42 --
# normal runs are unaffected. This exists because one test in this file was
# green for its whole life while passing only ~7 times in 12: a fixed seed plus
# a threshold near the effect's mean reports the seed, not the effect.
SEED = int(os.environ.get("ASSEMBLIES_AUDIT_SEED", "42"))
# For effects whose seed-to-seed spread is comparable to the margin being
# asserted, so the assertion is on a mean rather than on one draw. Same purpose
# as test_ac_conformance.SEEDS; the thresholds calibrated against these are
# quoted with the mean and sd that were measured over exactly this tuple.
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)


def _make_brain(**kwargs):
    """Create a Brain with standard test parameters."""
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


def _snap(brain, area_name):
    """Snapshot current assembly with real neuron IDs (not compact indices)."""
    from neural_assemblies.assembly_calculus.ops import _snap as snap_impl
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

    @pytest.mark.parametrize("k, n", [(0, 0), (-1, 10), (11, 10)])
    def test_chance_overlap_rejects_invalid_population_domain(self, k, n):
        with pytest.raises(ValueError, match="0 <= k <= n"):
            chance_overlap(k, n)

    def test_chance_overlap_rejects_boolean_parameters(self):
        with pytest.raises(ValueError, match="integers"):
            chance_overlap(True, 10)

    def test_chance_overlap_accepts_numpy_integer_scalars(self):
        assert chance_overlap(np.int64(10), np.int64(100)) == pytest.approx(0.1)


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
        """Projecting A→B then B→A PARTIALLY restores the original A assembly.

        Theory: reciprocal projection writes a return path, so the source can be
        recovered from the target afterwards.

        RESTORATION IS PARTIAL, AND THE OLD THRESHOLD TESTED A NO-OP. This
        asserted ``> 0.6`` at a single seed and passed for years by measuring
        exactly 1.0000 -- because the B->A weight block was never materialised,
        so every back-projection round hit the engine's zero-drive branch, which
        PRESERVES the target's winners. A never moved, so "recovery" was A
        sitting still. Perfect restoration is the signature of a dead fiber, not
        a working one, which is why 1.0 is now an explicit failure below.

        Calibrated against the reference (``.reference/dmitropolsky-assemblies``
        running ``simulations.fixed_assembly_recip_proj``'s protocol): it
        restores 0.740 at these parameters, and its own header comment documents
        partial restoration ("first B->A gets only 25% ... restore up to 42%" at
        its defaults). We measure mean 0.5687 +/- 0.0394 over 8 seeds, so the
        floor here is mean - ~3 sd. norm_init is pinned off per the repo's rule
        that reference reproductions do so; with it on, recurrence DEGRADES
        restoration instead of improving it (0.53 -> 0.22), increasingly so as
        k/n falls.

        Asserted on a MEAN OVER SEEDS: 0.5687 with sd 0.0394 means a single-seed
        threshold anywhere near the mean reports the seed rather than the effect,
        the hazard this file's header already warns about.
        """
        restored = []
        for seed in SEEDS:
            b = _make_brain(seed=seed, norm_init=False)
            b.add_stimulus("stim", K)
            b.add_area("A", N, K, BETA)
            b.add_area("B", N, K, BETA)

            original_a = project(b, "stim", "A", rounds=ROUNDS)

            # reciprocal_project fixes A itself and runs the target->source edge
            # that writes the return path; that edge is what makes the recovery
            # below possible at all.
            reciprocal_project(b, "A", "B", rounds=ROUNDS)

            b.project({}, {"B": ["A"]})
            for _ in range(ROUNDS - 1):
                b.project({}, {"B": ["A"], "A": ["A"]})
            restored.append(original_a.overlap(_snap(b, "A")))

        mean = sum(restored) / len(restored)
        ch = chance_overlap(K, N)
        assert mean > 0.45, (
            f"reciprocal restoration collapsed: {mean:.4f} over {len(SEEDS)} "
            f"seeds (expected ~0.57, reference 0.740), values {restored}"
        )
        assert mean > ch * 5, (
            f"restoration {mean:.4f} is not meaningfully above chance {ch:.4f}"
        )
        # A DEAD FIBER READS PERFECT. If the return path is never materialised
        # the back-projection preserves A's winners and every seed reports
        # exactly 1.0; that is what this test used to measure.
        assert not all(r == 1.0 for r in restored), (
            "every seed restored EXACTLY 1.0 -- the back-projection is a no-op "
            "and A is being preserved rather than restored; check that the "
            "B->A block is materialised (ASSEMBLIES_STRICT_DRIVE=1)"
        )

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

        ASSERTED ON A MEAN OVER SEEDS, and at a lower multiple of chance than
        the docstring above would suggest, because that is what the effect
        actually supports at these parameters. Measured over 12 seeds:

            init discipline      mean      sd    single-seed pass at 3x chance
            content-addressed   0.0383  0.0191   7/12
            legacy streamed     0.0417  0.0258   7/12

        So the previous single-seed `> 3 * chance` assertion sat essentially AT
        the mean and passed a little over half the time; it had been green only
        because seed 42 happened to land high. It went red when a change to
        synapse initialisation re-rolled that seed (0.06 -> 0.03) -- which
        looked like a regression and was not: both disciplines give the same
        distribution.

        The mean over seeds clears 2x chance by more than three standard
        errors, which is a claim the measurement supports. That the effect is
        only ~3.8x chance with a standard deviation half its size is itself
        worth knowing, and is tracked rather than hidden by a lucky seed.
        """
        overlaps = [self._associate_overlap(s)
                    for s in (42, 7, 123, 2024, 5, 99, 314, 1618)]
        chance = chance_overlap(K, N)
        mean = sum(overlaps) / len(overlaps)
        assert mean > chance * 2, (
            f"Association overlaps {[round(o, 3) for o in overlaps]} have "
            f"mean {mean:.4f}, not clearly above chance {chance:.3f}"
        )

    def _associate_overlap(self, seed: int) -> float:
        """Overlap in C between cueing with A alone and with B alone."""
        b = _make_brain(seed=seed)
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
        return c1.overlap(c2)


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

        recovered, recovery = pattern_complete(
            b, "A", fraction=0.5, rounds=5, seed=42,
            observation_mode="plastic",
        )
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
        _, recovery_hi = pattern_complete(
            b_hi, "A", fraction=0.8, rounds=5, seed=42,
            observation_mode="plastic",
        )

        # Test with less cue
        b_lo = copy.deepcopy(b)
        _, recovery_lo = pattern_complete(
            b_lo, "A", fraction=0.3, rounds=5, seed=42,
            observation_mode="plastic",
        )

        assert recovery_hi >= recovery_lo, (
            f"Expected more cue to give at least as good recovery: "
            f"0.8→{recovery_hi:.3f} vs 0.3→{recovery_lo:.3f}"
        )

    def test_pattern_completion_returns_assembly(self):
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)
        recovered, recovery = pattern_complete(
            b, "A", fraction=0.5, rounds=5, seed=42,
            observation_mode="plastic",
        )

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

        # Overlap should be low — within a few multiples of chance
        assert measured < 0.5, (
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
