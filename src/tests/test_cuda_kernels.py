"""
Tests for CUDA kernel optimizations: hash table Hebbian + shared memory active set.

All tests require a GPU with CuPy installed and are skipped otherwise.
"""

import numpy as np
import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU not available")


def _make_engine(n=10_000, k=50, p=0.05, beta=0.05, seed=42, w_max=20.0):
    """Create a CudaImplicitEngine with standard test parameters."""
    from src.core.engine import create_engine
    engine = create_engine(
        "cuda_implicit", p=p, seed=seed, w_max=w_max,
    )
    engine.add_area("A", n=n, k=k, beta=beta)
    engine.add_stimulus("s", size=k)
    engine.add_connectivity("s", "A", p=p)
    return engine


class TestHashTableCorrectness:
    """Verify the hash-table Hebbian kernel produces correct, deterministic results."""

    def test_learned_connections_created(self):
        """After repeated projection, learned connections should accumulate."""
        engine = _make_engine()
        for _ in range(20):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])

        # Sync GPU to ensure Hebbian kernel writes are visible
        cp.cuda.Device().synchronize()
        coo = engine._pair_learned[("s", "A")]
        num_learned = int(coo.num_learned[0])
        assert num_learned > 0, "No learned connections created after 20 rounds"

    def test_determinism_across_engines(self):
        """Two engines with same seed produce identical winners."""
        rounds = 20

        e1 = _make_engine(seed=123)
        for _ in range(rounds):
            e1.project_into("A", from_stimuli=["s"], from_areas=[])
        w1 = np.sort(e1.get_winners("A"))

        e2 = _make_engine(seed=123)
        for _ in range(rounds):
            e2.project_into("A", from_stimuli=["s"], from_areas=[])
        w2 = np.sort(e2.get_winners("A"))

        np.testing.assert_array_equal(w1, w2)

    def test_hash_table_populated(self):
        """Hash table should have non-empty slots after learning."""
        engine = _make_engine()
        for _ in range(10):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])

        cp.cuda.Device().synchronize()
        coo = engine._pair_learned[("s", "A")]
        ht = cp.asnumpy(coo.hash_table)
        num_filled = np.count_nonzero(ht != 0xFFFFFFFF)
        num_learned = int(coo.num_learned[0])
        # Every learned connection should have a hash table entry
        assert num_filled >= num_learned, (
            f"Hash table has {num_filled} entries but {num_learned} learned connections"
        )


class TestWeightAccumulation:
    """Verify that weights accumulate and saturate near w_max."""

    def test_weights_saturate(self):
        """After many rounds, learned deltas should approach w_max - 1."""
        w_max = 10.0
        engine = _make_engine(n=5000, k=50, beta=0.1, w_max=w_max, seed=7)
        for _ in range(50):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])

        cp.cuda.Device().synchronize()
        coo = engine._pair_learned[("s", "A")]
        num = int(coo.num_learned[0])
        assert num > 0

        deltas = cp.asnumpy(coo.learned_delta[:num])
        max_delta = deltas.max()
        # With base weight 1.0, delta should approach w_max - 1 = 9.0
        # After 50 rounds it won't be exact, but should be > 1.0
        assert max_delta > 1.0, f"Max delta {max_delta} too low after 50 rounds"
        # Should not exceed w_max - 1
        assert max_delta <= w_max, f"Max delta {max_delta} exceeds w_max {w_max}"

    def test_no_overflow(self):
        """num_learned should not exceed max_learned."""
        engine = _make_engine(n=5000, k=50, seed=99)
        for _ in range(50):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])

        cp.cuda.Device().synchronize()
        coo = engine._pair_learned[("s", "A")]
        num = int(coo.num_learned[0])
        assert num <= coo.max_learned, (
            f"num_learned={num} exceeds max_learned={coo.max_learned}"
        )


class TestCOOWeightsReset:
    """Verify that _COOWeights.reset() restores a clean state."""

    def test_reset_clears_state(self):
        """After reset(), running again produces same results as a fresh engine."""
        engine = _make_engine(seed=55)
        # Run 15 rounds
        for _ in range(15):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])

        cp.cuda.Device().synchronize()
        coo = engine._pair_learned[("s", "A")]
        assert int(coo.num_learned[0]) > 0

        # Reset
        coo.reset()
        cp.cuda.Device().synchronize()
        assert int(coo.num_learned[0]) == 0
        ht = cp.asnumpy(coo.hash_table)
        assert np.all(ht == 0xFFFFFFFF), "Hash table not fully cleared after reset"

    def test_reset_then_rerun_matches_fresh(self):
        """After reset + clearing area state, results match a fresh engine."""
        seed = 77
        rounds = 10

        # Run engine, reset, clear area winners, run again
        e1 = _make_engine(seed=seed)
        for _ in range(rounds):
            e1.project_into("A", from_stimuli=["s"], from_areas=[])

        # Reset all learned pairs and area state
        for coo in e1._pair_learned.values():
            coo.reset()
        e1._areas["A"].winners = None
        e1._areas["A"].prev_winners = None

        for _ in range(rounds):
            e1.project_into("A", from_stimuli=["s"], from_areas=[])
        w1 = np.sort(e1.get_winners("A"))

        # Fresh engine with same seed
        e2 = _make_engine(seed=seed)
        for _ in range(rounds):
            e2.project_into("A", from_stimuli=["s"], from_areas=[])
        w2 = np.sort(e2.get_winners("A"))

        np.testing.assert_array_equal(w1, w2)


class TestMultiAreaProjection:
    """Verify kernel optimizations work with multi-area projections."""

    def test_area_to_area(self):
        """Stimulus -> A, then A -> B should create learned connections in both pairs."""
        from src.core.engine import create_engine
        engine = create_engine("cuda_implicit", p=0.05, seed=42, w_max=20.0)
        engine.add_area("A", n=5000, k=50, beta=0.05)
        engine.add_area("B", n=5000, k=50, beta=0.05)
        engine.add_stimulus("s", size=50)

        for _ in range(10):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])
        for _ in range(10):
            engine.project_into("B", from_stimuli=[], from_areas=["A"])

        cp.cuda.Device().synchronize()
        coo_sa = engine._pair_learned[("s", "A")]
        coo_ab = engine._pair_learned[("A", "B")]
        assert int(coo_sa.num_learned[0]) > 0
        assert int(coo_ab.num_learned[0]) > 0


class TestProjectRounds:
    """Verify engine.project_rounds() matches sequential project_into() calls."""

    def _make_two_engines(self, seed=42, n=5000, k=50):
        """Create two identical engines for comparison."""
        from src.core.engine import create_engine
        engines = []
        for _ in range(2):
            e = create_engine("cuda_implicit", p=0.05, seed=seed, w_max=20.0)
            e.add_area("A", n=n, k=k, beta=0.05)
            e.add_stimulus("s", size=k)
            engines.append(e)
        return engines

    def test_stim_only_matches_sequential(self):
        """project_rounds with stimulus matches sequential project_into calls.

        Note: project_rounds handles target self-recurrence internally
        (round_idx > 0), so from_areas should NOT include the target.
        The sequential path uses from_areas=["A"] to get self-recurrence,
        but project_rounds adds it automatically.
        """
        e_seq, e_fast = self._make_two_engines(seed=200)

        # Round 0: stim → A (no recurrence yet)
        e_seq.project_into("A", from_stimuli=["s"], from_areas=[])
        e_fast.project_into("A", from_stimuli=["s"], from_areas=[])

        # Rounds 1-9: stim + self-recurrence
        rounds = 9
        for _ in range(rounds):
            e_seq.project_into("A", from_stimuli=["s"], from_areas=["A"])

        # project_rounds handles self-recurrence (A→A) internally;
        # from_areas=[] because there are no OTHER source areas.
        e_fast.project_rounds(
            target="A", from_stimuli=["s"], from_areas=[],
            rounds=rounds, plasticity_enabled=True,
        )

        w_seq = np.sort(e_seq.get_winners("A"))
        w_fast = np.sort(e_fast.get_winners("A"))
        np.testing.assert_array_equal(w_seq, w_fast)

    def test_area_source_matches_sequential(self):
        """project_rounds with area source + recurrence matches sequential."""
        from src.core.engine import create_engine
        seed = 300

        # Create two identical engines with 2 areas each
        engines = []
        for _ in range(2):
            e = create_engine("cuda_implicit", p=0.05, seed=seed, w_max=20.0)
            e.add_area("A", n=5000, k=50, beta=0.05)
            e.add_area("B", n=5000, k=50, beta=0.05)
            e.add_stimulus("s", size=50)
            engines.append(e)
        e_seq, e_fast = engines

        # First, establish assembly in A
        for _ in range(5):
            e_seq.project_into("A", from_stimuli=["s"], from_areas=[])
            e_fast.project_into("A", from_stimuli=["s"], from_areas=[])

        # Round 0: A → B
        e_seq.project_into("B", from_stimuli=[], from_areas=["A"])
        e_fast.project_into("B", from_stimuli=[], from_areas=["A"])

        # Rounds 1-4: A + B→B recurrence
        rounds = 4
        for _ in range(rounds):
            e_seq.project_into("B", from_stimuli=[], from_areas=["A", "B"])

        # project_rounds handles B→B self-recurrence internally;
        # from_areas=["A"] includes only external source areas.
        e_fast.project_rounds(
            target="B", from_stimuli=[], from_areas=["A"],
            rounds=rounds, plasticity_enabled=True,
        )

        w_seq = np.sort(e_seq.get_winners("B"))
        w_fast = np.sort(e_fast.get_winners("B"))
        np.testing.assert_array_equal(w_seq, w_fast)

    def test_plasticity_off(self):
        """project_rounds with plasticity_enabled=False produces same winners."""
        e_seq, e_fast = self._make_two_engines(seed=400)

        e_seq.project_into("A", from_stimuli=["s"], from_areas=[])
        e_fast.project_into("A", from_stimuli=["s"], from_areas=[])

        rounds = 5
        for _ in range(rounds):
            e_seq.project_into("A", from_stimuli=["s"], from_areas=["A"],
                               plasticity_enabled=False)

        e_fast.project_rounds(
            target="A", from_stimuli=["s"], from_areas=["A"],
            rounds=rounds, plasticity_enabled=False,
        )

        w_seq = np.sort(e_seq.get_winners("A"))
        w_fast = np.sort(e_fast.get_winners("A"))
        np.testing.assert_array_equal(w_seq, w_fast)

    def test_single_round(self):
        """project_rounds with rounds=1 matches a single project_into."""
        e_seq, e_fast = self._make_two_engines(seed=500)

        # First, project once to establish winners
        e_seq.project_into("A", from_stimuli=["s"], from_areas=[])
        e_fast.project_into("A", from_stimuli=["s"], from_areas=[])

        # One more round via each path
        e_seq.project_into("A", from_stimuli=["s"], from_areas=["A"])
        e_fast.project_rounds(
            target="A", from_stimuli=["s"], from_areas=["A"],
            rounds=1, plasticity_enabled=True,
        )

        w_seq = np.sort(e_seq.get_winners("A"))
        w_fast = np.sort(e_fast.get_winners("A"))
        np.testing.assert_array_equal(w_seq, w_fast)


class TestOpsFastPath:
    """Verify ops.py operations work correctly via Brain.project_rounds()."""

    def _make_brain(self, seed=42):
        """Create a CUDA brain for ops testing."""
        from src.core.brain import Brain
        brain = Brain(p=0.05, seed=seed, w_max=20.0, engine="cuda_implicit")
        brain.add_area("A", n=5000, k=50, beta=0.05)
        brain.add_area("B", n=5000, k=50, beta=0.05)
        brain.add_area("C", n=5000, k=50, beta=0.05)
        brain.add_stimulus("s1", size=50)
        brain.add_stimulus("s2", size=50)
        return brain

    def test_project_converges(self):
        """ops.project produces a stable assembly with significant overlap."""
        from src.assembly_calculus.ops import project
        from src.assembly_calculus.assembly import overlap

        brain = self._make_brain(seed=600)
        asm = project(brain, "s1", "A", rounds=10)
        assert len(asm) == 50

        # Project again — should produce assembly well above chance (k/n=0.01).
        # Exact overlap depends on topk tie-breaking which varies between
        # CuPy and PyTorch backends, so we use a moderate threshold.
        asm2 = project(brain, "s1", "A", rounds=10)
        ov = overlap(asm, asm2)
        assert ov > 0.3, f"Re-projection overlap too low: {ov}"

    def test_reciprocal_project_creates_copy(self):
        """ops.reciprocal_project creates an assembly in the target area."""
        from src.assembly_calculus.ops import project, reciprocal_project

        brain = self._make_brain(seed=700)
        project(brain, "s1", "A", rounds=10)
        asm_b = reciprocal_project(brain, "A", "B", rounds=10)
        assert len(asm_b) == 50

    def test_associate_with_fix(self):
        """ops.associate with fixed sources (no stims) uses fast path."""
        from src.assembly_calculus.ops import project, associate
        from src.assembly_calculus.assembly import overlap

        brain = self._make_brain(seed=800)
        project(brain, "s1", "A", rounds=10)
        project(brain, "s2", "B", rounds=10)

        asm_c = associate(brain, "A", "B", "C", rounds=10)
        assert len(asm_c) == 50
