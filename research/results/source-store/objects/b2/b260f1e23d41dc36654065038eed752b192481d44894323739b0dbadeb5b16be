"""
Tests for CUDA kernel optimizations: hash table Hebbian + shared memory active set.

All tests require a GPU with CuPy installed and are skipped otherwise.
"""

import numpy as np
import pytest

from neural_assemblies.core.backend import cupy_available

pytestmark = pytest.mark.skipif(
    not cupy_available(), reason="CuPy/GPU not available"
)


def _make_engine(n=10_000, k=50, p=0.05, beta=0.05, seed=42, w_max=20.0):
    """Create a CudaImplicitEngine with standard test parameters."""
    from neural_assemblies.core.engine import create_engine
    engine = create_engine(
        "cuda_implicit", p=p, seed=seed, w_max=w_max,
    )
    engine.add_area("A", n=n, k=k, beta=beta)
    engine.add_stimulus("s", size=k)
    engine.add_connectivity("s", "A", p=p)
    return engine


class TestLearningAccumulates:
    """Repeated projection strengthens the fiber, bounded by w_max.

    REWRITTEN. These tests read `engine._pair_learned[("s", "A")].num_learned`,
    an attribute `CudaImplicitEngine` has never had -- `git log -S` finds it was
    never in `core/` at all. The COO/`num_learned` structure belongs to
    `kernels/implicit.ImplicitAssemblyArea`, a different object from the engine
    `_make_engine` returns, so these were not stale-after-a-refactor: they
    tested something that was never there. #40 called them stale, which is why
    nobody looked.

    They now assert the BEHAVIOUR they were named for, through
    `ProjectionResult.total_activation` -- which every engine populates, so the
    same assertions run on the CPU engines too. Testing engine internals is
    what made them unportable AND wrong; the same lesson cost an hour earlier
    today when a parity test reached for `_area_pot` (numpy_exact) against
    connectome weights (numpy_sparse).
    """

    def test_drive_grows_with_training(self):
        engine = _make_engine()
        first = engine.project_into("A", from_stimuli=["s"], from_areas=[])
        for _ in range(20):
            last = engine.project_into("A", from_stimuli=["s"], from_areas=[])
        assert last.total_activation > first.total_activation, (
            f"20 rounds of Hebbian training did not raise the drive "
            f"({first.total_activation:.1f} -> {last.total_activation:.1f}) "
            f"-- the fiber is not learning")

    def test_determinism_across_engines(self):
        """Two engines with same seed produce identical winners."""
        rounds = 20
        e1 = _make_engine(seed=123)
        e2 = _make_engine(seed=123)
        for _ in range(rounds):
            e1.project_into("A", from_stimuli=["s"], from_areas=[])
            e2.project_into("A", from_stimuli=["s"], from_areas=[])
        np.testing.assert_array_equal(
            np.sort(np.asarray(e1.get_winners("A"))),
            np.sort(np.asarray(e2.get_winners("A"))))

    def test_drive_saturates_rather_than_diverging(self):
        """`w_max` is a ceiling, so the drive must level off."""
        engine = _make_engine(beta=0.5, w_max=2.0)
        drives = []
        for _ in range(40):
            drives.append(engine.project_into(
                "A", from_stimuli=["s"], from_areas=[]).total_activation)
        early = drives[5] - drives[4]
        late = drives[-1] - drives[-2]
        assert late <= max(early, 1e-9), (
            f"drive still climbing at the same rate after 40 rounds with "
            f"w_max=2.0 (early step {early:.3f}, late step {late:.3f}) -- "
            f"the clamp is not binding")

    def test_no_overflow(self):
        """float32 overflow poisons column sums; this caught it once already."""
        engine = _make_engine(beta=0.9, w_max=20.0)
        for _ in range(60):
            r = engine.project_into("A", from_stimuli=["s"], from_areas=[])
        assert np.isfinite(r.total_activation), (
            f"total activation is {r.total_activation} after 60 high-beta "
            f"rounds -- weights have overflowed")


class TestConnectionReset:
    """`reset_area_connections` returns AREA->AREA fibers to untrained state.

    Scope matters and the name is accurate: it walks `_area_conns` only.
    Stimulus fibers are untouched. The first version of this test trained
    through a stimulus and asserted the reset dropped the drive -- it does not,
    and should not. Testing the method against the wrong fiber would have
    reported a defect that is not there.
    """

    def test_reset_drops_the_learned_area_to_area_drive(self):
        from neural_assemblies.core.engine import create_engine
        engine = create_engine("cuda_implicit", p=0.05, seed=42, w_max=20.0)
        engine.add_area("A", n=5000, k=50, beta=0.05)
        engine.add_area("B", n=5000, k=50, beta=0.05)
        engine.add_stimulus("s", size=50)

        for _ in range(10):
            engine.project_into("A", from_stimuli=["s"], from_areas=[])
        for _ in range(20):
            trained = engine.project_into("B", from_stimuli=[],
                                          from_areas=["A"])
        engine.reset_area_connections("B")
        after = engine.project_into("B", from_stimuli=[], from_areas=["A"])
        assert after.total_activation < trained.total_activation, (
            f"reset did not clear the learned A->B drive "
            f"({trained.total_activation:.1f} -> {after.total_activation:.1f})")

    def test_reset_leaves_stimulus_fibers_alone(self):
        """The complement, so the SCOPE is pinned and not just the effect."""
        engine = _make_engine()
        for _ in range(20):
            trained = engine.project_into("A", from_stimuli=["s"],
                                          from_areas=[])
        engine.reset_area_connections("A")
        after = engine.project_into("A", from_stimuli=["s"], from_areas=[])
        assert after.total_activation >= trained.total_activation * 0.9, (
            f"reset_area_connections cleared a STIMULUS fiber "
            f"({trained.total_activation:.1f} -> {after.total_activation:.1f})"
            f" -- it is documented to touch area->area only")


class TestMultiAreaProjection:
    def test_area_to_area_learns_on_both_fibers(self):
        """s -> A then A -> B must strengthen BOTH fibers, not just the first."""
        from neural_assemblies.core.engine import create_engine
        engine = create_engine("cuda_implicit", p=0.05, seed=42, w_max=20.0)
        engine.add_area("A", n=5000, k=50, beta=0.05)
        engine.add_area("B", n=5000, k=50, beta=0.05)
        engine.add_stimulus("s", size=50)

        sa_first = engine.project_into("A", from_stimuli=["s"], from_areas=[])
        for _ in range(10):
            sa_last = engine.project_into("A", from_stimuli=["s"],
                                          from_areas=[])
        ab_first = engine.project_into("B", from_stimuli=[], from_areas=["A"])
        for _ in range(10):
            ab_last = engine.project_into("B", from_stimuli=[],
                                          from_areas=["A"])
        assert sa_last.total_activation > sa_first.total_activation
        assert ab_last.total_activation > ab_first.total_activation


class TestProjectRounds:
    """`project_rounds` is a LOOP, and the old tests assumed it was not.

    They passed `from_areas=[]` to `project_rounds` while giving the sequential
    arm `from_areas=["A"]`, on the stated premise that "project_rounds handles
    target self-recurrence internally (round_idx > 0)". NO IMPLEMENTATION DOES
    THAT. The base in `core/engine.py`, the CUDA override and the torch
    override are the same pure loop over `project_into` with exactly the
    arguments given. So the two arms ran DIFFERENT PROTOCOLS -- one with A->A
    for nine rounds, one without -- and 0/50 winners matched, on numpy_sparse
    and cuda_implicit alike.

    That is why #40 filed this as "stale CUDA tests": the failure looked like
    GPU drift. It is neither stale nor GPU-specific; the tests encoded an API
    that was never built.

    The contract worth pinning is the one the fast path actually offers, and it
    holds exactly: same `from_areas`, same result, 50/50 on numpy_sparse,
    cuda_implicit and torch_sparse.
    """

    def _make_two_engines(self, seed=42, n=5000, k=50):
        from neural_assemblies.core.engine import create_engine
        engines = []
        for _ in range(2):
            e = create_engine("cuda_implicit", p=0.05, seed=seed, w_max=20.0)
            e.add_area("A", n=n, k=k, beta=0.05)
            e.add_stimulus("s", size=k)
            engines.append(e)
        return engines

    def test_stim_only_matches_sequential(self):
        """Identical arguments on both sides -- that is the whole contract."""
        e_seq, e_fast = self._make_two_engines(seed=200)
        e_seq.project_into("A", from_stimuli=["s"], from_areas=[])
        e_fast.project_into("A", from_stimuli=["s"], from_areas=[])

        rounds = 9
        for _ in range(rounds):
            e_seq.project_into("A", from_stimuli=["s"], from_areas=["A"])
        e_fast.project_rounds(
            target="A", from_stimuli=["s"], from_areas=["A"],
            rounds=rounds, plasticity_enabled=True,
        )
        np.testing.assert_array_equal(
            np.sort(np.asarray(e_seq.get_winners("A"))),
            np.sort(np.asarray(e_fast.get_winners("A"))))

    def test_area_source_matches_sequential(self):
        from neural_assemblies.core.engine import create_engine
        seed = 300
        engines = []
        for _ in range(2):
            e = create_engine("cuda_implicit", p=0.05, seed=seed, w_max=20.0)
            e.add_area("A", n=5000, k=50, beta=0.05)
            e.add_area("B", n=5000, k=50, beta=0.05)
            e.add_stimulus("s", size=50)
            engines.append(e)
        e_seq, e_fast = engines

        for _ in range(5):
            e_seq.project_into("A", from_stimuli=["s"], from_areas=[])
            e_fast.project_into("A", from_stimuli=["s"], from_areas=[])
        e_seq.project_into("B", from_stimuli=[], from_areas=["A"])
        e_fast.project_into("B", from_stimuli=[], from_areas=["A"])

        rounds = 4
        for _ in range(rounds):
            e_seq.project_into("B", from_stimuli=[], from_areas=["A", "B"])
        e_fast.project_rounds(
            target="B", from_stimuli=[], from_areas=["A", "B"],
            rounds=rounds, plasticity_enabled=True,
        )
        np.testing.assert_array_equal(
            np.sort(np.asarray(e_seq.get_winners("B"))),
            np.sort(np.asarray(e_fast.get_winners("B"))))

    def test_project_rounds_does_not_add_recurrence_for_you(self):
        """Pins the SEMANTICS the old tests got wrong, so nobody re-assumes it.

        Omitting the target from `from_areas` means NO self-recurrence. If a
        future engine starts adding it internally, these two must diverge and
        this test says so before anything downstream silently changes.
        """
        e_with, e_without = self._make_two_engines(seed=400)
        for e, srcs in ((e_with, ["A"]), (e_without, [])):
            e.project_into("A", from_stimuli=["s"], from_areas=[])
            e.project_rounds(target="A", from_stimuli=["s"], from_areas=srcs,
                             rounds=9, plasticity_enabled=True)
        same = np.array_equal(
            np.sort(np.asarray(e_with.get_winners("A"))),
            np.sort(np.asarray(e_without.get_winners("A"))))
        assert not same, (
            "project_rounds now produces the same result with and without the "
            "target in from_areas -- it has started injecting self-recurrence, "
            "which changes the meaning of every call site")


