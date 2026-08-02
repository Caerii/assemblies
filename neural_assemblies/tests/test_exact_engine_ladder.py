"""The acceptance ladder for `numpy_exact` (task #85).

Ordered so each rung is falsifiable on its own, and named so a failure says
which claim broke. L0-L2 are NECESSARY but not sufficient: an engine can pass
all three and still merge disjoint inputs, which is the defect the engine
exists to remove. **L3 is the acceptance test.**

  L0  same substrate   initial weights identical to the explicit engine
  L1  same drive       pre-kWTA drive identical, both norm_init settings
  L2  same dynamics    winners identical over a multi-round protocol
  L3  THE POINT        graded similarity: chance for disjoint inputs, not 0.906
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine._exact import NumpyExactEngine

N, K, P, BETA, SEED = 600, 30, 0.1, 0.05, 42


def _exact(norm_init=True, n=N, k=K, seed=SEED):
    e = NumpyExactEngine(p=P, seed=seed, norm_init=norm_init)
    e.add_stimulus("s", k)
    e.add_area("A", n, k, BETA)
    e.add_area("B", n, k, BETA)
    return e


def _explicit(norm_init=True, n=N, k=K, seed=SEED):
    b = Brain(p=P, seed=seed, engine="numpy_sparse", norm_init=norm_init)
    b.add_stimulus("s", k)
    b.add_area("A", n, k, BETA)
    b.add_area("B", n, k, BETA)
    b._engine.materialize_area("A")
    b._engine.materialize_area("B")
    return b


# -- L0 -- same substrate ---------------------------------------------------

class TestL0Substrate:
    """Initial weights must be the SAME NUMBERS, not merely the same law."""

    def test_stimulus_fiber_cannot_be_compared_elementwise_and_here_is_why(self):
        """The sparse engine's stim fiber is STREAM-drawn, so parity is not
        available on this path -- and that is a defect there, not here.

        Door 6 ("stim->area init was still drawn from the stream") was reverted
        pending this engine. Measured: declaring three UNUSED stimuli rewrites
        the s->A fiber in `numpy_sparse` (97/600 values survive) while leaving
        this engine bit-identical. So an elementwise assertion against the
        sparse engine would be asserting that we reproduce draw-order
        dependence. Test the property instead.
        """
        def sparse(extra):
            b = Brain(p=P, seed=SEED, engine="numpy_sparse")
            for i in range(extra):
                b.add_stimulus(f"x{i}", K)
            b.add_stimulus("s", K)
            b.add_area("A", N, K, BETA)
            b._engine.materialize_area("A")
            return np.asarray(b._engine._stim_conns["s"]["A"].weights,
                              dtype=np.float64)[:N]

        def exact(extra):
            e = NumpyExactEngine(p=P, seed=SEED)
            for i in range(extra):
                e.add_stimulus(f"x{i}", K)
            e.add_stimulus("s", K)
            e.add_area("A", N, K, BETA)
            return e._stim_base["s"]["A"]

        assert not np.array_equal(sparse(0), sparse(3)), (
            "numpy_sparse became order-independent on the stimulus fiber -- if "
            "door 6 has landed, this test and L1/L2 should compare elementwise")
        assert np.array_equal(exact(0), exact(3)), (
            "declaring an unused stimulus changed this engine's substrate; "
            "content addressing is broken")

    def test_stimulus_afferent_counts_have_the_right_law(self):
        """What CAN be asserted absolutely: it is Binomial(stim_size, p)."""
        e = _exact(n=20_000)
        base = e._stim_base["s"]["A"]
        assert base.mean() == pytest.approx(K * P, rel=0.05)
        assert base.var() == pytest.approx(K * P * (1 - P), rel=0.15)

    @pytest.mark.parametrize("row", [0, 7, N - 1])
    def test_area_fiber_rows_are_identical(self, row):
        b, e = _explicit(), _exact()
        conn = np.asarray(b._engine._area_conns["A"]["B"].weights, dtype=np.float64)
        mine = e._fiber_rows("A", "B", np.array([row]), N)[0]
        assert np.array_equal(conn[row, :N], mine)

    def test_recomputation_is_order_independent(self):
        """Doors 5 and 6 are gone BY CONSTRUCTION, not by patch."""
        e = _exact()
        rows = np.array([5, 2, 9])
        forward = e._fiber_rows("A", "B", rows, N)
        backward = e._fiber_rows("A", "B", rows[::-1], N)[::-1]
        assert np.array_equal(forward, backward)
        # and asking for one row alone gives the same numbers as in a block
        assert np.array_equal(e._fiber_rows("A", "B", np.array([2]), N)[0],
                              forward[1])


# -- L1 -- same drive -------------------------------------------------------

def _seed_source(b, e, pattern):
    """Put the SAME source assembly in both engines, bypassing the stimulus.

    The stimulus path cannot be compared elementwise (see L0), so the source is
    set directly. Legal because the area-fiber substrates ARE identical, and
    the engines index that fiber's rows the same way -- which is exactly what
    `test_area_fiber_rows_are_identical` establishes.
    """
    b._engine.set_winners("A", np.asarray(pattern, dtype=np.uint32))
    e.set_winners("A", np.asarray(pattern, dtype=np.uint32))


def _ab_drive(e, pattern, norm_init):
    """The exact A->B drive vector this engine computes, untouched by k-WTA."""
    block = e._fiber_rows("A", "B", np.asarray(pattern, dtype=np.int64), N)
    d = block.sum(axis=0)
    if norm_init:
        d = d * e._area_norm("A", "B")
    return d


def _assert_same_drive_modulo_ties(got, mine, drive, k, label):
    """The precise claim: the DRIVE agrees; only the tied band may differ.

    A raw overlap threshold cannot express this. With `norm_init=False` the
    drive is an integer afferent count, so the k-th boundary lands inside a
    large exactly-tied band (measured here: 30 neurons tied for 11 slots at
    k*p=3, and raising k*p does NOT help because the tie is intrinsic to a
    discrete drive). Whichever neurons come out of that band is decided by
    convention, not by the model -- so requiring identical winners would be
    requiring identical conventions, which is not what parity means.

    `norm_init=True` divides by a per-neuron 1/d_j, making the drive
    real-valued, and there the winners DO match exactly.
    """
    boundary = np.sort(drive)[-k]
    strictly_above = set(np.flatnonzero(drive > boundary).tolist())

    assert strictly_above <= got, (
        f"{label}: sparse dropped a neuron whose drive strictly exceeds the "
        f"boundary -- that is a drive disagreement, not a tie-break")
    assert strictly_above <= mine, (
        f"{label}: exact dropped a strictly-above-boundary neuron")
    for name, sel in (("sparse", got), ("exact", mine)):
        below = [i for i in sel if drive[i] < boundary]
        assert not below, (
            f"{label}: {name} selected {len(below)} neurons BELOW the boundary "
            f"drive; the two engines are not computing the same drive")


class TestL1Drive:
    """Pre-kWTA drive over the AREA fiber, read through the winners."""

    @pytest.mark.parametrize("norm_init", [False, True])
    def test_area_projection_agrees_on_the_drive(self, norm_init):
        b, e = _explicit(norm_init), _exact(norm_init)
        pattern = np.arange(K, dtype=np.uint32) * 3      # arbitrary, shared
        _seed_source(b, e, pattern)
        drive = _ab_drive(e, pattern, norm_init)

        b.project({}, {"A": ["B"]})
        got = set(np.array(b.areas["B"].winners, dtype=np.int64).tolist())
        mine = set(e.project_into("B", [], ["A"]).winners.astype(np.int64).tolist())

        _assert_same_drive_modulo_ties(got, mine, drive, K,
                                       f"norm_init={norm_init}")

    def test_normalised_drive_gives_EXACT_winner_parity(self):
        """No tied band under norm_init, so nothing is left to convention."""
        b, e = _explicit(True), _exact(True)
        pattern = np.arange(K, dtype=np.uint32) * 3
        _seed_source(b, e, pattern)
        b.project({}, {"A": ["B"]})
        got = np.sort(np.array(b.areas["B"].winners, dtype=np.int64))
        mine = np.sort(e.project_into("B", [], ["A"]).winners.astype(np.int64))
        assert np.array_equal(got, mine)


# -- L2 -- same dynamics ----------------------------------------------------

class TestL2Dynamics:
    """Now with plasticity compounding: beta, w_max and the norm scale."""

    def test_repeated_plastic_projection_stays_identical_under_norm_init(self):
        """Six plastic rounds, exact winner parity throughout.

        This is the rung that would catch a `w_max` clamp applied on the wrong
        scale, or a potentiation exponent that drifts -- both compound, so a
        single-round test has no power against them.
        """
        b, e = _explicit(True), _exact(True)
        pattern = np.arange(K, dtype=np.uint32) * 3
        for rnd in range(6):
            _seed_source(b, e, pattern)          # hold the source fixed
            b.project({}, {"A": ["B"]})
            e.project_into("B", [], ["A"])
            got = np.sort(np.array(b.areas["B"].winners, dtype=np.int64))
            mine = np.sort(np.asarray(e._areas["B"].winners, dtype=np.int64))
            assert np.array_equal(got, mine), (
                f"diverged at plastic round {rnd}: overlap "
                f"{len(set(got.tolist()) & set(mine.tolist()))}/{K}")


# -- L3 -- THE ACCEPTANCE TEST ----------------------------------------------

class TestL3GradedSimilarity:
    """The rung that justifies the engine existing.

    `numpy_sparse` gives 0.906 overlap for FULLY DISJOINT inputs at low area
    load where the exact substrate gives chance (measured 18.0x chance +/-
    0.0231 at 8 seeds, `research/notes/graded_similarity_and_sampler_load.md`).
    L0-L2 can all pass on an engine that still does that.
    """

    @staticmethod
    def _read(e, pattern, n):
        e._areas["A"].winners = np.asarray(pattern, dtype=np.uint32)
        e._areas["A"].fixed_assembly = True
        e.project_into("B", [], ["A"], plasticity_enabled=False)
        return set(np.asarray(e._areas["B"].winners, dtype=np.int64).tolist())

    def test_disjoint_inputs_land_at_chance(self):
        e = _exact()
        e.project_into("B", ["s"], [])          # give B a population
        rng = np.random.default_rng(7)
        pool = rng.permutation(N)
        a = self._read(e, pool[:K], N)
        b = self._read(e, pool[K:2 * K], N)
        ov = len(a & b) / K
        chance = K / N
        assert ov < 4 * chance, (
            f"disjoint inputs overlap {ov:.4f} against chance {chance:.4f} "
            f"({ov/chance:.1f}x). The sparse engine reads 18x here; if this "
            f"engine does too it has not fixed the defect it exists to fix.")

    def test_similarity_is_graded_and_monotone(self):
        """Not just "disjoint is fine" -- the whole curve must be ordered."""
        e = _exact()
        e.project_into("B", ["s"], [])
        rng = np.random.default_rng(7)
        pool = rng.permutation(N)
        ref, disjoint = list(pool[:K]), list(pool[K:])
        base = self._read(e, ref, N)
        curve = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            shared = int(round(f * K))
            pat = ref[:shared] + disjoint[:K - shared]
            curve.append(len(self._read(e, pat, N) & base) / K)
        assert curve[-1] == 1.0, "identical inputs must give identical output"
        assert all(x <= y + 1e-9 for x, y in zip(curve, curve[1:])), (
            f"similarity is not monotone in shared fraction: {curve}")
        assert curve[-2] > curve[0] + 0.1, (
            f"curve is flat, so nothing is graded: {curve}")
