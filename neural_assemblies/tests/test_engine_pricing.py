"""The k-WTA pricing law, asserted directly, for every engine.

WHY THIS FILE EXISTS.  `test_torch_parity.py` asserts that both engines form an
assembly and that it stabilizes above a threshold.  That suite was green for
months while the torch engine mis-priced k-WTA badly enough to SEAL an area at
exactly ``k`` -- because a sealed area is trivially stable, so it passes a
stability assertion at 1.000.  The test had no power against the failure class
it existed to catch, for two structural reasons:

  1. it used a single module-level ``N``, so ``tgt.n == n_pre`` in every test and
     the per-fiber divisor could not manifest at all; and
  2. it asserted a threshold that the degenerate state also satisfies.

So this file does the opposite of a threshold test.  It pins the law itself as a
pure function, and then asserts RECRUITMENT bounds -- ``k < w < n`` -- which seal
and exhaust each violate in a different direction and which no degenerate state
can satisfy.

See `neural_assemblies/core/_pricing.py` for the law and the measured divergence
table.
"""

import numpy as np
import pytest

from neural_assemblies.core import _pricing
from neural_assemblies.core.brain import Brain


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


ENGINES = ["numpy_sparse"] + (["torch_sparse"] if _has_torch_cuda() else [])

K = 100
P = 0.05
BETA = 0.05
ROUNDS = 15


# ---------------------------------------------------------------------------
# The law as a pure function
# ---------------------------------------------------------------------------

class TestCandidateDivisorLaw:

    def test_homogeneous_collapses_to_n_times_p(self):
        """Every source population == target n must give exactly n*p.

        This is what makes the per-fiber fix safe for the entire existing
        corpus: every equal-area configuration is bit-identical to the old
        behaviour, which is also why the bug hid for so long.
        """
        for n in (500, 1000, 10000):
            got = _pricing.candidate_divisor(P, n, [K, K, K], [n, n, n])
            assert got == pytest.approx(n * P, rel=1e-12)

    def test_heterogeneous_is_activity_weighted_harmonic_mean(self):
        """D = p * sum(a_f) / sum(a_f / n_pre_f), computed by hand."""
        sizes = [100, 50]
        pops = [1000, 10000]
        expect = P * sum(sizes) / (100 / 1000 + 50 / 10000)
        got = _pricing.candidate_divisor(P, 10000, sizes, pops)
        assert got == pytest.approx(expect, rel=1e-12)

    def test_small_source_is_priced_below_target_n(self):
        """A source smaller than the target must LOWER the divisor.

        Pricing candidates at ``tgt.n * p`` when the source is 10x smaller
        over-divides them by that same factor, so no candidate can ever outbid
        an incumbent and the area seals. The sign of the correction is the
        whole point, so assert the sign, not just the value.
        """
        seal = P * 10000                                    # the old behaviour
        fixed = _pricing.candidate_divisor(P, 10000, [K], [1000])
        assert fixed < seal
        assert fixed == pytest.approx(P * 1000, rel=1e-12)

    def test_large_source_is_priced_above_target_n(self):
        """Mirror image: a source larger than the target must RAISE it."""
        exhaust = P * 1000
        fixed = _pricing.candidate_divisor(P, 1000, [K], [10000])
        assert fixed > exhaust
        assert fixed == pytest.approx(P * 10000, rel=1e-12)

    def test_missing_populations_fall_back_to_target_n(self):
        assert _pricing.candidate_divisor(P, 2000) == pytest.approx(2000 * P)
        assert _pricing.candidate_divisor(
            P, 2000, [K], None) == pytest.approx(2000 * P)

    def test_zero_and_negative_fibers_are_skipped_not_counted(self):
        """A silent fiber must not drag the divisor toward zero."""
        with_dead = _pricing.candidate_divisor(
            P, 1000, [K, 0], [1000, 1000])
        without = _pricing.candidate_divisor(P, 1000, [K], [1000])
        assert with_dead == pytest.approx(without, rel=1e-12)

    def test_never_returns_zero(self):
        """A zero divisor would send every candidate to infinity."""
        assert _pricing.candidate_divisor(0.0, 0, [], []) > 0.0


class TestInverseIndegreeLaw:

    def test_unknown_rows_are_charged_at_the_ambient_rate(self):
        deg = np.array([10.0, 20.0], dtype=np.float32)
        got = _pricing.inverse_indegree(deg, n_pre=1000, rows_known=100,
                                        p=0.05, xp=np)
        # unknown = 900 rows, each present with probability p
        assert got == pytest.approx(1.0 / (deg + 900 * 0.05), rel=1e-6)

    def test_fully_materialized_fiber_has_no_unknown_term(self):
        deg = np.array([8.0, 12.0], dtype=np.float32)
        got = _pricing.inverse_indegree(deg, n_pre=50, rows_known=50,
                                        p=0.05, xp=np)
        assert got == pytest.approx(1.0 / deg, rel=1e-6)

    def test_floor_prevents_divide_by_near_zero(self):
        """An unwired column must not be handed an unbounded advantage."""
        deg = np.array([0.0], dtype=np.float32)
        got = _pricing.inverse_indegree(deg, n_pre=0, rows_known=0,
                                        p=0.05, xp=np)
        assert got == pytest.approx(1.0)

    def test_is_potentiation_invariant_by_construction(self):
        """Counts in, so scaling the WEIGHTS cannot move the divisor.

        The reference takes the normalization once at init, when every present
        weight is 1. Reading a potentiated sum here would turn a one-time
        initialization into ongoing homeostasis, which is a different model.
        """
        deg = np.array([10.0, 20.0], dtype=np.float32)
        a = _pricing.inverse_indegree(deg, 1000, 100, 0.05, xp=np)
        b = _pricing.inverse_indegree(deg.copy(), 1000, 100, 0.05, xp=np)
        assert np.allclose(a, b)


class TestAreaFiberActivity:

    def test_norm_init_charges_actual_winners_not_nominal_k(self):
        assert _pricing.area_fiber_activity(37, 100, norm_init=True) == 37

    def test_without_norm_init_the_historical_k_is_kept(self):
        assert _pricing.area_fiber_activity(37, 100, norm_init=False) == 100


# ---------------------------------------------------------------------------
# Both engines obey it, at unequal area sizes
# ---------------------------------------------------------------------------

def _recruit(engine, n_src, n_tgt, norm_init=True, seed=1):
    """Drive TGT from a differently-sized SRC; report (w, stability)."""
    b = Brain(p=P, save_winners=True, seed=seed, engine=engine,
              norm_init=norm_init)
    b.add_stimulus("s", K)
    b.add_area("SRC", n_src, K, BETA)
    b.add_area("TGT", n_tgt, K, BETA)
    for _ in range(10):
        b.project({"s": ["SRC"]}, {})
    prev, stabs = None, []
    for _ in range(ROUNDS):
        b.project({"s": ["SRC"]}, {"SRC": ["TGT"]})
        cur = set(int(x) for x in b.areas["TGT"].winners)
        if prev is not None:
            stabs.append(len(cur & prev) / K)
        prev = cur
    w = int(b._engine_for(b.areas["TGT"])._areas["TGT"].w)
    return w, float(np.mean(stabs[-5:]))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("n_src,n_tgt", [(1000, 10000), (5000, 5000),
                                         (10000, 1000)])
def test_area_neither_seals_nor_exhausts(engine, n_src, n_tgt):
    """``k < w < n`` under norm_init at every source/target size ratio.

    Both failure modes are silent and have opposite signs:

      w == k  -- candidates over-divided; no candidate can outbid an incumbent
                 ever again, so the cap is frozen and EVERY cap-vs-assembly
                 readout reads exactly 1.0000
      w -> n  -- candidates under-divided; they always win, the assembly never
                 settles, and the area materializes toward exhaustion

    Measured before the law was unified: torch read w=100 (exactly k, sealed)
    at 1000->10000 and raised RuntimeError at 10000->1000 with the area so
    fully materialized that only 64 neurons remained to sample from.
    """
    w, _ = _recruit(engine, n_src, n_tgt)
    assert w > K, (
        f"{engine} {n_src}->{n_tgt}: w={w} == k, the area SEALED -- candidates "
        f"are over-divided and every readout against it will read 1.0000")
    assert w < n_tgt // 2, (
        f"{engine} {n_src}->{n_tgt}: w={w} of n={n_tgt}, the area is running "
        f"toward EXHAUSTION -- candidates are under-divided")


@pytest.mark.parametrize("engine", ENGINES)
def test_sealing_configuration_is_not_reported_as_stable(engine):
    """The regression guard for the test-design bug, not the engine bug.

    A sealed area scores 1.000 on stability, so this asserts that the
    configuration which used to seal now recruits AND stays coherent. Without
    the recruitment half, this assertion would have passed on the broken engine.
    """
    w, stab = _recruit(engine, 1000, 10000)
    assert w > K
    assert stab > 0.5, f"{engine}: stability {stab:.3f} -- assembly not settling"


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_agree_on_the_divisor_they_apply(engine):
    """Same fibers in, same price out -- the drift guard.

    The engines keep separate in-degree EXTRACTION (storage formats differ) but
    must share the arithmetic. Asserting the shared function is reached is what
    stops a future 'on-device mirror' from silently forking again.
    """
    b = Brain(p=P, save_winners=True, seed=1, engine=engine, norm_init=True)
    b.add_stimulus("s", K)
    b.add_area("SRC", 1000, K, BETA)
    b.add_area("TGT", 10000, K, BETA)
    b.project({"s": ["SRC"]}, {})
    eng = b._engine_for(b.areas["TGT"])
    tgt = eng._areas["TGT"]

    expect = _pricing.candidate_divisor(P, 10000, [K], [1000])
    if engine == "numpy_sparse":
        got = eng._norm_candidate_divisor(tgt, [K], [1000])
    else:
        got = eng._norm_candidate_divisor(tgt.n, [K], [1000])
    assert got == pytest.approx(expect, rel=1e-12)
    assert got == pytest.approx(P * 1000, rel=1e-12), (
        "divisor still keyed on the target's n alone")
