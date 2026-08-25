"""The k-WTA bound-and-prune must be EXACT, not approximate.

A prune that is merely usually right is worse than no prune at all: it changes
which neurons win, silently, in a way that looks like a result. So every test
here compares against brute force on the same data rather than against a
remembered number, and the adversarial cases are the ones that have actually
bitten:

  * structural ZEROS must not read as potentiated (`!= 1.0` vs `> 1.0`);
  * the bound must hold in float64 while the values stay float32;
  * ties must break the same way pruned as unpruned.

`assert_pruned_matches_brute` is the whole contract; the rest of the file is
the space of inputs it is asserted over.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.numpy_engine._kwta_prune import (
    PotentiatedSupport, bound_outside, evaluate_set, masked_drive, tau_of,
)


def _fiber(rng, n_rows, n_cols, p, dtype=np.float32):
    """A base connectome: Bernoulli 0/1, exactly as the engine initialises."""
    return (rng.random((n_rows, n_cols)) < p).astype(dtype)


def _train(w, sup, rng, rounds, k_rows, k_cols, beta=0.1):
    """Apply plasticity events and record them, exactly as the engine does:
    the FULL cross product `rows x cols`, multiplicatively."""
    n_rows, n_cols = w.shape
    for _ in range(rounds):
        rows = rng.choice(n_rows, size=min(k_rows, n_rows), replace=False)
        cols = rng.choice(n_cols, size=min(k_cols, n_cols), replace=False)
        w[np.ix_(rows, cols)] *= (1.0 + beta)
        sup.note(rows, cols)
    return w


def _brute_topk(drive, k):
    """Reference top-k with a STABLE order, so ties are deterministic."""
    return np.sort(np.argsort(-np.asarray(drive, dtype=np.float64),
                              kind="stable")[:k])


def assert_pruned_matches_brute(w, sup, rows, k, stim=None):
    """THE CONTRACT. Returns how many columns the prune evaluated."""
    n_cols = w.shape[1]
    exact = w[rows, :].sum(axis=0).astype(np.float64)
    if stim is not None:
        exact = exact + np.asarray(stim, dtype=np.float64)

    corr, touched = sup.correction(rows, w, n_cols)
    ev = evaluate_set(touched, stim, k, n_cols)

    # The bound must never sit BELOW the true drive: that is the failure that
    # silently drops real winners, and it is what `!= 1.0` produced.
    upper = (np.asarray(stim, dtype=np.float64) if stim is not None
             else np.zeros(n_cols)) + float(len(rows)) + corr
    assert np.all(upper + 1e-9 >= exact), (
        "bound below true drive at "
        f"{int(np.argmax(exact - upper))}: U={upper[np.argmax(exact-upper)]} "
        f"< drive={exact[np.argmax(exact-upper)]}")

    tau = tau_of(exact, ev, k)
    if tau is None:
        return None                       # fewer than k evaluated -> fallback
    if bound_outside(stim, ev, n_cols, len(rows)) >= tau:
        return None                       # bound loses -> fallback

    got = _brute_topk(masked_drive(exact, ev), k)
    want = _brute_topk(exact, k)
    assert np.array_equal(got, want), (
        f"pruned top-{k} {got.tolist()} != brute {want.tolist()}")
    # and the VALUES at the winners must be bit-identical, not merely ranked
    assert np.array_equal(exact[got], exact[want])
    return len(ev)


class TestExactness:
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_brute_force_after_training(self, seed):
        rng = np.random.default_rng(seed)
        w = _fiber(rng, 200, 3000, 0.05)
        sup = PotentiatedSupport()
        _train(w, sup, rng, rounds=25, k_rows=20, k_cols=20)
        rows = rng.choice(200, size=20, replace=False)
        assert_pruned_matches_brute(w, sup, rows, k=10)

    @pytest.mark.parametrize("seed", range(4))
    def test_matches_brute_force_with_a_stimulus_term(self, seed):
        rng = np.random.default_rng(100 + seed)
        w = _fiber(rng, 150, 2000, 0.05)
        sup = PotentiatedSupport()
        _train(w, sup, rng, rounds=30, k_rows=15, k_cols=15)
        rows = rng.choice(150, size=15, replace=False)
        stim = rng.random(2000) * 3.0
        assert_pruned_matches_brute(w, sup, rows, k=10, stim=stim)

    def test_untrained_fiber_falls_back_rather_than_guessing(self):
        """With no potentiation there is nothing to evaluate, so the prune
        must DECLINE. Returning a top-k here would be pure luck."""
        rng = np.random.default_rng(3)
        w = _fiber(rng, 100, 1000, 0.05)
        sup = PotentiatedSupport()
        rows = rng.choice(100, size=10, replace=False)
        assert assert_pruned_matches_brute(w, sup, rows, k=10) is None


class TestTheThreeTraps:
    def test_structural_zeros_are_not_potentiated(self):
        """`!= 1.0` counts every structural zero as touched. That is not just
        slow -- a zero would contribute (0 - 1) = -1 to the correction and push
        the bound BELOW the true drive, dropping real winners."""
        w = np.zeros((4, 6), dtype=np.float32)
        w[0, 0] = w[1, 1] = 1.0             # present, unpotentiated
        sup = PotentiatedSupport()
        sup.note([0, 1, 2, 3], np.arange(6))   # the whole block is "touched"
        corr, touched = sup.correction([0, 1, 2, 3], w, 6)
        assert corr.sum() == 0.0, corr
        assert len(touched) == 0, touched

    def test_a_potentiated_cell_over_a_zero_base_stays_excluded(self):
        """`w *= 1+beta` cannot grow a zero, so a touched cell whose base was
        absent is still absent and must not be evaluated."""
        w = np.zeros((2, 4), dtype=np.float32)
        w[0, 1] = 1.0
        sup = PotentiatedSupport()
        sup.note([0, 1], np.arange(4))
        w[np.ix_([0, 1], np.arange(4))] *= 1.1
        corr, touched = sup.correction([0, 1], w, 4)
        assert touched.tolist() == [1]
        assert corr[1] == pytest.approx(0.1, abs=1e-6)

    def test_correction_is_float64_even_though_weights_are_float32(self):
        """f32 accumulation drifts ~1e-6 on drives of ~120, which shows up as
        spurious bound violations -- a fast exact path turning slow at random."""
        rng = np.random.default_rng(11)
        w = _fiber(rng, 120, 500, 0.5, dtype=np.float32)
        sup = PotentiatedSupport()
        _train(w, sup, rng, rounds=40, k_rows=60, k_cols=60)
        corr, _ = sup.correction(np.arange(120), w, 500)
        assert corr.dtype == np.float64

    def test_pruned_vector_keeps_full_length_and_index(self):
        """Shrinking the vector renumbers columns and changes the tie-break."""
        drive = np.array([5.0, 9.0, 1.0, 9.0, 2.0], dtype=np.float32)
        out = masked_drive(drive, np.array([1, 3], dtype=np.int64))
        assert len(out) == len(drive)
        assert out[1] == 9.0 and out[3] == 9.0
        assert np.isneginf(out[0]) and np.isneginf(out[2])

    def test_ties_break_identically_pruned_and_unpruned(self):
        """A pruned column has drive <= U < tau, so it cannot tie with a
        winner -- but the retained ones must still tie among themselves the
        same way ([[exact-tables-are-tie-fragile]])."""
        drive = np.array([7.0, 7.0, 7.0, 1.0, 1.0], dtype=np.float64)
        ev = np.array([0, 1, 2], dtype=np.int64)
        assert np.array_equal(_brute_topk(masked_drive(drive, ev), 2),
                              _brute_topk(drive, 2))


class TestTheBound:
    def test_bound_outside_uses_the_largest_unevaluated_stimulus(self):
        stim = np.array([0.0, 10.0, 0.0, 3.0], dtype=np.float64)
        ev = np.array([1], dtype=np.int64)          # the big one IS evaluated
        assert bound_outside(stim, ev, 4, active_total=5) == pytest.approx(8.0)
        assert bound_outside(stim, np.empty(0, np.int64), 4, 5) == \
            pytest.approx(15.0)

    def test_bound_with_no_stimulus_is_just_the_active_count(self):
        assert bound_outside(None, np.empty(0, np.int64), 10, 7) == 7.0

    def test_evaluate_set_includes_the_stimulus_top_k(self):
        """A column with NO potentiation can still win on stimulus alone, so
        skipping the stimulus top-k is a correctness bug, not a slow path."""
        stim = np.zeros(50)
        stim[42] = 100.0
        ev = evaluate_set(np.array([1, 2], dtype=np.int64), stim, k=3,
                          n_cols=50)
        assert 42 in ev.tolist()

    def test_tau_declines_when_fewer_than_k_were_evaluated(self):
        assert tau_of(np.arange(10.0), np.array([0, 1], dtype=np.int64), 5) \
            is None


class TestSupportMaintenance:
    def test_note_is_a_cross_product_and_dedupes(self):
        sup = PotentiatedSupport()
        sup.note([0, 1], np.array([5, 3]))
        sup.note([1], np.array([3, 9]))
        w = np.ones((2, 10), dtype=np.float32) * 2.0     # all potentiated
        _corr, touched = sup.correction([0, 1], w, 10)
        assert touched.tolist() == [3, 5, 9]

    def test_clear_drops_everything(self):
        """Consolidation renumbers compact indices, so a stale row->col map
        would point at other neurons' columns."""
        sup = PotentiatedSupport()
        sup.note([0], np.array([1, 2]))
        sup.clear()
        w = np.ones((1, 4), dtype=np.float32) * 2.0
        corr, touched = sup.correction([0], w, 4)
        assert len(touched) == 0 and corr.sum() == 0.0

    def test_columns_beyond_the_current_extent_are_dropped(self):
        """An area grows; a support recorded when it was wider must not index
        past the logical extent."""
        sup = PotentiatedSupport()
        sup.note([0], np.array([1, 50]))
        w = np.ones((1, 4), dtype=np.float32) * 2.0
        _corr, touched = sup.correction([0], w, 4)
        assert touched.tolist() == [1]


def test_prune_is_sublinear_in_area_size():
    """The evaluated set is set by the STRUCTURE of what was trained, not by
    n. This is the whole claim: the advantage grows with area size, which is
    why it is the barrier to scaling to large brains.

    Run in the regime where the prune APPLIES (see the module docstring:
    `p * (1+beta)^T > 1`). The same training structure at every size -- same
    rows, same 100 columns -- with only the area growing around it.
    """
    counts = {}
    for n_cols in (2000, 8000, 32000):
        rng = np.random.default_rng(7)
        w = _fiber(rng, 120, n_cols, 0.5)
        sup = PotentiatedSupport()
        rows_t = np.arange(40)
        cols_t = np.arange(100)
        for _ in range(12):
            w[np.ix_(rows_t, cols_t)] *= 1.3
            sup.note(rows_t, cols_t)
        n = assert_pruned_matches_brute(w, sup, rows_t[:20], k=5)
        counts[n_cols] = n

    assert all(v is not None for v in counts.values()), (
        f"prune declined where it should apply: {counts}")
    # 16x the area must not cost 16x the evaluation.
    assert counts[32000] < 3 * counts[2000], counts
    assert counts[32000] <= 120, counts       # ~the trained support, not n


def test_prune_declines_below_its_applicability_condition():
    """`p * (1+beta)^T > 1` is not a heuristic -- below it the potentiated
    signal genuinely cannot clear the base bound, and declining is CORRECT.
    A prune that fired here would be guessing."""
    rng = np.random.default_rng(5)
    w = _fiber(rng, 120, 4000, 0.05)          # p=0.05
    sup = PotentiatedSupport()
    rows_t, cols_t = np.arange(40), np.arange(100)
    for _ in range(4):                        # 1.1^4 = 1.46; 0.05*1.46 << 1
        w[np.ix_(rows_t, cols_t)] *= 1.1
        sup.note(rows_t, cols_t)
    assert assert_pruned_matches_brute(w, sup, rows_t[:20], k=5) is None
