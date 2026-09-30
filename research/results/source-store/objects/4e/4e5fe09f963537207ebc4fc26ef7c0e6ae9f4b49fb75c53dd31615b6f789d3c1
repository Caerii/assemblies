"""Content-addressed synapse initialisation: order-free AND actually Bernoulli.

Two independent properties are pinned here, and the second is the one that is
easy to lose without noticing.

ORDER-FREEDOM is the point of the module: a synapse's weight must not depend on
how many other synapses were materialised first. Losing it is what made probing
unsound -- parsing the same three items in opposite orders gave connectomes
differing in >13000 cells.

DISTRIBUTIONAL FIDELITY is the price of admission. A hash that returns the right
DENSITY but the wrong dependence structure produces no error and no crash; every
connectome just quietly stops being Bernoulli. The raw kernel hash does exactly
that (see test_raw_kernel_hash_is_biased), so these tests compare against
numpy's Generator rather than against p alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.numpy_engine._seeding import (
    fnv1a_pair_seed,
    hash_area_weights,
    hash_bernoulli_2d,
    hash_stim_counts,
    stable_seed,
)

P = 0.05
SEED = fnv1a_pair_seed(42, "NOUN_CORE", "ROLE_AGENT")


def _dispersion(counts, n_draws, p):
    """Observed variance / Binomial(n_draws, p) variance. Independence => ~1."""
    return float(np.var(counts) / (n_draws * p * (1 - p)))


# -- order freedom ---------------------------------------------------------

def test_block_equals_its_quadrants_assembled_in_reverse():
    whole = hash_bernoulli_2d(0, 256, 0, 256, SEED, P)
    piecewise = np.zeros_like(whole)
    for r0, r1, c0, c1 in [(128, 256, 128, 256), (0, 128, 128, 256),
                           (128, 256, 0, 128), (0, 128, 0, 128)]:
        piecewise[r0:r1, c0:c1] = hash_bernoulli_2d(r0, r1, c0, c1, SEED, P)
    assert np.array_equal(whole, piecewise)


def test_growing_by_rows_then_cols_matches_cols_then_rows():
    """The real growth pattern: an L-shaped region added to an existing block."""
    full = hash_bernoulli_2d(0, 96, 0, 96, SEED, P)
    rows_first = np.zeros_like(full)
    rows_first[:64, :64] = hash_bernoulli_2d(0, 64, 0, 64, SEED, P)
    rows_first[64:, :64] = hash_bernoulli_2d(64, 96, 0, 64, SEED, P)
    rows_first[:, 64:] = hash_bernoulli_2d(0, 96, 64, 96, SEED, P)

    cols_first = np.zeros_like(full)
    cols_first[:64, :64] = hash_bernoulli_2d(0, 64, 0, 64, SEED, P)
    cols_first[:64, 64:] = hash_bernoulli_2d(0, 64, 64, 96, SEED, P)
    cols_first[64:, :] = hash_bernoulli_2d(64, 96, 0, 96, SEED, P)

    assert np.array_equal(full, rows_first)
    assert np.array_equal(full, cols_first)


def test_no_generator_is_consumed():
    """Initialisation must not advance any shared stream."""
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    hash_area_weights(0, 128, 0, 128, SEED, P)
    hash_stim_counts(100, 0, 128, SEED, P)
    assert rng.bit_generator.state == before


# -- distributional fidelity -----------------------------------------------

def test_density_matches_p():
    m = hash_bernoulli_2d(0, 1024, 0, 1024, SEED, P)
    assert m.mean() == pytest.approx(P, abs=0.004)


def test_in_and_out_degree_are_binomially_dispersed():
    """The property the raw kernel hash loses.

    ``norm_init`` divides each postsynaptic neuron's incoming weights by its
    in-degree, so a degenerate in-degree distribution does not merely look
    wrong -- it disables the mechanism while leaving every density check green.
    """
    m = hash_bernoulli_2d(0, 1024, 0, 1024, SEED, P)
    assert _dispersion(m.sum(axis=1), 1024, P) == pytest.approx(1.0, abs=0.35)
    assert _dispersion(m.sum(axis=0), 1024, P) == pytest.approx(1.0, abs=0.35)


def test_adjacent_synapses_are_uncorrelated():
    m = hash_bernoulli_2d(0, 512, 0, 512, SEED, P)
    for a in (m, m.T):
        r = np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1]
        assert abs(r) < 0.01


def test_distinct_fibers_agree_only_at_chance():
    other = fnv1a_pair_seed(42, "VERB_CORE", "ROLE_ACTION")
    a = hash_bernoulli_2d(0, 512, 0, 512, SEED, P)
    b = hash_bernoulli_2d(0, 512, 0, 512, other, P)
    chance = P * P + (1 - P) * (1 - P)
    assert float((a == b).mean()) == pytest.approx(chance, abs=0.01)


def test_raw_kernel_hash_is_biased():
    """Pins WHY the finalizer is not optional.

    ``(r*A) ^ (c*B) ^ seed`` is what ``kernels/implicit.py`` and
    ``CudaImplicitEngine`` use, and the Bernoulli test reads its low 24 bits --
    which are close to a function of the low bits of r and c alone. Column
    dispersion collapses to ~0.02 (in-degree almost constant) and neighbouring
    cells correlate at about -0.05. This test documents the defect rather than
    the fix, so it fails loudly if someone "simplifies" the finalizer away.
    """
    raw = hash_bernoulli_2d(0, 1024, 0, 1024, SEED, P, finalize=False)
    assert _dispersion(raw.sum(axis=0), 1024, P) < 0.2
    r = np.corrcoef(raw[:-1].ravel(), raw[1:].ravel())[0, 1]
    assert r < -0.02


# -- semantics -------------------------------------------------------------

def test_inhibitory_synapses_are_a_subset_at_the_right_rate():
    w = hash_area_weights(0, 512, 0, 512, SEED, P,
                          inhibitory_prob=0.3, inhibitory_weight=-1.0)
    present, inhibitory = (w != 0), (w < 0)
    assert (inhibitory & ~present).sum() == 0
    assert inhibitory.sum() / present.sum() == pytest.approx(0.3, abs=0.05)


def test_inhibitory_draw_is_independent_of_presence():
    """Salting matters: unsalted, every present synapse is inhibitory."""
    w = hash_area_weights(0, 512, 0, 512, SEED, P,
                          inhibitory_prob=P, inhibitory_weight=-1.0)
    assert (w < 0).sum() / (w != 0).sum() == pytest.approx(P, abs=0.03)


def test_excitatory_only_is_plain_bernoulli():
    a = hash_area_weights(0, 256, 0, 256, SEED, P, inhibitory_prob=0.0)
    assert np.array_equal(a, hash_bernoulli_2d(0, 256, 0, 256, SEED, P))


def test_stim_counts_match_binomial_mean_and_are_addressed_by_neuron():
    counts = hash_stim_counts(200, 0, 4096, SEED, P)
    assert counts.mean() == pytest.approx(200 * P, abs=0.5)
    assert _dispersion(counts, 200, P) == pytest.approx(1.0, abs=0.35)
    tail = hash_stim_counts(200, 1000, 1064, SEED, P)
    assert np.array_equal(tail, counts[1000:1064])


def test_stim_counts_chunking_does_not_change_the_answer():
    a = hash_stim_counts(3000, 0, 64, SEED, P, chunk=1024)
    b = hash_stim_counts(3000, 0, 64, SEED, P, chunk=97)
    assert np.array_equal(a, b)


# -- seeds ------------------------------------------------------------------

def test_pair_seed_separates_names_and_is_process_stable():
    assert fnv1a_pair_seed(42, "ab", "c") != fnv1a_pair_seed(42, "a", "bc")
    assert fnv1a_pair_seed(42, "A", "B") != fnv1a_pair_seed(43, "A", "B")
    # Pinned literals, not self-comparisons: within one process ANY hash agrees
    # with itself, so only a fixed expected value can catch a per-process seed
    # creeping back in. Run the suite under several PYTHONHASHSEED values and
    # this is the assertion that fails first.
    assert fnv1a_pair_seed(42, "A", "B") == 0xA197CF64
    assert stable_seed("A", "B", 1) == 3229202387
    assert stable_seed("A", "B", 1) != stable_seed("A", "B", 2)


# -- mix32 is written in place, and that must not change a single bit --------

def _mix32_naive(h):
    """The obvious spelling ``mix32`` replaced, kept as the oracle.

    ``mix32`` mutates one reusable temporary instead of allocating five fresh
    ``n x n`` arrays -- 1.52x faster, and 56% of ``materialize_area``'s cost.
    Being faster is worthless if it moves any weight, and the whole connectome
    is addressed through this function, so the equivalence is pinned here
    rather than assumed from review.
    """
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(0x85EBCA6B)
        h = h ^ (h >> np.uint32(13))
        h = h * np.uint32(0xC2B2AE35)
        h = h ^ (h >> np.uint32(16))
    return h


def test_mix32_in_place_is_bit_identical_to_the_naive_spelling():
    from neural_assemblies.core.numpy_engine._seeding import mix32
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2**32, size=(257, 263), dtype=np.uint64
                     ).astype(np.uint32)
    assert np.array_equal(mix32(h), _mix32_naive(h))
    # Edge values the shift/multiply chain is most likely to disagree on.
    edges = np.array([0, 1, 2**31, 2**32 - 1, 0xFFFF, 0x10000],
                     dtype=np.uint32).reshape(2, 3)
    assert np.array_equal(mix32(edges), _mix32_naive(edges))


def test_mix32_default_does_not_consume_its_input():
    """``copy=False`` is an optimisation for callers that own ``h``.

    The default must stay non-destructive: a caller reusing ``h`` after mixing
    would otherwise read scrambled values with no error anywhere.
    """
    from neural_assemblies.core.numpy_engine._seeding import mix32
    h = np.arange(64, dtype=np.uint32)
    before = h.copy()
    mix32(h)
    assert np.array_equal(h, before), "mix32(copy=True) mutated its argument"
    mix32(h, copy=False)
    assert not np.array_equal(h, before)


class TestScatteredRowsKernel:
    """`hash_area_weights_rows` must equal the contiguous kernel cell-for-cell.

    The scattered kernel exists so `VirtualWeights.row_sum` can regenerate k
    winner rows in one call instead of k; if the two spellings ever produce
    different values for the same absolute cell, every virtual fiber silently
    diverges from its dense twin.
    """

    def test_contiguous_rows_match_block_kernel(self):
        from neural_assemblies.core.numpy_engine._seeding import (
            hash_area_weights, hash_area_weights_rows)
        import numpy as np
        block = hash_area_weights(10, 40, 5, 90, 12345, 0.3, 0.1, -0.5)
        rows = hash_area_weights_rows(np.arange(10, 40), 5, 90, 12345,
                                      0.3, 0.1, -0.5)
        assert np.array_equal(block, rows)

    def test_scattered_rows_match_per_row_calls(self):
        from neural_assemblies.core.numpy_engine._seeding import (
            hash_area_weights, hash_area_weights_rows)
        import numpy as np
        ids = np.array([3, 999, 17, 250000, 0, 42], dtype=np.int64)
        batched = hash_area_weights_rows(ids, 0, 120, 777, 0.4, 0.05, -1.0)
        for i, r in enumerate(ids):
            single = hash_area_weights(int(r), int(r) + 1, 0, 120, 777,
                                       0.4, 0.05, -1.0)
            assert np.array_equal(batched[i], single[0]), f"row {r}"

    def test_no_inhibitory_branch_matches_too(self):
        from neural_assemblies.core.numpy_engine._seeding import (
            hash_area_weights, hash_area_weights_rows)
        import numpy as np
        ids = np.array([8, 1, 500], dtype=np.int64)
        batched = hash_area_weights_rows(ids, 3, 60, 42, 0.4)
        for i, r in enumerate(ids):
            single = hash_area_weights(int(r), int(r) + 1, 3, 60, 42, 0.4)
            assert np.array_equal(batched[i], single[0]), f"row {r}"

    def test_numpy_fallback_equals_rust_for_scattered_rows(self):
        """With rust wired into `hash_area_weights_rows`, the other tests in
        this class compare rust to rust. This one pins the numpy SPECIFICATION
        against the accelerator, per the module's contract that the numpy code
        remains the spec."""
        import os

        import numpy as np

        from neural_assemblies.core.numpy_engine import _seeding as SD
        if SD.rust_kernels() is None:
            import pytest
            pytest.skip("accelerator not built")
        ids = np.array([7, 123456, 0, 88, 41999], dtype=np.int64)
        via_rust = SD.hash_area_weights_rows(ids, 2, 300, 991, 0.35, 0.1, -0.7)
        os.environ["NEURAL_ASSEMBLIES_NO_RUST"] = "1"
        try:
            via_numpy = SD.hash_area_weights_rows(ids, 2, 300, 991, 0.35,
                                                  0.1, -0.7)
        finally:
            os.environ.pop("NEURAL_ASSEMBLIES_NO_RUST", None)
        assert np.array_equal(via_rust, via_numpy)

    def test_point_kernel_matches_block_indexing(self):
        """`hash_area_weights_at` element i must equal block[row_i, col_i]."""
        import numpy as np

        from neural_assemblies.core.numpy_engine._seeding import (
            hash_area_weights, hash_area_weights_at)
        rng = np.random.default_rng(3)
        rows = rng.integers(0, 400, 300)
        cols = rng.integers(0, 250, 300)
        block = hash_area_weights(0, 400, 0, 250, 991, 0.35, 0.1, -0.7)
        pts = hash_area_weights_at(rows, cols, 991, 0.35, 0.1, -0.7)
        assert np.array_equal(pts, block[rows, cols])
        block2 = hash_area_weights(0, 400, 0, 250, 991, 0.35)
        pts2 = hash_area_weights_at(rows, cols, 991, 0.35)
        assert np.array_equal(pts2, block2[rows, cols])
