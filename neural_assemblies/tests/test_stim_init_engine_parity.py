"""stim->area initialisation DIVERGES between the numpy and GPU engines.

This documents a real, currently-unfixed difference, so that it is a known
property with a test rather than a surprise found halfway through a
cross-engine comparison.

WHAT DIFFERS. For a stimulus that is not firing, each newly materialised target
neuron needs a count of how many of the stimulus's fibers reach it --
Binomial(stim_size, p).

  * ``numpy_engine/_sparse.py`` draws them SEQUENTIALLY from the shared
    generator, in ``_expand_stim_vectors_fast``. The value a neuron gets
    therefore depends on WHEN it was materialised, and on how many other
    stimuli drew before it.
  * ``cuda_engine.py`` and ``torch_engine/_engine.py`` both call
    ``hash_stim_counts``, which fixes the count for neuron j by j.

``hash_stim_counts`` lives in ``numpy_engine/_seeding.py`` -- the numpy engine
ships the content-addressed primitive and does not use it.

WHAT THAT MEANS IN PRACTICE. The two are the SAME DISTRIBUTION and are
statistically interchangeable; they are not bit-identical and cannot be made so
without changing one of them. So cross-engine results may be compared in
aggregate (accuracy, spread, capacity) but never cell-by-cell, and a golden
recorded on one engine will not reproduce on the other.

WHY IT IS NOT SIMPLY FIXED HERE. Switching numpy to the hash is a
science-affecting change -- it alters every drawn value and requires
re-baselining goldens -- and measured 2.8x SLOWER when applied eagerly, because
``hash_stim_counts`` materialises the stimulus x neuron connectivity where the
binomial draw is one call. It only pays combined with lazy extension, which is
tracked separately. See task #48.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.numpy_engine._seeding import (
    fnv1a_pair_seed, hash_stim_counts,
)

STIM_SIZE, P = 40, 0.05


def _sequential_tail(seed, count):
    """What the numpy engine does today."""
    return np.random.default_rng(seed).binomial(
        STIM_SIZE, P, size=count).astype(np.float32)


def _content_tail(count, brain_seed=42, stim="w0", area="L"):
    """What cuda_engine and torch_engine do today."""
    return hash_stim_counts(
        STIM_SIZE, 0, count, fnv1a_pair_seed(brain_seed, stim, area), P)


def test_engines_disagree_on_values():
    """The divergence is REAL. If this ever fails, they were unified -- good,
    but the goldens and the docstring above both need updating."""
    assert not np.array_equal(_sequential_tail(0, 64), _content_tail(64)), (
        "numpy and GPU stim init now agree; unify the docs and re-baseline")


def test_engines_agree_on_distribution():
    """Statistically interchangeable, which is what makes aggregate
    cross-engine comparison legitimate despite the value divergence."""
    n = 20000
    seq = _sequential_tail(1, n)
    con = _content_tail(n)
    expected = STIM_SIZE * P
    # Mean within 5% of the analytic value for both.
    assert abs(seq.mean() - expected) < 0.05 * expected, seq.mean()
    assert abs(con.mean() - expected) < 0.05 * expected, con.mean()
    # And within 5% of each other.
    assert abs(seq.mean() - con.mean()) < 0.05 * expected


def test_content_addressed_is_position_independent():
    """The property the numpy path lacks: a neuron's count does not depend on
    when, or in how many pieces, it was materialised."""
    whole = _content_tail(64)
    in_pieces = np.concatenate([
        hash_stim_counts(STIM_SIZE, a, b,
                         fnv1a_pair_seed(42, "w0", "L"), P)
        for a, b in ((0, 7), (7, 30), (30, 64))
    ])
    assert np.array_equal(whole, in_pieces)


@pytest.mark.parametrize("count", [1, 17, 256])
def test_content_addressed_is_deterministic_across_calls(count):
    assert np.array_equal(_content_tail(count), _content_tail(count))
