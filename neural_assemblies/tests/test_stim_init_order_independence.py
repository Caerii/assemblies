"""stim->area init must be a fixed fact of (stimulus, neuron), not of draw order.

Door 6 of the family in [[content-addressed-synapse-init]]: *randomness
standing in for a fixed structural fact must be keyed on that fact, not drawn
from a stream.* Area->area init was fixed (`hash_area_weights`), and the torch
and cuda engines had already moved stim->area onto `hash_stim_counts` -- but
numpy_sparse was still calling `self._rng.binomial(size, p, add_len)`. Mixed
disciplines inside one engine, which is worse than either.

WHY IT SURVIVED SO LONG. A FIRING stimulus never takes the background path:
its weights come from `compute_input_splits`. So an ordinary projection is
unaffected and no assembly moved -- `test_assembly_is_unmoved` below still
passes against the OLD engine. It bit three narrower places:

  * `materialize_area`, so the sampler-free ARBITER carried an order-dependent
    stimulus component -- while being the thing other results are checked
    against;
  * the drive of a never-fired neuron, which exact-drive needs;
  * numpy-vs-torch parity, since the two engines drew differently.

Measured before the fix: an unused extra stimulus left only 21.3% of a
materialized stim vector identical. After: 100%.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain

N, K, P, BETA, SEED = 2000, 45, 0.05, 0.05, 42


def _brain(ghost: bool):
    """Identical brains but for one stimulus that is declared and never fires."""
    b = Brain(p=P, seed=SEED, engine="numpy_sparse", norm_init=False)
    if ghost:
        b.add_stimulus("ghost", K)
    b.add_stimulus("s", K)
    b.add_area("A", N, K, BETA)
    return b


def _materialized_stim_vector(ghost: bool):
    b = _brain(ghost)
    b._engine.materialize_area("A")
    return np.asarray(b._engine._stim_conns["s"]["A"].weights, dtype=np.float64)


def test_unused_stimulus_does_not_move_stim_weights():
    """The assertion that fails against a stream-drawn background."""
    v0 = _materialized_stim_vector(False)
    v1 = _materialized_stim_vector(True)

    assert len(v0) == N
    np.testing.assert_array_equal(
        v0, v1,
        err_msg="declaring an unused stimulus moved the stim->area weights, "
                "so the background fill is still consuming a shared stream")


def test_the_distribution_is_unchanged():
    """Content addressing must re-key the draw, not replace it with something else.

    Each entry counts, over the stimulus' `K` fibers, how many connect to that
    neuron -- so the mean is `K * p` either way. If this drifts, the hash is
    not reproducing Binomial(K, p) and every stim-driven golden is wrong for a
    new reason.
    """
    v = _materialized_stim_vector(False)
    assert v.mean() == pytest.approx(K * P, rel=0.15), (
        f"mean {v.mean():.3f} is not Binomial({K}, {P}) = {K * P}")
    assert v.min() >= 0
    assert v.max() <= K


def test_repeated_construction_is_identical():
    """Same construction, same weights -- the diagonal case."""
    np.testing.assert_array_equal(_materialized_stim_vector(False),
                                  _materialized_stim_vector(False))


def test_assembly_is_unmoved():
    """Documents the blast radius, and why this hid.

    A firing stimulus is filled from `compute_input_splits`, never from the
    background draw, so ordinary projection was already order-independent.
    This passes against the OLD engine too -- it is here to record that the
    fix is NOT expected to move assemblies, so if it ever starts failing,
    something else changed.
    """
    from neural_assemblies.assembly_calculus.ops import overlap, project

    a0 = project(_brain(False), "s", "A", rounds=10)
    a1 = project(_brain(True), "s", "A", rounds=10)
    assert overlap(a0, a1) == 1.0
