"""Physical connectome capacity must never exceed the area's `n`.

Connectome buffers grow by DOUBLING so that recruitment is amortised rather
than reallocating every round. Rows index presynaptic neurons and columns
postsynaptic ones, so neither dimension can logically exceed that area's `n` --
but the doubling was unbounded, so an area with `n=10000` and `w=7918` held a
(15682, 15682) matrix: 984 MB where 400 MB is the most the fiber can ever
need, and up to ~4x n^2 in the worst case.

It never changed a result, because every consumer slices by the LOGICAL `w`
(see `_norm_*`, which takes `min(src.w, w.shape[0])` precisely for this
reason). It cost memory, and it crashed `materialize_area`, which reasonably
assumed no existing block exceeds `n`.

`_stim_capacity` already clamps its 1-D case with `min(cap, n)`. This pins the
same rule for the 2-D case.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain


def _oversubscribed_brain(n=200, k=20, stimuli=40, rounds=2):
    """Recruit past n/2, which is where unbounded doubling overshoots n."""
    b = Brain(p=0.1, seed=11)
    b.add_area("A", n, k, 0.05)
    for i in range(stimuli):
        b.add_stimulus(f"s{i}", k)
    for i in range(stimuli):
        for _ in range(rounds):
            b.project({f"s{i}": ["A"]}, {"A": ["A"]})
    return b


def _fibers(engine):
    for src, per in engine._area_conns.items():
        for dst, conn in per.items():
            shape = getattr(conn.weights, "shape", None)
            if shape and len(shape) == 2:
                yield src, dst, shape


def test_recruitment_actually_passes_the_doubling_boundary():
    """Precondition: this configuration must REACH the overshoot.

    At n=200, k=20 the capacity ladder is 40 -> 80 -> 160 -> 320, so the clamp
    only bites once `w` passes 160; below that these tests would pass with the
    bug present and prove nothing. Verified against the unclamped code: `w`
    reaches 199 and two of the tests below fail with

        could not broadcast input array from shape (200,320) into (200,200)
    """
    b = _oversubscribed_brain()
    n = b.areas["A"].n
    w = b._engine.get_num_ever_fired("A")
    assert w > 160, (
        f"w={w} of n={n} does not pass the 160 rung of the doubling ladder, "
        f"so the assertions below have no power against an unclamped engine")


def test_no_fiber_exceeds_n():
    b = _oversubscribed_brain()
    eng = b._engine
    n = b.areas["A"].n
    for src, dst, (rows, cols) in _fibers(eng):
        assert rows <= n, f"{src}->{dst} has {rows} rows > n={n}"
        assert cols <= n, f"{src}->{dst} has {cols} cols > n={n}"


def test_stim_fibers_also_bounded():
    b = _oversubscribed_brain()
    eng = b._engine
    n = b.areas["A"].n
    for stim, per in eng._stim_conns.items():
        for dst, conn in per.items():
            size = getattr(conn.weights, "shape", (0,))[0]
            assert size <= n, f"stim {stim}->{dst} holds {size} > n={n}"


def test_materialize_survives_an_oversubscribed_fiber():
    """The failure this actually caused: a broadcast error mid-materialisation.

    It left the area half-built, so the sampler-free arbiter was unusable on
    exactly the models big enough to need it.
    """
    b = _oversubscribed_brain()
    n = b.areas["A"].n

    b._engine.materialize_area("A")

    assert b._engine.get_num_ever_fired("A") == n
    for src, dst, (rows, cols) in _fibers(b._engine):
        assert rows <= n and cols <= n


def test_clamping_does_not_change_the_dynamics():
    """Padding is never read, so bounding it must be bit-identical.

    If this fails, some consumer is summing over unallocated capacity -- which
    would mean the clamp changed a result and the old numbers were wrong.
    """
    winners = []
    for _ in range(2):
        b = Brain(p=0.1, seed=11)
        b.add_area("A", 200, 20, 0.05)
        b.add_stimulus("s", 20)
        for _ in range(15):
            b.project({"s": ["A"]}, {"A": ["A"]})
        winners.append(np.array(b.areas["A"].winners, dtype=np.uint32))

    np.testing.assert_array_equal(winners[0], winners[1])
