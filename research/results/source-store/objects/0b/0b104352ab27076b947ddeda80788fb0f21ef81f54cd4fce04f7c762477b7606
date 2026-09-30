"""`materialize_area` must wire stimulus fibers, not just area fibers.

Both stim-expansion helpers read their ``stim_names`` argument as "stimuli
FIRING on this step" and deliberately leave those weights at zero, because
`_expand_connectomes` fills them afterwards from each new winner's own afferent
split. Every other stimulus takes the background ``binomial(stim_size, p)``
draw.

`materialize_area` used to pass every stimulus CONNECTED to the area, so all of
them took the firing branch -- and nothing fires during materialization, so
nothing ever filled them. The area fiber was wired correctly the whole time,
which is what made it look fine:

    stimA->A : shape=(1000,)     nonzero=0                      <- silent
    A->A     : shape=(1000,1000) density=0.0988 (p=0.0991)      <- correct

A materialized area therefore received zero stimulus drive, took the
"Zero signal -> preserve current assembly" early return in `project_into`, and
saved an EMPTY winner set every round -- 52 snapshots of length 0 on the merge
protocol. It went unnoticed because the protocol this method was built for (the
NEMO coin) seeds a k-subset and drives it RECURRENTLY, with no stimulus at all.

WHY IT IS WORTH A FILE.  Materializing is the route to running this engine with
no candidate sampling: at ``w == n`` there are no unmaterialized neurons, so
`sample_new_winner_inputs` never runs and k-WTA sees exact drive. Measured on
the merge protocol, support in units of k (distinct winners over 50 rounds):

    n, k        explicit    materialized    sampled
    1000, 32       6.1          5.9           8.9
    2000, 45       6.1          7.2           8.2
    4000, 63       6.9          7.2          10.8

Materialized sparse reproduces the explicit engine; the sampled path is
systematically high. That comparison is only possible if this works.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import _sparse as SP

N, K, P, BETA = 1000, 32, 0.0991, 0.05


def _brain():
    b = Brain(P, engine="numpy_sparse", norm_init=False, seed=0,
              save_winners=True)
    b.add_stimulus("stimA", K)
    b.add_stimulus("stimB", K)
    b.add_area("A", N, K, BETA)
    return b


@pytest.fixture
def _restore_fastpath():
    before = SP._env_stim_fastpath()
    yield
    SP.set_stim_fastpath(before)


@pytest.mark.parametrize("fastpath", [True, False])
def test_stim_fibers_are_wired_after_materialize(fastpath, _restore_fastpath):
    """Every connected stimulus gets its background draw, not zeros.

    Asserts the MEAN as well as the length: a fiber resized to n and left at
    zero has exactly the right shape, which is why the defect survived every
    shape-based check.
    """
    SP.set_stim_fastpath(fastpath)
    b = _brain()
    eng = b._engine_for(b.areas["A"])
    eng.materialize_area("A")

    for stim in ("stimA", "stimB"):
        w = np.asarray(eng._stim_conns[stim]["A"].weights)
        assert len(w) == N
        assert int((w != 0).sum()) > 0, (
            f"{stim}->A materialized to {N} zeros -- the area will read zero "
            f"drive and preserve an empty assembly forever")
        # Binomial(K, P) per neuron.
        assert w.mean() == pytest.approx(K * P, rel=0.15)


def test_area_fiber_was_never_the_broken_half():
    """Pin the half that always worked, so a fix cannot regress it."""
    b = _brain()
    eng = b._engine_for(b.areas["A"])
    eng.materialize_area("A")
    aa = eng._area_conns["A"]["A"].weights
    aa = np.asarray(aa.toarray() if hasattr(aa, "toarray") else aa)
    assert aa.shape == (N, N)
    assert (aa != 0).mean() == pytest.approx(P, rel=0.1)


def test_materialized_area_elects_a_real_assembly():
    """The observable symptom, asserted directly.

    `project_into` returns EARLY when total drive is zero, preserving whatever
    the area last held -- so the failure surfaces as an empty (or stale)
    assembly rather than an exception.
    """
    b = _brain()
    eng = b._engine_for(b.areas["A"])
    eng.materialize_area("A")
    for _ in range(5):
        b.project({"stimA": ["A"]}, {"A": ["A"]})

    winners = np.asarray(b.areas["A"].winners)
    assert len(winners) == K, (
        f"materialized area elected {len(winners)} winners, expected {K}")
    saved = b.areas["A"].saved_winners
    assert all(len(s) == K for s in saved[-5:]), (
        f"empty winner snapshots: {[len(s) for s in saved]}")


@pytest.mark.parametrize("fastpath", [True, False])
def test_materialized_run_is_independent_of_the_candidate_sampler(
        fastpath, _restore_fastpath, monkeypatch):
    """At w == n the sampler must never be consulted.

    This is the property that makes a materialized run usable as an arbiter for
    the sampler itself: if it still called into `sample_new_winner_inputs`, the
    comparison would be circular.
    """
    SP.set_stim_fastpath(fastpath)
    b = _brain()
    eng = b._engine_for(b.areas["A"])
    eng.materialize_area("A")

    offered = []
    orig = eng._sparse_sim.sample_new_winner_inputs

    def spy(*a, **kw):
        out = orig(*a, **kw)
        offered.append(int(np.asarray(out).size))
        return out

    monkeypatch.setattr(eng._sparse_sim, "sample_new_winner_inputs", spy)
    for _ in range(5):
        b.project({"stimA": ["A"]}, {"A": ["A"]})

    # It is still CALLED -- the call site is unconditional -- but it must offer
    # nothing: `k_eff = min(k, max(0, (n - w) - 1))` is 0 once w == n, which is
    # the graceful-saturation branch. Assert what matters (no candidate enters
    # the k-WTA), not whether the function was entered.
    assert offered and all(c == 0 for c in offered), (
        f"the sampler offered candidates {offered} on a fully materialized "
        f"area, so a 'no-sampler' run is not actually sampler-free and cannot "
        f"arbitrate the sampler")
    assert int(eng._areas["A"].w) == N
