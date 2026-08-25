"""The two homeostatic-scaling research knobs: default-identical, and real.

WHY THIS EXISTS. `synaptic_scaling_scope` and `synaptic_scaling_setpoint` were
added to answer a mechanism question -- what merges assemblies under substrate
C -- and both default to the pre-existing behaviour. A knob like that has
exactly two ways to be wrong, and a study built on it inherits both:

  * it silently changes the DEFAULT, so every result recorded before it is now
    incomparable with every result after it; or
  * it silently does NOTHING, so a sweep comes back flat and the flatness is
    read as a fact about the mechanism rather than a fact about the knob.

This project has been bitten by the second shape more than once -- a selector
over an unfilled field is a silent no-op ([[dormant-selectors-hide-whole-
phases]]), and an unmoved metric can mean saturated rather than unwired
([[unmoved-metric-can-mean-saturated-error]]). So each knob is pinned twice:
byte-identical when unset, and demonstrably different when set.

These tests pin the MECHANISM, not the study's numbers.
"""
from __future__ import annotations

import random

import numpy as np

from neural_assemblies.core.brain import Brain

N, K, BETA, P, T, M = 400, 20, 0.10, 0.5, 6, 3
AREA = "A"


def _build(seed, scope=None, setpoint=None):
    random.seed(seed)
    np.random.seed(seed)
    b = Brain(p=P, seed=seed, engine="numpy_sparse",
              recurrent_projection=True, synaptic_scaling=True)
    b.add_area(AREA, N, K, BETA)
    eng = b._engine_for(b.areas[AREA])
    if scope is not None:
        eng.synaptic_scaling_scope = scope
    if setpoint is not None:
        eng.synaptic_scaling_setpoint = setpoint
    return b, eng


def _train(b, seed):
    winners = []
    for i in range(M):
        s = f"s{i}"
        b.add_stimulus(s, K)
        b.inhibit_areas([AREA])
        for _ in range(T):
            b.project({s: [AREA]}, {AREA: [AREA]})
        winners.append(tuple(sorted(int(x) for x in b.areas[AREA].winners)))
    return winners


def _weights(eng):
    conn = eng._area_conns[AREA][AREA]
    rows = int(eng._areas[AREA].w)
    w = np.asarray(conn.weights)
    return np.asarray(w[:rows, :min(rows, w.shape[1])], dtype=np.float64)


def _run(seed, **kw):
    b, eng = _build(seed, **kw)
    return _train(b, seed), _weights(eng)


class TestDefaultsAreUnchanged:
    """Setting a knob to its documented default must be BYTE-identical."""

    def test_scope_default_is_byte_identical(self):
        base_w, base_m = _run(7)
        knob_w, knob_m = _run(7, scope="winners")
        assert base_w == knob_w
        assert np.array_equal(base_m, knob_m)

    def test_setpoint_default_is_byte_identical(self):
        base_w, base_m = _run(7)
        knob_w, knob_m = _run(7, setpoint="population")
        assert base_w == knob_w
        assert np.array_equal(base_m, knob_m)

    def test_unset_engine_has_no_attribute_to_trip_over(self):
        """The read is a getattr with a default, so an engine that predates the
        knobs -- an unpickled one, say -- must behave as the default."""
        _, eng = _build(7)
        assert not hasattr(eng, "synaptic_scaling_scope")
        assert not hasattr(eng, "synaptic_scaling_setpoint")


class TestKnobsActuallyBite:
    """A knob that changes nothing turns a sweep into a silent no-op."""

    def test_scope_all_rescales_columns_that_never_won(self):
        """The whole point of scope="all": normalize the CANDIDATES, i.e. the
        columns k-WTA is about to compare, not only the winners it already
        produced. So some column that never won must come out rescaled."""
        _, w_win = _run(7)
        _, w_all = _run(7, scope="all")
        # Under "winners" a never-won column keeps its raw unit weights, so a
        # substantial exactly-1.0 population survives; under "all" every
        # materialized column has been touched, so essentially none does.
        # The bar is on the CONTRAST, not on either level: what fraction of
        # columns never win is a property of the toy's size, not of the knob.
        frac_win = float(np.mean(w_win[w_win != 0.0] == 1.0))
        frac_all = float(np.mean(w_all[w_all != 0.0] == 1.0))
        assert frac_win > 0.1, frac_win
        assert frac_all < 0.01, (frac_win, frac_all)

    def test_setpoint_degree_pins_each_column_to_its_own_mass(self):
        """population -> every touched column shares one setpoint;
        degree -> each column's mass returns to its own in-degree, so the
        spread of column masses is strictly WIDER."""
        _, w_pop = _run(7, scope="all")
        _, w_deg = _run(7, scope="all", setpoint="degree")
        cv_pop = float(np.std(w_pop.sum(axis=0)) / np.mean(w_pop.sum(axis=0)))
        cv_deg = float(np.std(w_deg.sum(axis=0)) / np.mean(w_deg.sum(axis=0)))
        assert cv_deg > cv_pop, (cv_pop, cv_deg)

    def test_the_two_knobs_are_independent(self):
        _, a = _run(7, scope="all")
        _, b = _run(7, setpoint="degree")
        _, c = _run(7, scope="all", setpoint="degree")
        assert not np.array_equal(a, b)
        assert not np.array_equal(a, c)
        assert not np.array_equal(b, c)


def test_scaling_composes_with_norm_init():
    """The measured repair is BOTH mechanisms at once, so the combination has
    to be constructible and has to actually run both. They were treated as
    rival substrates for months; nothing in the engine required that."""
    random.seed(3)
    np.random.seed(3)
    b = Brain(p=P, seed=3, engine="numpy_sparse", recurrent_projection=True,
              norm_init=True, synaptic_scaling=True)
    b.add_area(AREA, N, K, BETA)
    eng = b._engine_for(b.areas[AREA])
    _train(b, 3)
    assert eng.synaptic_scaling is True
    assert getattr(eng, "norm_init", False) is True
    w = _weights(eng)
    # norm_init leaves storage on the unit scale (its 1/d_j is a READ-time
    # scale), so the proof that scaling also ran is that the stored weights are
    # no longer unit-valued.
    assert float(np.mean(w[w != 0.0] == 1.0)) < 0.5
