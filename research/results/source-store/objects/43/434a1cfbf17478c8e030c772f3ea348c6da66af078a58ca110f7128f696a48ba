"""The recovery study's bars (PREREG_refraction_memory.md Amendment 29), on
synthetic records: a limit that peaks at an interior recovery time near n/k,
far above both ends, and grows as (n/k)^2, must pass; a monotone curve fails
B2; a best no better than the ends fails B1; an edge far from n/k fails B3."""
import math

import pytest

from research.experiments import memory_recovery as mr

pytestmark = pytest.mark.requires_torch


def _obs(*, shape="peak", law=0.34, edge_mult=2.0):
    cells = {}
    for spec in mr.plan(mr.CELLS):
        tile = spec["tile"]
        best = law * tile ** 2
        taus = {}
        for t in mr.TAUS:
            name = mr.tau_name(t)
            if shape == "flat":
                v = best
            elif t is None:
                v = 0.3 * best
            elif shape == "monotone":
                v = best * (1 - 0.5 * math.exp(-t / tile)) if t else 0.3 * best
            else:
                v = 0.3 * best if t == 0 else (best if t <= edge_mult * tile else 0.3 * best)
            taus[name] = {"curve": {}, "limit": v}
        cells[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = dict(spec, taus=taus)
    return {"cells": cells}


def test_an_interior_window_near_n_over_k_passes():
    bars = mr.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_a_curve_that_keeps_rising_fails_b2():
    assert not mr.evaluate(_obs(shape="monotone"))["bars"]["B2"]


def test_no_gain_over_the_ends_fails_b1():
    assert not mr.evaluate(_obs(shape="flat"))["bars"]["B1"]


def test_an_edge_far_above_n_over_k_fails_b3():
    assert not mr.evaluate(_obs(edge_mult=8.0))["bars"]["B3"]


def test_decay_zero_is_hebbian_and_none_is_cumulative():
    assert mr.decay_of(0) == 0.0 and mr.decay_of(None) is None
    assert mr.decay_of(16) == pytest.approx(math.exp(-1 / 16))
