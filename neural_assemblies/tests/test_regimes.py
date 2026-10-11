"""The regimes study's bars (PREREG_refraction_memory.md Amendments 19-20),
on synthetic sweeps: capacity flat in k and linear in n below the floor and
falling with k above it, with recognition outrunning recall as fan-in falls,
must pass; an (n/k)^2 law must fail X1, and a constant recognition-to-recall
ratio must fail R1 and R3."""
import pytest

from research.experiments import memory_lib as lib
from research.experiments import memory_regimes as rg

pytestmark = pytest.mark.requires_torch


def _window(cap):
    if cap <= 0:
        return {"never": True, "upper_censored": False, "upper": None, "lower": None}
    return {"never": False, "upper_censored": False, "upper": cap, "lower": None,
            "last_above": cap}


def _observations(distinct_of, ratio_of):
    cells = {}
    for spec in rg.plan(rg.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        grid = spec["betas"]
        star = grid[len(grid) * 2 // 3]
        best = distinct_of(n, k, p)
        sweep = {}
        for b in grid:
            d = best if b == star else best * 0.5
            r1 = (100.0 if best == 0 else d * ratio_of(n, k, p)) if b == star else 50.0
            sweep[f"{b:g}"] = {"beta": b, "windows": {"complete_distinct": _window(d),
                                                       "rank1": _window(r1)}}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "above_floor": spec["above_floor"],
                                   "theta": spec["theta"], "sweep": sweep}
    return {"cells": cells}


def _two_regimes(n, k, p):
    if p == 0.01:
        return 0.0 if k * p < 1.5 else 50.0 * k * p
    if lib.above_floor(n, k, p):
        return 1600.0 * (56 / k) ** 0.5
    if (n, k, p) == (4000, 40, 0.5):
        return rg.A17[(4000, 40, 0.5)]
    return 0.36 * n


def _diverging(n, k, p):
    return 60.0 / (k * p)


def test_two_regimes_with_diverging_recognition_pass():
    out = rg.evaluate(_observations(_two_regimes, _diverging))
    assert all(out["bars"].values()), out["bars"]


def test_an_nk_squared_law_fails_x1():
    out = rg.evaluate(_observations(lambda n, k, p: 0.4 * (n / k) ** 2, _diverging))
    assert not out["bars"]["X1"]


def test_a_ratio_that_does_not_grow_as_fan_in_falls_fails_r1_and_r3():
    # rising with k p, not constant: a constant ratio is computed as
    # (d * r) / d, whose last bit wobbles, and Spearman would rank the wobble
    bars = rg.evaluate(_observations(_two_regimes, lambda n, k, p: 2.0 + k * p / 100))["bars"]
    assert not bars["R1"] and not bars["R3"]


def test_spearman_handles_ties_and_order():
    assert rg.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert rg.spearman([1, 2, 3, 4], [1, 1, 2, 2]) > 0.8
