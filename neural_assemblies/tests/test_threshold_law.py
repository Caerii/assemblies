"""The threshold-law study's bars (PREREG_refraction_memory.md Amendment 17),
on synthetic sweeps: optima at 0.2 of the convergence-threshold form must pass,
optima that ignore it (a fixed beta) must fail, and a regime with no interior
optimum below the floor must fail T2 rather than pass it vacuously."""
import math

import pytest

from research.experiments import memory_threshold_law as tl

pytestmark = pytest.mark.requires_torch


def _window(value):
    return {"never": value <= 0, "upper_censored": False, "upper": value,
            "lower": None, "last_above": value}


def _observations(star, below_completes=True):
    cells = {}
    for spec in tl.plan(tl.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        base = tl.A14.get((n, k, p), 500.0)
        completes = spec["above_floor"] or below_completes
        sweep = {f"{b:g}": {"beta": b, "c_first_item": 4, "windows": {"complete_distinct": _window(
                     max(base * (1 - 0.5 * math.log(b / star(n, k, p)) ** 2), 0.0) if completes else 0.0)}}
                 for b in spec["betas"]}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "above_floor": spec["above_floor"],
                                   "theta": spec["theta"], "sweep": sweep}
    return {"cells": cells}


def test_optima_at_the_fraction_pass():
    bars = tl.evaluate(_observations(tl.beta_pred))["bars"]
    assert bars["TV"] and bars["T1"] and bars["T2"]


def test_a_fixed_beta_fails():
    bars = tl.evaluate(_observations(lambda n, k, p: 0.07))["bars"]
    assert not bars["T1"]


def test_no_completion_below_the_floor_fails_t2():
    bars = tl.evaluate(_observations(tl.beta_pred, below_completes=False))["bars"]
    assert bars["T1"] and not bars["T2"]


def test_grids_bracket_the_prediction():
    for n, k, p in tl.CELLS:
        grid = tl.betas(n, k, p)
        assert min(grid) >= tl.MIN_BETA and min(grid) < tl.beta_pred(n, k, p) < max(grid)
