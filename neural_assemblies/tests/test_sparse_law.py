"""The sparse-law study's bars (PREREG_refraction_memory.md Amendment 16), on
synthetic sweeps: an optimum that follows the connectivity-noise law must
pass, and one that follows the plain fan-in law (no (1 - p)) must fail the
law and the dense-side bars."""
import math

import pytest

from research.experiments import memory_sparse_law as sl

pytestmark = pytest.mark.requires_torch


def _window(value):
    return {"never": value <= 0, "upper_censored": False, "upper": value,
            "lower": None, "last_above": value}


def _observations(star):
    cells = {}
    for spec in sl.plan(sl.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        base = sl.A14.get((n, k, p), 480.0 if k * p == 30 else 1150.0)
        sweep = {f"{b:g}": {"beta": b, "c_first_item": 4, "windows": {
                     "complete_distinct": _window(max(base * (1 - 0.5 * math.log(b / star(k, p)) ** 2), 0.0))}}
                 for b in spec["betas"]}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "beta_pred": spec["beta_pred"],
                                   "sweep": sweep}
    return {"cells": cells}


def test_the_noise_law_passes():
    bars = sl.evaluate(_observations(sl.beta_pred))["bars"]
    assert all(bars.values()), bars


def test_the_plain_fan_in_law_fails_s2_and_s3():
    def plain(k, p):
        return math.expm1(0.29 / math.sqrt(k * p / 2))
    bars = sl.evaluate(_observations(plain))["bars"]
    assert bars["S1"] and not bars["S2"] and not bars["S3"]


def test_grids_bracket_the_prediction_above_the_saturation_floor():
    for n, k, p in sl.CELLS:
        grid = sl.betas(k, p)
        assert min(grid) >= sl.MIN_BETA and min(grid) < sl.beta_pred(k, p) < max(grid)
