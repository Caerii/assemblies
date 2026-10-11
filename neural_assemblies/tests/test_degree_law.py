"""The degree-law study's bars (PREREG_refraction_memory.md Amendment 21), on
synthetic sweeps: capacities on the in-degree law must pass; an (n/k)^2 law
must fail the prediction (D1) and the equal-degree agreement (D2); a law
linear in d must fail the exponent (D3)."""
import pytest

from research.experiments import memory_lib as lib
from research.experiments import memory_degree_law as dl

pytestmark = pytest.mark.requires_torch


def _window(cap):
    if cap <= 0:
        return {"never": True, "upper_censored": False, "upper": None, "lower": None}
    return {"never": False, "upper_censored": False, "upper": cap, "lower": None,
            "last_above": cap}


def _observations(law):
    cells = {}
    for spec in dl.plan(dl.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        best = dl.INSTRUMENT.get((n, k, p)) or law(n, k, p)
        grid = spec["betas"]
        sweep = {f"{b:g}": {"beta": b, "windows": {"complete_distinct": _window(
            best if i == len(grid) // 3 else best * 0.6)}} for i, b in enumerate(grid)}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "d": spec["d"],
                                   "predicted": spec["predicted"], "theta": spec["theta"],
                                   "sweep": sweep}
    return {"cells": cells}


def test_the_degree_law_passes():
    out = dl.evaluate(_observations(lambda n, k, p: 1.05 * dl.predicted(n, k, p)))
    assert all(out["bars"].values()), out["bars"]
    assert abs(out["gamma"] - dl.GAMMA) < 0.02


def test_an_nk_squared_law_fails_d1_and_d2():
    bars = dl.evaluate(_observations(lambda n, k, p: 0.4 * (n / k) ** 2))["bars"]
    assert not bars["D1"] and not bars["D2"]


def test_a_law_linear_in_d_fails_d3():
    bars = dl.evaluate(_observations(lambda n, k, p: 1.2 * n * p))["bars"]
    assert not bars["D3"]


def test_every_grid_brackets_the_onset_band():
    for n, k, p in dl.CELLS:
        grid = dl.betas(n, k, p)
        theta = lib.theta(n, k, p)
        assert min(grid) / theta <= 0.16 and max(grid) / theta >= 0.29 and min(grid) >= dl.MIN_BETA
