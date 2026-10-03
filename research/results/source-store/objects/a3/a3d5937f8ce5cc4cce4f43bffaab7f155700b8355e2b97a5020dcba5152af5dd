"""The onset study's bars (PREREG_refraction_memory.md Amendment 18), on
synthetic sweeps: a universal onset with dense load-assisted windows and a
sparse rise above the onset must pass; an onset at a fixed beta, a dense
onset that completes from the first item, and an onset at the grid's first
rate must each fail."""
import pytest

from research.experiments import memory_onset as on

pytestmark = pytest.mark.requires_torch


def _window(cap, lower):
    if cap <= 0:
        return {"never": True, "upper_censored": False, "upper": None, "lower": None}
    return {"never": False, "upper_censored": False, "upper": cap, "lower": lower,
            "last_above": cap}


def _observations(onset_of, *, dense_lower=200.0, start_at_grid=False):
    cells = {}
    for spec in on.plan(on.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        grid = spec["betas"]
        b_on = grid[0] if start_at_grid else min(b for b in grid if b >= onset_of(n, k, p))
        sweep = {}
        for b in grid:
            if b < b_on:
                cap, lower = 0.0, None
            elif p == 0.05:              # sparse: immediate, rising to 1.68 x onset
                cap, lower = 200.0 * min(b / b_on, 1.68) / (1 + max(b / b_on - 1.68, 0)), None
            else:                        # dense: load-assisted, best at the onset
                base = on.A17.get((n, k, p), 1500.0)
                cap = base / (b / b_on) ** 2
                lower = dense_lower if b == b_on else None
            sweep[f"{b:g}"] = {"beta": b, "windows": {"complete_distinct": _window(cap, lower)}}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "above_floor": spec["above_floor"],
                                   "theta": spec["theta"], "sweep": sweep}
    return {"cells": cells}


def _law(n, k, p):
    return 0.16 * on.tl.theta(n, k, p)


def test_a_universal_onset_passes_every_bar():
    bars = on.evaluate(_observations(_law))["bars"]
    assert all(bars.values()), bars


def test_an_onset_at_a_fixed_beta_fails_o1():
    assert not on.evaluate(_observations(lambda n, k, p: 0.06))["bars"]["O1"]


def test_a_dense_onset_from_the_first_item_fails_o2():
    assert not on.evaluate(_observations(_law, dense_lower=None))["bars"]["O2"]


def test_an_onset_at_the_grid_start_is_not_found():
    assert not on.evaluate(_observations(_law, start_at_grid=True))["bars"]["O1"]


def test_grids_are_fine_and_bracket_the_band():
    for n, k, p in on.CELLS:
        grid = on.betas(n, k, p)
        theta = on.tl.theta(n, k, p)
        assert min(grid) >= on.MIN_BETA
        assert min(grid) / theta <= on.FRACTION_BAND[0] and max(grid) / theta >= 0.29
        assert all(b / a < 1.07 for a, b in zip(grid, grid[1:]))
