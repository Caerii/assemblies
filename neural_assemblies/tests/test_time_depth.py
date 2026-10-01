"""The time-depth study's bars (PREREG_refraction_memory.md Amendment 15), on
synthetic sweeps: a memory whose capacity depends only on the per-item write
T ln(1 + beta) must pass the depth bars, and one whose best per-round rate
does not move with T must fail them."""
import math

import pytest

from research.experiments import memory_time_depth as td

pytestmark = pytest.mark.requires_torch


def _window(value):
    return {"never": value <= 0, "upper_censored": False, "upper": value,
            "lower": None, "last_above": value}


def _observations(capacity):
    cells = {}
    for spec in td.plan(td.CELLS):
        n, k, T = spec["n"], spec["k"], spec["T"]
        sweep = {f"{b:g}": {"beta": b, "c_first_item": 4,
                            "windows": {"complete_distinct": _window(capacity(n, k, T, b))}}
                 for b in spec["betas"]}
        cells[f"{n}/{k}/T{T}"] = {"n": n, "k": k, "T": T, "law_beta": spec["law_beta"],
                                  "sweep": sweep}
    return {"cells": cells}


def _write_only(n, k, T, beta):
    """Capacity a function of the per-item write alone, peaked at the law."""
    write = T * math.log1p(beta)
    best = 8 * math.log1p(td.beta_for_depth(8, k))
    scale = {(2000, 60): 492.0, (4000, 60): 1616.0}[(n, k)]
    return max(scale * (1 - 2.0 * (math.log(write / best)) ** 2), 0.0)


def test_grids_respect_the_saturation_floor_and_carry_beta_point_one():
    for T in td.DEPTHS:
        grid = td.betas(T, 60)
        assert min(grid) >= td.MIN_BETA and 0.1 in grid


def test_a_write_only_memory_passes_the_depth_bars():
    bars = td.evaluate(_observations(_write_only))["bars"]
    assert bars["DV"] and bars["D1"] and bars["D2"] and bars["D3"] and bars["D4"]


def test_a_per_round_optimum_fixed_in_t_fails_d3():
    def fixed(n, k, T, beta):
        scale = {(2000, 60): 492.0, (4000, 60): 1616.0}[(n, k)]
        return max(scale * (1 - 2.0 * math.log(beta / 0.07) ** 2), 0.0)
    bars = td.evaluate(_observations(fixed))["bars"]
    assert not bars["D3"] and not bars["D2"]
