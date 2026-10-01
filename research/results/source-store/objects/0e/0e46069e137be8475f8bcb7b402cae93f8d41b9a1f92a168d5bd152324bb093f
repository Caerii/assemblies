"""The learning-rate transfer study's estimator and bars (PREREG_refraction_memory.md
Amendment 12), on synthetic sweeps: the optimum estimator must find a known
vertex and refuse an edge maximum, and the bars must PASS when the optimum
scales as 1/sqrt(k p) and FAIL when it stays put (the standard
parameterisation's prediction)."""
import math

import pytest

from research.experiments import memory_learning_rate as lr
from research.experiments import memory_pattern_efficiency as pe

pytestmark = pytest.mark.requires_torch


def test_optimum_finds_the_vertex_and_refuses_an_edge():
    betas = list(lr.BETAS)
    peak = 0.06
    values = [1000 - 300 * (math.log(b) - math.log(peak)) ** 2 for b in betas]
    assert abs(lr.optimum(betas, values) / peak - 1) < 0.05
    assert lr.optimum(betas, [float(i) for i in range(len(betas))]) is None
    assert lr.optimum(betas, [0.0] * len(betas)) is None


def test_capacity_reads_never_and_censored_windows():
    assert lr.capacity({"never": True}) == 0.0
    assert lr.capacity({"never": False, "upper_censored": True, "last_above": 512,
                        "upper": None}) == 512
    assert lr.capacity({"never": False, "upper_censored": False, "upper": 300.0}) == 300.0


def _observations(star):
    """Completion capacity peaked at star(n, k), larger at larger k p."""
    cells = {}
    for (n, k) in lr.CELLS:
        sweep = {}
        for beta in lr.BETAS:
            top = (n / k) ** 2 * (0.3 + 0.002 * k)
            value = max(top * (1 - 0.4 * (math.log(beta) - math.log(star(n, k))) ** 2), 0.0)
            window = {"never": value <= 0, "upper_censored": False, "upper": value,
                      "lower": None, "last_above": value}
            rank = {"never": False, "upper_censored": False,
                    "upper": (lr.CELLS[(n, k)] or 400.0) if beta == 0.1 else 500.0,
                    "lower": None, "last_above": 0}
            sweep[f"{beta:g}"] = {"beta": beta, "c_first_item": 5,
                                  "windows": {"complete": window, "rank1": rank}}
        cells[f"{n}/{k}"] = {"n": n, "k": k, "sweep": sweep}
    return {"cells": cells}


def _distinct_observations(star):
    cells = {}
    for (n, k) in lr.DISTINCT_CELLS:
        sweep = {}
        for beta in lr.DISTINCT_BETAS:
            top = (n / k) ** 2 * (0.3 + 0.002 * k)
            value = max(top * (1 - 0.4 * (math.log(beta) - math.log(star(n, k))) ** 2), 0.0)
            window = {"never": value <= 0, "upper_censored": False, "upper": value,
                      "lower": None, "last_above": value}
            rank = {"never": False, "upper_censored": False,
                    "upper": (lr.DISTINCT_CELLS[(n, k)] or 400.0) if beta == 0.1 else 500.0,
                    "lower": None, "last_above": 0}
            sweep[f"{beta:g}"] = {"beta": beta, "c_first_item": 5,
                                  "windows": {"complete_distinct": window, "rank1": rank}}
        cells[f"{n}/{k}"] = {"n": n, "k": k, "sweep": sweep}
    return {"cells": cells}


def test_distinct_bars_pass_on_the_predicted_law_and_fail_off_it():
    assert 0.1 in lr.DISTINCT_BETAS and len(lr.DISTINCT_BETAS) == 11

    def law(n, k):
        return math.expm1(lr.GAMMA / math.sqrt(k * pe.P / 2))
    bars = lr.evaluate_distinct(_distinct_observations(law))["bars"]
    assert all(bars.values()), bars
    linear = lr.evaluate_distinct(_distinct_observations(
        lambda n, k: math.expm1(8.7 / (k * pe.P / 2))))["bars"]
    assert not linear["U2"] and not linear["U3"]
    fixed = lr.evaluate_distinct(_distinct_observations(lambda n, k: 0.06))["bars"]
    assert not fixed["U2"] and not fixed["U3"]


def test_a_fan_in_scaled_optimum_passes_and_a_fixed_one_fails():
    def scaled(n, k):
        return math.expm1(0.5 / math.sqrt(k * pe.P / 2))
    bars = lr.evaluate(_observations(scaled))["bars"]
    assert bars["TV"] and bars["T1"] and bars["T2"] and bars["T3"] and bars["T4"]
    fixed = lr.evaluate(_observations(lambda n, k: 0.06))["bars"]
    assert fixed["T1"] and not fixed["T2"] and not fixed["T3"]
