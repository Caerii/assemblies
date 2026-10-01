"""The write-strength sweep's search and bars (PREREG_refraction_memory.md
Amendment 10), on synthetic curves: the adaptive search must bracket a
crossing it cannot see in advance, and the evaluator must refuse a sweep whose
ceiling does not move with write strength."""
import pytest

from research.experiments import memory_write_strength as ws
from research.experiments import memory_pattern_efficiency as pe

pytestmark = pytest.mark.requires_torch


def _step(m_star):
    def measure(M):
        value = 1.0 if M < m_star else 0.0
        return {"rank1": [value] * 3, "complete": [value] * 3}
    return measure


@pytest.mark.parametrize("m_star", [20, 300, 5000, 90000])
def test_adaptive_search_brackets_the_crossing(m_star):
    cache = {}
    ws.adaptive_ceiling(_step(m_star), "rank1", 1024, 1 << 17, cache)
    ceiling = ws.ceilings(cache, [1, 2, 3])["rank1"]
    assert not ceiling["censored"] and not ceiling["below_at_first"]
    assert ceiling["lo"] < m_star <= ceiling["hi"]
    assert ceiling["hi"] / ceiling["lo"] <= 2


def test_adaptive_search_reports_a_censored_curve():
    cache = {}
    ws.adaptive_ceiling(_step(10 ** 9), "rank1", 1024, 4096, cache)
    assert ws.ceilings(cache, [1, 2, 3])["rank1"]["censored"]


def _sweep(shape, n, k):
    sweep = {}
    for c in ws.COUNTS:
        cache = {}
        ws.adaptive_ceiling(_step(shape(c) * (n / k) ** 2), "rank1", 64, 1 << 17, cache)
        ws.adaptive_ceiling(_step(shape(c) * (n / k) ** 2), "complete", 64, 1 << 17, cache)
        sweep[str(c)] = {"ceilings": ws.ceilings(cache, [1, 2, 3])}
    return sweep


def _observations(shape):
    return {"cells": {f"{n}/{k}": {"n": n, "k": k, "c_model": 5 if k == 60 else 6,
                                   "sweep": _sweep(shape, n, k)}
                      for n, k in pe.IN_REGIME}}


def test_a_u_shaped_sweep_passes_the_shape_bars():
    def u(c):
        return {1: 3.0, 2: 2.0, 3: 1.4, 4: 1.0, 5: 0.8, 6: 0.7, 8: 0.6, 11: 0.6,
                16: 0.7, 23: 0.9, 32: 1.2}[c]
    bars = ws.evaluate(_observations(u), random_anchors={})["bars"]
    assert bars["WS-1"] and bars["WS-2"]


def test_a_flat_sweep_fails_the_shape_bars():
    bars = ws.evaluate(_observations(lambda c: 0.8), random_anchors={})["bars"]
    assert not bars["WS-1"] and not bars["WS-2"]
