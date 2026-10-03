"""The criticality study's bars (PREREG_refraction_memory.md Amendment 24),
on synthetic records: per-brain onsets that tighten with n around a converging
mean, and a read-out that slows at the onset, must pass; a spread that does
not shrink must fail F1, and a read-out that does not slow must fail F3."""
import random

import pytest

from research.experiments import memory_criticality as cr

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ensemble(values):
    return {"keys": SEEDS, "values": values, "mean": sum(values) / len(values)}


def _observations(sd_of, slows=True):
    cells = {}
    for spec in cr.plan(cr.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        theta, grid = spec["theta"], spec["betas"]
        rng = random.Random(n)
        onset = {s: 0.17 + rng.gauss(0, sd_of(n)) for s in SEEDS}
        done_at = {b: [1.0 if b / theta >= onset[s] else 0.0 for s in SEEDS] for b in grid}
        first = min(b for b in grid if sum(done_at[b]) / len(SEEDS) > 0.5)
        sweep = {}
        for b in grid:
            done = done_at[b]
            mean = sum(done) / len(done)
            settle = 20.0 if (slows and b == first) else 6.0
            cap = (0.0 if mean <= 0.5 else
                   1597.0 if (n, k, p) in cr.A18 and b == first else 1000.0)
            window = ({"never": True, "upper_censored": False, "upper": None, "lower": None}
                      if cap <= 0 else {"never": False, "upper_censored": False, "upper": cap,
                                        "lower": None, "last_above": cap})
            ensembles = {str(M): {"complete_distinct": _ensemble(done),
                                  "settle": _ensemble([settle] * len(SEEDS))} for M in (32, 64)}
            sweep[f"{b:g}"] = {"beta": b, "windows": {"complete_distinct": window},
                               "ensembles": ensembles}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "theta": theta, "sweep": sweep}
    return {"cells": cells}


def test_a_sharpening_transition_that_slows_passes():
    out = cr.evaluate(_observations(lambda n: 0.024 * (2000 / n) ** 0.5))
    assert out["bars"]["F1"] and out["bars"]["F2"] and out["bars"]["F3"], out["bars"]


def test_a_spread_that_does_not_shrink_fails_f1():
    assert not cr.evaluate(_observations(lambda n: 0.02))["bars"]["F1"]


def test_a_readout_that_does_not_slow_fails_f3():
    bars = cr.evaluate(_observations(lambda n: 0.024 * (2000 / n) ** 0.5, slows=False))["bars"]
    assert not bars["F3"]
