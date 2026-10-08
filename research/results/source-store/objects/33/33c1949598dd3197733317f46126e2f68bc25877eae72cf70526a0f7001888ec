"""Amendment 39's bars (PREREG_refraction_memory.md), on synthetic records: the
predicted gain passes; no gain fails T1; an early cliff under the rule fails T2
and T3; a slow decline fails T4."""
import math

import pytest

from research.experiments import memory_load_law as ml
from research.experiments import memory_load_tau as mt

pytestmark = pytest.mark.requires_torch


def _ladder(n, k, p, rho50, width):
    rows = {}
    for r in mt.LADDER:
        lo, hi = rho50 / width, rho50 * width
        full = 1.0 if r <= lo else (0.0 if r >= hi else 1.0 - (r - lo) / (hi - lo))
        rows[str(int(round(r * ml.unit(n, k, p))))] = {"rho": r, "steps": [], "full": full}
    return {"ladder": rows}


def _obs(ruled=0.115, control=0.088, width=1.08):
    return {"cells": {f"{n}/{k}/{p:g}": {"n": n, "k": k, "p": p, "tau": {
        str(mt.CONTROL): _ladder(n, k, p, control, 1.08),
        str(mt.rule(n, k)): _ladder(n, k, p, ruled, width)}} for n, k, p in mt.CELLS}}


def test_the_predicted_gain_passes():
    out = mt.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_no_gain_fails_t1():
    assert not mt.evaluate(_obs(ruled=0.09, control=0.088))["bars"]["T1"]


def test_an_early_cliff_under_the_rule_fails_t2_and_t3():
    bars = mt.evaluate(_obs(ruled=0.085, control=0.06))["bars"]
    assert not bars["T2"] and not bars["T3"]


def test_a_slow_decline_fails_t4():
    assert not mt.evaluate(_obs(width=1.6))["bars"]["T4"]


def test_a_missing_arm_fails_every_bar():
    obs = _obs()
    for cell in obs["cells"].values():
        cell["tau"].pop(str(mt.CONTROL))
    assert not any(mt.evaluate(obs)["bars"].values())


def test_the_cells_are_new_in_regime_and_the_rule_is_inside_the_probed_window():
    run = {(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5),
           (8000, 120, 0.5), (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5),
           (15000, 50, 0.7), (20000, 200, 0.3), *ml.CELLS, *ml.REPORTED}
    from research.experiments import memory_load_drift as md
    run |= set(md.CELLS)
    assert not set(mt.CELLS) & run and mt.SMOKE_CELL in run
    for n, k, p in mt.CELLS:
        assert k * p >= 1.15 * 3 * math.log(n)
        assert 0.43 <= mt.rule(n, k) / (n / k) <= 0.85       # the probe's good window
