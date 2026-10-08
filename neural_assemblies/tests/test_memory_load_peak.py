"""Amendment 41's bars (PREREG_refraction_memory.md), on synthetic records: a gain
that falls from the small area to the large passes; no peak at the small area
fails P1; a gain that does not fall fails P2; an early cliff fails P3; a peak
that persists at n/k = 200 fails P4."""
import math

import pytest

from research.experiments import memory_load_law as ml
from research.experiments import memory_load_peak as mp

pytestmark = pytest.mark.requires_torch


def _rows(n, k, p, rho50, width=1.06):
    rows = {}
    for r in mp.LADDER:
        lo, hi = rho50 / width, rho50 * width
        full = 1.0 if r <= lo else (0.0 if r >= hi else 1.0 - (r - lo) / (hi - lo))
        rows[str(int(round(r * ml.unit(n, k, p))))] = {"rho": r, "steps": [], "full": full}
    return rows


def _obs(gains=(1.6, 1.3, 1.1, 1.0), base=0.12):
    cells = {}
    for (n, k, p), g in zip(mp.CELLS, gains):
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "arms": {
            "0.5": {"tau": 0, "ladder": _rows(n, k, p, base)},
            "1.0": {"tau": 0, "ladder": _rows(n, k, p, base * g)}}}
    return {"cells": cells}


def test_a_falling_gain_passes():
    out = mp.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_no_peak_at_the_small_area_fails_p1():
    assert not mp.evaluate(_obs(gains=(1.1, 1.05, 1.0, 0.95)))["bars"]["P1"]


def test_a_gain_that_does_not_fall_fails_p2():
    assert not mp.evaluate(_obs(gains=(1.4, 1.4, 1.5, 1.5)))["bars"]["P2"]


def test_an_early_cliff_fails_p3():
    assert not mp.evaluate(_obs(base=0.08, gains=(1.6, 1.3, 1.1, 1.0)))["bars"]["P3"]


def test_a_peak_that_persists_fails_p4():
    assert not mp.evaluate(_obs(gains=(1.8, 1.6, 1.4, 1.3)))["bars"]["P4"]


def test_the_cells_are_new_differ_in_n_over_k_alone_and_sit_in_the_linear_regime():
    seen = {(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5), (8000, 120, 0.5),
            (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5), (15000, 50, 0.7), (20000, 200, 0.3),
            (4000, 400, 0.5), (4000, 400, 0.1), (8000, 400, 0.1), (4000, 200, 0.15),
            (6000, 90, 0.4), (12000, 80, 0.5), (6000, 40, 0.5), (18000, 60, 0.6), (24000, 80, 0.5),
            (21000, 70, 0.5), (16000, 80, 0.45), (8000, 80, 0.5), (12000, 70, 0.5)}
    assert not set(mp.CELLS) & seen and mp.SMOKE_CELL in seen
    ratios = [n / k for n, k, _ in mp.CELLS]
    assert ratios == sorted(ratios) and ratios[0] == 20 and ratios[-1] == 200
    for n, k, p in mp.CELLS:
        assert 34 <= k * p <= 36 and k * p >= 3 * math.log(n)
        assert 3.5 <= k * p / math.log(n) <= 4.5
