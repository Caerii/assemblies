"""The sequence-length study's bars (PREREG_refraction_memory.md Amendment
27), on synthetic records: Hebbian cliffs at 0.09 p (n/k)^2 with refraction
extending them must pass; a law constant off the band fails L1; a gradual
Hebbian decline fails L3; refraction that does not help fails L4."""
import pytest

from research.experiments import memory_sequence_length as sl

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(v):
    return {"keys": SEEDS, "values": [v] * len(SEEDS), "mean": v}


def _curve(limit, *, gradual=False):
    out = {}
    for L in sl.LADDER:
        if gradual:
            v = max(0.0, min(1.0, 1.0 - (L - limit / 4) / (2 * limit)))
        else:
            v = 1.0 if L <= limit / 1.2 else (0.0 if L >= limit * 1.2 else 0.5)
        out[str(L)] = _ens(v)
        if L > 4 * limit:
            break
    return out


def _obs(*, constant=0.09, refr=3.0, gradual=False):
    cells = {}
    for n, k, p in sl.CELLS:
        lh = constant * p * (n / k) ** 2
        heb = {"s": 0.0, "curve": _curve(lh, gradual=gradual), "limit": lh}
        ref = {"s": 0.5, "curve": _curve(refr * lh), "limit": refr * lh}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "arms": {"hebbian": heb, "refracted": ref}}
    return {"cells": cells}


def test_a_willshaw_cliff_extended_by_refraction_passes():
    bars = sl.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_a_law_constant_off_the_band_fails_l1():
    assert not sl.evaluate(_obs(constant=0.2))["bars"]["L1"]


def test_a_gradual_decline_fails_l3():
    assert not sl.evaluate(_obs(gradual=True))["bars"]["L3"]


def test_refraction_that_does_not_help_fails_l4():
    assert not sl.evaluate(_obs(refr=1.2))["bars"]["L4"]


def test_the_ladder_runs_from_16_to_8192():
    assert sl.LADDER[0] == 16 and sl.LADDER[-1] == 8192
