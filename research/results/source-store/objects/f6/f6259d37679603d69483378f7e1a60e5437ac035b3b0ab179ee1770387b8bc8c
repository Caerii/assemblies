"""The load law's bars (PREREG_refraction_memory.md Amendment 37), on synthetic
records: a sharp cliff at the predicted rho must pass; a cliff far from it fails
R1; an early cliff fails R2; a slow decline fails R3."""
import pytest

from research.experiments import memory_load_law as ml

pytestmark = pytest.mark.requires_torch


def _cell(n, k, p, rho50, width):
    rows = {}
    for r in ml.LADDER:
        lo, hi = rho50 / width, rho50 * width
        full = 1.0 if r <= lo else (0.0 if r >= hi else 1.0 - (r - lo) / (hi - lo))
        L = int(round(r * ml.unit(n, k, p)))
        rows[str(L)] = {"rho": r, "steps": [], "full": full}
    return {"n": n, "k": k, "p": p, "ladder": rows}


def _obs(rho50=0.142, width=1.15):
    return {"cells": {f"{n}/{k}/{p:g}": _cell(n, k, p, rho50, width) for n, k, p in ml.CELLS + ml.REPORTED}}


def test_the_predicted_cliff_passes():
    bars = ml.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_a_cliff_far_from_the_prediction_fails_r1():
    assert not ml.evaluate(_obs(rho50=0.25))["bars"]["R1"]


def test_an_early_cliff_fails_r2():
    assert not ml.evaluate(_obs(rho50=0.095))["bars"]["R2"]


def test_a_slow_decline_fails_r3():
    assert not ml.evaluate(_obs(width=1.6))["bars"]["R3"]


def test_the_held_out_cells_were_never_in_the_survey():
    survey = {(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5),
              (8000, 120, 0.5), (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5)}
    assert not set(ml.CELLS) & survey and not set(ml.REPORTED) & survey
