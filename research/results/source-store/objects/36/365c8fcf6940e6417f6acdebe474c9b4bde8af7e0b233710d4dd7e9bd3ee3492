"""The recognition study's bars (PREREG_refraction_memory.md Amendment 22),
on synthetic sweeps: recognition far above recall at a weaker write, the gap
widening as fan-in falls, must pass; a gap that does not widen must fail Q2,
and recognition peaking at recall's own write must fail Q3."""
import pytest

from research.experiments import memory_recognition as rc

pytestmark = pytest.mark.requires_torch


def _window(cap):
    if cap <= 0:
        return {"never": True, "upper_censored": False, "upper": None, "lower": None}
    return {"never": False, "upper_censored": False, "upper": cap, "lower": None,
            "last_above": cap}


def _observations(ratio_of, recognition_at_recall=False):
    cells = {}
    for spec in rc.plan(rc.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        grid = spec["betas"]
        recall_i = len(grid) - 3
        recog_i = recall_i if recognition_at_recall else 2
        distinct = rc.A19.get((n, k, p), 1400.0)
        recognition = distinct * ratio_of(k * p)
        sweep = {}
        for i, b in enumerate(grid):
            sweep[f"{b:g}"] = {"beta": b, "windows": {
                "rank1": _window(recognition if i == recog_i else recognition * 0.3),
                "complete_distinct": _window(distinct if i == recall_i else distinct * 0.4)}}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "theta": spec["theta"],
                                   "above_floor": spec["above_floor"], "sweep": sweep}
    return {"cells": cells}


def test_a_widening_gap_at_a_weaker_write_passes():
    out = rc.evaluate(_observations(lambda kp: 2.5 + 200.0 / kp))
    assert all(out["bars"].values()), out["bars"]


def test_a_gap_that_does_not_widen_fails_q2():
    assert not rc.evaluate(_observations(lambda kp: 2.0 + kp / 20))["bars"]["Q2"]


def test_recognition_at_recalls_own_write_fails_q3():
    bars = rc.evaluate(_observations(lambda kp: 2.5 + 200.0 / kp,
                                     recognition_at_recall=True))["bars"]
    assert not bars["Q3"]
