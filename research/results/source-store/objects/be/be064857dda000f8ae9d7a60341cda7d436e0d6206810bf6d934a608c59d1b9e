"""Amendment 40's bars (PREREG_refraction_memory.md), on synthetic records: graded
short sequences that outlast the single cliff pass; an early cliff fails A1;
short sequences failing with the single one fail A2; all-or-none brains fail
A3."""
import math

import pytest

from research.experiments import memory_load_law as ml
from research.experiments import memory_load_many as mm

pytestmark = pytest.mark.requires_torch


def _rows(n, k, p, rho50, width, graded):
    rows = {}
    for r in mm.LADDER:
        lo, hi = rho50 / width, rho50 * width
        full = 1.0 if r <= lo else (0.0 if r >= hi else 1.0 - (r - lo) / (hi - lo))
        if graded:
            whole = [full] * 20
        else:                                   # all or none per brain
            whole = [1.0] * round(20 * full) + [0.0] * (20 - round(20 * full))
        rows[str(int(round(r * ml.unit(n, k, p))))] = {"rho": r, "whole": whole,
                                                       "full": sum(whole) / 20}
    return {"ladder": rows}


def _obs(single=0.115, short=0.145, width=1.25, graded=True):
    cells = {}
    for n, k, p in mm.CELLS:
        arms = {"single": _rows(n, k, p, single, 1.08, False)}
        for a in mm.ARMS[1:]:
            arms[str(a)] = _rows(n, k, p, short, width, graded)
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": 0, "arms": arms}
    return {"cells": cells}


def test_graded_short_sequences_past_the_cliff_pass():
    out = mm.evaluate(_obs())
    assert all(out["bars"].values()), out["bars"]


def test_an_early_cliff_fails_a1():
    assert not mm.evaluate(_obs(single=0.085))["bars"]["A1"]


def test_short_sequences_failing_with_the_single_one_fail_a2():
    assert not mm.evaluate(_obs(short=0.118))["bars"]["A2"]


def test_all_or_none_brains_fail_a3():
    assert not mm.evaluate(_obs(graded=False))["bars"]["A3"]


def test_a_missing_arm_fails_every_bar():
    obs = _obs()
    for cell in obs["cells"].values():
        cell["arms"].pop("16")
    assert not any(mm.evaluate(obs)["bars"].values())


def test_the_cells_are_new_and_in_regime():
    from research.experiments import memory_load_drift as md
    from research.experiments import memory_load_tau as mt
    run = {(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5),
           (8000, 120, 0.5), (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5),
           (15000, 50, 0.7), (20000, 200, 0.3), *ml.CELLS, *ml.REPORTED, *md.CELLS, *mt.CELLS}
    assert not set(mm.CELLS) & run and mm.SMOKE_CELL in run
    for n, k, p in mm.CELLS:
        assert k * p >= 1.15 * 3 * math.log(n)
    for spec in mm.plan(mm.CELLS):
        if spec["arm"] != "single":
            assert all(L % spec["arm"] == 0 for L in spec["ladder"])
