"""Amendment 38's bars (PREREG_refraction_memory.md), on synthetic records: a cliff
at the constant passes K1 and fails D1; a cliff at the drift passes D1 and fails
K1; an early cliff fails S1; a slow decline fails S2. And on the device: a
brain's record does not depend on the batch it ran in."""
import math

import pytest

from research.experiments import memory_load_drift as md
from research.experiments import memory_load_law as ml

pytestmark = pytest.mark.requires_torch


def _cell(n, k, p, rho50, width):
    rows = {}
    for r in md.LADDER:
        lo, hi = rho50 / width, rho50 * width
        full = 1.0 if r <= lo else (0.0 if r >= hi else 1.0 - (r - lo) / (hi - lo))
        rows[str(int(round(r * ml.unit(n, k, p))))] = {"rho": r, "steps": [], "full": full}
    return {"n": n, "k": k, "p": p, "ladder": rows}


def _obs(at, width=1.1):
    return {"cells": {f"{n}/{k}/{p:g}": _cell(n, k, p, at(n, k, p), width) for n, k, p in md.CELLS}}


def test_a_cliff_at_the_constant_passes_k1_not_d1():
    out = md.evaluate(_obs(lambda n, k, p: 0.142))
    assert out["bars"]["K1"] and not out["bars"]["D1"], out
    assert set(out["nearer"].values()) == {"constant"}


def test_a_cliff_at_the_drift_passes_d1_not_k1():
    out = md.evaluate(_obs(md.drift))
    assert out["bars"]["D1"] and not out["bars"]["K1"], out
    assert set(out["nearer"].values()) == {"drift"}


def test_an_early_cliff_fails_s1():
    assert not md.evaluate(_obs(lambda n, k, p: 0.085))["bars"]["S1"]


def test_a_slow_decline_fails_s2():
    assert not md.evaluate(_obs(lambda n, k, p: 0.142, width=1.6))["bars"]["S2"]


def test_the_judged_cells_were_never_run_and_the_forms_part():
    run = {(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5),
           (8000, 120, 0.5), (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5),
           *ml.CELLS, *ml.REPORTED}
    assert not set(md.CELLS) & run and md.SMOKE_CELL in run
    for n, k, p in md.CELLS:
        assert k * p >= 3 * math.log(n)
        assert md.CONSTANT / md.drift(n, k, p) > 1.3


def test_the_fit_reproduces_the_nine_cells_it_was_fitted_to():
    seen = {(2000, 60, .5): .146, (4000, 60, .5): .142, (8000, 60, .5): .119,
            (4000, 120, .5): .160, (8000, 120, .5): .171, (8000, 120, .25): .131,
            (16000, 120, .5): .123, (6000, 90, .4): .1405, (12000, 80, .5): .1149}
    gaps = [abs(math.log(md.drift(*c) / r)) for c, r in seen.items()]
    assert max(gaps) < 0.15 and sum(g * g for g in gaps) / len(gaps) < 0.065 ** 2


@pytest.mark.requires_cuda
def test_a_brains_record_does_not_depend_on_its_batch():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    n, k, p = 2000, 60, 0.5
    seeds = [900, 901, 902]
    spec = {"n": n, "k": k, "p": p, "beta": round(md.tl.theta(n, k, p), 5)}
    L = int(round(0.15 * ml.unit(n, k, p)))     # at the cliff: replay lengths vary
    one = md.reliability({**spec, "batch": 1}, L, seeds, "cuda")
    three = md.reliability({**spec, "batch": 3}, L, seeds, "cuda")
    assert one == three
