"""Amendment 43's bars (PREREG_refraction_memory.md), on synthetic records drawn
from the hazard model itself: they pass; a long arm that fails earlier than the
model fails K1/K2; a single sequence whose cliff the model misplaces fails K3;
a length that does not cost fails K4. And the fit recovers (c, h)."""
import math

import pytest

from research.experiments import memory_load_hazard as mh
from research.experiments import memory_load_law as ml

pytestmark = pytest.mark.requires_torch


def _model(rho):
    """A cue capture and hazard that worsen with load, as in Amendment 40."""
    c = 1 / (1 + math.exp((rho - 0.15) / 0.006))
    h = 1e-4 * math.exp((rho - 0.11) / 0.006)
    return c, min(h, 0.5)


def _obs(break_arm=None, break_single=False):
    cells = {}
    for n, k, p, tau in mh.CELLS:
        arms = {}
        for spec in mh.plan([(n, k, p, tau)]):
            arm, rows = spec["arm"], {}
            for L in spec["ladder"]:
                rho = L / ml.unit(n, k, p)
                c, h = _model(rho)
                length = L if arm == "single" else arm
                full = c * (1 - h) ** (length - 1)
                if arm == break_arm:
                    full *= 0.5
                if arm == "single" and break_single:
                    c2, h2 = _model(rho * 1.15)
                    full = c2 * (1 - h2) ** (length - 1)
                rows[str(L)] = {"rho": rho, "whole": [full] * 20, "full": full}
            arms[str(arm)] = {"ladder": rows}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "arms": arms}
    return {"cells": cells}


def test_records_drawn_from_the_model_pass():
    out = mh.evaluate(_obs())
    assert all(out["bars"].values()), (out["bars"], out["errors"], out["rho50"])


def test_a_long_arm_off_the_model_fails():
    assert not mh.evaluate(_obs(break_arm=256))["bars"]["K1"]
    assert not mh.evaluate(_obs(break_arm=1024))["bars"]["K2"]


def test_a_misplaced_single_cliff_fails_k3():
    assert not mh.evaluate(_obs(break_single=True))["bars"]["K3"]


def test_the_fit_recovers_capture_and_hazard():
    c, h = 0.9, 0.003
    got = mh.fit(mh.predict(c, h, 16), mh.predict(c, h, 64))
    assert abs(got[0] - c) < 1e-9 and abs(got[1] - h) < 1e-9


def test_a_length_that_costs_nothing_fails_k4():
    obs = _obs()
    for cell in obs["cells"].values():
        cell["arms"]["64"] = cell["arms"]["16"]
    assert not mh.evaluate(obs)["bars"]["K4"]


def test_the_cells_are_new_and_every_arm_shares_the_ladder():
    seen = {(10000, 100, 0.35), (10000, 120, 0.3), (8000, 80, 0.5), (8000, 60, 0.5), (8000, 120, 0.5),
            (8000, 120, 0.25), (8000, 400, 0.1)}
    assert not {c[:3] for c in mh.CELLS} & seen
    for n, k, p, tau in mh.CELLS:
        assert k * p >= 3 * math.log(n) and tau == round(n / k / 2)
        specs = mh.plan([(n, k, p, tau)])
        assert all(s["ladder"] == specs[0]["ladder"] for s in specs)
        assert all(L % mh.QUANTUM == 0 for L in specs[0]["ladder"])
