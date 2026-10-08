"""Amendment 42's bars (PREREG_refraction_memory.md), on synthetic records: the
probe's picture passes; type-like codes fail W1; costly moderate reuse fails W2;
costly mild noise fails N1; noise that does not compound with load fails N2."""
import math

import pytest

from research.experiments import memory_reuse_noise as rn

pytestmark = pytest.mark.requires_torch


def _obs(same=0.09, reuse_cost=0.0, mild_cost=0.0, heavy=(0.88, 0.75, 0.32)):
    cells = {}
    for n, k, p, tau in rn.CELLS:
        reuse = {str(r): {str(u): {"whole": [1.0 - (reuse_cost if u else 0.0)] * 20,
                                   "same": [same] * 20 if u else [], "different": [0.02] * 20 if u else []}
                          for u in rn.USES} for r in rn.REUSE_RHO}
        noise = {}
        for i, r in enumerate(rn.NOISE_RHO):
            noise[str(r)] = {str(nu): {"whole": [heavy[i] if nu == 0.1 else 1.0 - (mild_cost if nu else 0.0)] * 20}
                             for nu in rn.NUS}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "reuse": reuse, "noise": noise}
    return {"cells": cells}


def test_the_probes_picture_passes():
    out = rn.evaluate(_obs())
    assert all(out["bars"].values()), out["bars"]


def test_type_like_codes_fail_w1():
    assert not rn.evaluate(_obs(same=0.6))["bars"]["W1"]


def test_costly_moderate_reuse_fails_w2():
    assert not rn.evaluate(_obs(reuse_cost=0.05))["bars"]["W2"]


def test_costly_mild_noise_fails_n1():
    assert not rn.evaluate(_obs(mild_cost=0.05))["bars"]["N1"]


def test_noise_that_does_not_compound_fails_n2():
    assert not rn.evaluate(_obs(heavy=(0.8, 0.8, 0.75)))["bars"]["N2"]


def test_a_missing_arm_fails_every_bar():
    obs = _obs()
    for c in obs["cells"].values():
        c["noise"].pop("0.11")
    assert not any(rn.evaluate(obs)["bars"].values())


def test_the_cells_are_new_in_regime_and_follow_the_tau_rule():
    seen = {(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5), (8000, 120, 0.5),
            (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5), (15000, 50, 0.7), (20000, 200, 0.3),
            (4000, 400, 0.5), (4000, 400, 0.1), (8000, 400, 0.1), (4000, 200, 0.15),
            (6000, 90, 0.4), (12000, 80, 0.5), (6000, 40, 0.5), (18000, 60, 0.6), (24000, 80, 0.5),
            (21000, 70, 0.5), (16000, 80, 0.45), (8000, 80, 0.5), (12000, 70, 0.5),
            (6000, 300, 0.12), (5000, 100, 0.35), (10000, 100, 0.35), (14000, 70, 0.5)}
    assert not {c[:3] for c in rn.CELLS} & seen and rn.SMOKE_CELL[:3] in seen
    for n, k, p, tau in rn.CELLS:
        assert k * p >= 3 * math.log(n)
        assert tau == (round(n / k) if n / k <= 50 else round(n / k / 2))
