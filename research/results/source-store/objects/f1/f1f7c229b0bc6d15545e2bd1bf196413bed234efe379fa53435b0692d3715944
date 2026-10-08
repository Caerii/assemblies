"""The uniform-noise study's bars (PREREG_refraction_memory.md Amendment 36),
on synthetic records: uniform noise milder than Amendment 34's top-slot noise,
the larger area never derailed by it, the top-slot medians reproducing
Amendment 34, and a random half cue that works must pass; each failure mode
fails its bar."""
import pytest

from research.experiments import memory_noise as mn

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(values):
    return {"keys": SEEDS, "values": values, "mean": sum(values) / len(values)}


def _obs(*, uniform_small=97.0, uniform_big=399.0, top_scale=1.0, random_cue=1.0):
    cells = {}
    for (n, k, p) in mn.CELLS:
        top = mn.A34_MEDIAN[(n, k, p)] * top_scale
        uni = uniform_small if n == 4000 else uniform_big
        noise = {s: {f"{nu:g}": _ens([v] * 20) for nu in mn.NUS}
                 for s, v in (("top", top), ("uniform", uni))}
        cue = {"strong": {"0": _ens([1.0] * 20), "0.25": _ens([1.0] * 20)},
               "random": {"0": _ens([random_cue] * 20), "0.25": _ens([random_cue] * 20)}}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "noise": noise, "cue": cue}
    return {"cells": cells}


def test_the_expected_record_passes():
    bars = mn.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_uniform_noise_as_harsh_as_top_fails_u1():
    assert not mn.evaluate(_obs(uniform_small=45.0))["bars"]["U1"]


def test_a_derailed_large_area_fails_u2():
    assert not mn.evaluate(_obs(uniform_big=300.0))["bars"]["U2"]


def test_top_slot_medians_far_from_amendment_34_fail_u3():
    assert not mn.evaluate(_obs(top_scale=0.5))["bars"]["U3"]


def test_a_random_half_cue_that_fails_fails_u4():
    assert not mn.evaluate(_obs(random_cue=0.5))["bars"]["U4"]


def test_a_missing_cell_fails_every_bar():
    obs = _obs()
    obs["cells"].pop("8000/60/0.5")
    assert not any(mn.evaluate(obs)["bars"].values())
