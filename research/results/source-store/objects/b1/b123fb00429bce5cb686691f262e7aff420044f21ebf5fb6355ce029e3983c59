"""Amendment 49's bars (PREREG_refraction_memory.md), on synthetic records: a separated store that
rescues the collapse passes; harm to a healthy store fails W1; too small a rescue fails W2; a
short reach fails W3; collapsed brains left fail W4; too many interventions fail W5; captures
left fail W6; a run whose store loop was not equivalent fails everything."""
import pytest

from research.experiments import memory_write_separation as ws

pytestmark = pytest.mark.requires_torch


def _arm(std, sep, cost=0.015, captured=0.0, writes=1000):
    return {"writes": writes,
            "standard": {"reliability": list(std), "captured": [0.03] * len(std), "cluster": [20] * len(std),
                         "interventions": 0},
            "separated": {"reliability": list(sep), "captured": [captured] * len(sep), "cluster": [0] * len(sep),
                          "interventions": int(cost * writes)}}


def _obs(healthy=1.0, rescued=0.96, reach=0.89, cost=0.015, captured=0.0, low=0, equivalent=True):
    cells = {}
    for n, k, p, tau in ws.CELLS:
        sep60 = [0.1] * low + [reach] * (20 - low)
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "arms": {
            "10": _arm([1.0] * 20, [healthy] * 20, cost=cost, captured=captured),
            "50": _arm([0.05] * 20, [rescued] * 20, cost=cost, captured=captured),
            "60": _arm([0.01] * 20, sep60, cost=cost, captured=captured)}}
    return {"cells": cells, "equivalent": equivalent}


def test_a_rescuing_separation_passes():
    out = ws.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_harm_to_a_healthy_store_fails_w1():
    assert not ws.evaluate(_obs(healthy=0.9))["bars"]["W1"]


def test_too_small_a_rescue_fails_w2():
    assert not ws.evaluate(_obs(rescued=0.7))["bars"]["W2"]


def test_a_short_reach_fails_w3():
    assert not ws.evaluate(_obs(reach=0.5))["bars"]["W3"]


def test_collapsed_brains_left_fail_w4():
    assert not ws.evaluate(_obs(low=3))["bars"]["W4"]


def test_too_many_interventions_fail_w5():
    assert not ws.evaluate(_obs(cost=0.05))["bars"]["W5"]


def test_captures_left_fail_w6():
    assert not ws.evaluate(_obs(captured=0.01))["bars"]["W6"]


def test_a_non_equivalent_loop_fails_everything():
    assert not any(ws.evaluate(_obs(equivalent=False))["bars"].values())


def test_the_cells_are_new():
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_read_rescue as rr
    from research.experiments import memory_reuse_budget as rb
    from research.experiments import memory_signal_margin as sm
    used = ({c[:3] for c in ra.CELLS} | {c[:3] for c in rr.CELLS} | {c[:3] for c in rb.CELLS}
            | {c[:3] for c in sm.CELLS} | {(10000, 75, 0.48), (12000, 80, 0.45), (8000, 60, 0.6)})
    assert not {c[:3] for c in ws.CELLS} & used
