"""The write-rule study's bars (PREREG_refraction_memory.md Amendment 25), on
synthetic records: an online round write that stores attractors beside
deferred and burst writes that store none, and a deferred write that stores
the item's trajectory, must pass; a deferred write that completes items must
fail W1, and one whose next-round reading is no better than its same-round
reading must fail W4."""
import pytest

from research.experiments import memory_write_rules as wr

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(v):
    return {"keys": SEEDS, "values": [v] * len(SEEDS), "mean": v}


def _window(cap):
    if cap <= 0:
        return {"never": True, "upper_censored": False, "upper": None, "lower": None}
    return {"never": False, "upper_censored": False, "upper": cap, "lower": None,
            "last_above": cap}


def _observations(*, deferred_cap=0.0, deferred_next=0.6, online_frac=0.2):
    cells = {}
    for spec in wr.plan(wr.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        round_best = wr.A18[(n, k, p)]
        best = {"round": round_best, "online_burst": online_frac * round_best,
                "deferred": deferred_cap, "burst": 0.0}
        traj = {"round": (0.8, 0.7, 0.8), "online_burst": (0.6, 0.5, 0.6),
                "deferred": (deferred_next, 0.04, 0.03), "burst": (0.05, 0.04, 0.03)}
        rules = {}
        for rule in wr.RULES:
            sweep = {}
            for j, b in enumerate(spec["betas"]):
                cap = best[rule] if j == 5 else 0.5 * best[rule]
                sweep[f"{b:g}"] = {"beta": b, "windows": {"complete_distinct": _window(cap)}}
            nxt, same, own = traj[rule]
            rules[rule] = {"sweep": sweep, "trajectory": {
                "beta": spec["theta"], "next": _ens(nxt), "same": _ens(same), "own": _ens(own)}}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "theta": spec["theta"],
                                   "rules": rules}
    return {"cells": cells}


def test_online_attractors_and_deferred_trajectories_pass():
    bars = wr.evaluate(_observations())["bars"]
    assert all(bars.values()), bars


def test_a_deferred_write_that_completes_fails_w1():
    assert not wr.evaluate(_observations(deferred_cap=40.0))["bars"]["W1"]


def test_a_deferred_write_without_a_trajectory_fails_w4():
    assert not wr.evaluate(_observations(deferred_next=0.1))["bars"]["W4"]


def test_gating_that_costs_little_fails_w3():
    assert not wr.evaluate(_observations(online_frac=0.9))["bars"]["W3"]


def test_the_grid_spans_a_tenth_to_three_theta():
    for spec in wr.plan(wr.CELLS):
        assert len(spec["betas"]) == wr.N_RATES
        assert spec["betas"][0] / spec["theta"] == pytest.approx(0.1, rel=1e-3)
        assert spec["betas"][-1] / spec["theta"] == pytest.approx(3.2, rel=1e-3)
