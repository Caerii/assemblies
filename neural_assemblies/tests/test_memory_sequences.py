"""The sequence study's bars (PREREG_refraction_memory.md Amendment 26), on
synthetic records: an area whose deferred and strongly refracted writes
replay their trajectories while the weakly refracted write holds still must
pass S1 and S2; capacities that agree across equal n/k pass S3, across equal
in-degree pass S3d; an online write that does not move fails S2."""
import pytest

from research.experiments import memory_sequences as sq

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(v):
    return {"keys": SEEDS, "values": [v] * len(SEEDS), "mean": v}


def _observations(*, caps, online_own=0.02, deferred_read=0.98):
    cells = {}
    for spec in sq.plan(sq.CELLS):
        n, k, p = spec["n"], spec["k"], spec["p"]
        arms = {}
        for arm in spec["arms"]:
            attractor = arm["rule"] == "round" and arm["s"] < 1
            own = 0.9 if attractor else (online_own if arm["rule"] == "round" else 0.01)
            metric = "complete" if attractor else "replay"
            top = caps[(n, k, p)] if arm["rule"] == "deferred" else 0.5 * caps[(n, k, p)]
            rates = {}
            for j, b in enumerate(arm["betas"]):
                cap = top if j == 2 or len(arm["betas"]) == 1 else 0.6 * top
                read = deferred_read if arm["rule"] == "deferred" else 0.8
                curve = {str(M): {metric: _ens(read if M <= cap else 0.1)}
                         for M in spec["checkpoints"] if M <= 4 * cap}
                rates[f"{b:g}"] = {"beta": b, "metric": metric, "capacity": cap,
                                   "own": _ens(own), "ensembles": curve}
            arms[arm["name"]] = {"rule": arm["rule"], "s": arm["s"], "rates": rates}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "theta": spec["theta"],
                                   "arms": arms}
    return {"cells": cells}


BY_NK = {(2000, 60, 0.5): 500.0, (4000, 60, 0.5): 2000.0, (4000, 120, 0.5): 520.0}
BY_D = {(2000, 60, 0.5): 400.0, (4000, 60, 0.5): 1500.0, (4000, 120, 0.5): 1450.0}


def test_sequences_by_n_over_k_pass_s1_s2_s3():
    bars = sq.evaluate(_observations(caps=BY_NK))["bars"]
    assert bars["S1"] and bars["S2"] and bars["S3"] and not bars["S3d"], bars


def test_sequences_by_degree_pass_s3d_not_s3():
    bars = sq.evaluate(_observations(caps=BY_D))["bars"]
    assert bars["S3d"] and not bars["S3"], bars


def test_an_online_write_that_holds_still_fails_s2():
    assert not sq.evaluate(_observations(caps=BY_NK, online_own=0.4))["bars"]["S2"]


def test_a_deferred_write_that_does_not_replay_fails_s1():
    assert not sq.evaluate(_observations(caps=BY_NK, deferred_read=0.6))["bars"]["S1"]


def test_capacity_interpolates_the_crossing_and_floors_at_zero():
    assert sq.capacity({100: 1.0, 200: 0.9, 400: 0.6, 800: 0.1}) == pytest.approx(459.5, abs=0.5)
    assert sq.capacity({100: 0.3, 200: 0.2}) == 0.0
    assert sq.capacity({100: 0.9, 200: 0.8}) == 200.0
