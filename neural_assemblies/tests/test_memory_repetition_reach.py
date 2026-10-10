"""Amendment 53's bars (PREREG_refraction_memory.md), on synthetic records: the probe's pattern
passes; a standard store already healthy at b = 3 fails P1; a short rescue fails P2; both no
better than sleep fails P3; a comparator blind to repetition fails P4; a lifecycle that harms the
healthy control fails P5; collapsed brains left fail P6; tokens merged at once fail P7; a
non-equivalent store loop fails everything."""
import pytest

from research.experiments import memory_repetition_reach as rr

pytestmark = pytest.mark.requires_torch


def _arm(standard, sleep, comparator, both, flags=0.01, removed=0.002, low=0, overlap=0.11, n=20):
    vals = [0.05] * low + [both] * (n - low)
    return {"standard": [standard] * n, "sleep": [sleep] * n, "comparator": [comparator] * n,
            "both": vals, "sleep_removed": 0.1, "both_removed": removed, "flags": flags,
            "pair_overlap": [overlap] * (n - 1) + [None]}


def _obs(b3=(0.0, 0.59, 0.85, 0.92), b5=(0.91, 0.99, 1.0, 1.0), random=(1.0, 1.0, 1.0, 1.0),
         flags3=0.04, flags_random=0.0008, removed=0.002, low=0, overlap=0.11, equivalent=True):
    cells = {}
    for n, k, p, tau in rr.CELLS:
        cells[f"{n}/{k}/{p:g}"] = {
            "n": n, "k": k, "p": p, "tau": tau, "calibration": {"threshold": 1.4},
            "b3": _arm(*b3, flags=flags3, removed=removed, low=low, overlap=overlap),
            "b5": _arm(*b5, removed=removed),
            "random": _arm(*random, flags=flags_random, removed=0.0)}
    return {"cells": cells, "equivalent": equivalent}


def test_the_probe_pattern_passes():
    out = rr.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_no_edge_fails_p1():
    assert not rr.evaluate(_obs(b3=(0.6, 0.8, 0.9, 0.95)))["bars"]["P1"]


def test_a_short_rescue_fails_p2():
    assert not rr.evaluate(_obs(b3=(0.0, 0.3, 0.6, 0.7)))["bars"]["P2"]


def test_no_composition_fails_p3():
    assert not rr.evaluate(_obs(b3=(0.0, 0.85, 0.8, 0.9)))["bars"]["P3"]
    assert not rr.evaluate(_obs(b3=(0.0, 0.5, 0.95, 0.9)))["bars"]["P3"]


def test_a_blind_comparator_fails_p4():
    assert not rr.evaluate(_obs(flags3=0.01))["bars"]["P4"]
    assert not rr.evaluate(_obs(flags3=0.03, flags_random=0.005))["bars"]["P4"]


def test_harm_or_a_heavy_sleep_fails_p5():
    assert not rr.evaluate(_obs(random=(1.0, 1.0, 1.0, 0.9)))["bars"]["P5"]
    assert not rr.evaluate(_obs(b5=(0.91, 0.99, 1.0, 0.9)))["bars"]["P5"]
    assert not rr.evaluate(_obs(removed=0.05))["bars"]["P5"]


def test_collapsed_brains_left_fail_p6():
    assert not rr.evaluate(_obs(low=3))["bars"]["P6"]


def test_an_immediate_merge_fails_p7():
    assert not rr.evaluate(_obs(overlap=0.4))["bars"]["P7"]


def test_a_non_equivalent_loop_fails_everything():
    assert not any(rr.evaluate(_obs(equivalent=False))["bars"].values())


def test_the_cells_and_brains_are_new():
    """by the ledger (research/experiments/memory_lib/ledger.py): new cells and new brains, outside
    the probe range, against every earlier registration"""
    from research.experiments import memory_lib as lib
    entry = next(e for e in lib.entries() if e.module == "memory_repetition_reach")
    assert lib.check_new(entry.cells, entry.seeds, entry.reference_seeds, entry.amendment) == []
