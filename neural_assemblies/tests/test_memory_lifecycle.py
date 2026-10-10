"""Amendment 52's bars (PREREG_refraction_memory.md), on synthetic records: a lifecycle that does
more than either half passes; no synergy fails L1; a short reach fails L2 or L3; collapsed brains
left fail L4; a heavy sleep after the comparator fails L5; a non-equivalent store loop fails
everything."""
import pytest

from research.experiments import memory_lifecycle as lc

pytestmark = pytest.mark.requires_torch


def _arm(standard, sleep, comparator, both, removed=0.005, low=0, n=20):
    vals = [0.05] * low + [both] * (n - low)
    return {"standard": [standard] * n, "sleep": [sleep] * n, "comparator": [comparator] * n,
            "both": vals, "sleep_removed": 0.12, "both_removed": removed, "flags": 0.05}


def _obs(a80=(0.0, 0.17, 0.61, 0.76), a100=(0.0, 0.03, 0.25, 0.55), removed=0.005, low=0, equivalent=True):
    cells = {}
    for n, k, p, tau in lc.CELLS:
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "calibration": {"threshold": 1.4},
                                   "80": _arm(*a80, removed=removed, low=low),
                                   "100": _arm(*a100, removed=removed, low=low)}
    return {"cells": cells, "equivalent": equivalent}


def test_a_synergistic_lifecycle_passes():
    out = lc.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_no_synergy_fails_l1():
    assert not lc.evaluate(_obs(a100=(0.0, 0.03, 0.5, 0.55)))["bars"]["L1"]


def test_a_short_reach_fails_l2_or_l3():
    assert not lc.evaluate(_obs(a80=(0.0, 0.17, 0.4, 0.5)))["bars"]["L2"]
    assert not lc.evaluate(_obs(a100=(0.0, 0.03, 0.1, 0.35)))["bars"]["L3"]


def test_collapsed_brains_left_fail_l4():
    assert not lc.evaluate(_obs(low=3))["bars"]["L4"]


def test_a_heavy_sleep_fails_l5():
    assert not lc.evaluate(_obs(removed=0.05))["bars"]["L5"]


def test_a_non_equivalent_loop_fails_everything():
    assert not any(lc.evaluate(_obs(equivalent=False))["bars"].values())


def test_the_cells_and_brains_are_new():
    """by the ledger (research/experiments/memory_lib/ledger.py): new cells and new brains, outside
    the probe range, against every earlier registration"""
    from research.experiments import memory_lib as lib
    entry = next(e for e in lib.entries() if e.module == "memory_lifecycle")
    assert lib.check_new(entry.cells, entry.seeds, entry.reference_seeds, entry.amendment) == []
