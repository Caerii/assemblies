"""Amendment 55's bars (PREREG_refraction_memory.md), on synthetic records: the probe's pattern
passes; a set point that touches a healthy store fails G1; a short repair fails G2; a short
lifecycle reach fails G3; a set point well below the reference rule fails G4; collapsed brains
left fail G5; a heavy sleep fails G6; a healthy brain dreaming past its own set point fails G7; a
non-equivalent store loop fails everything."""
import pytest

from research.experiments import memory_setpoint_sleep as sp

pytestmark = pytest.mark.requires_torch


def _arm(before, median, setpoint, removed=0.01, low=0, n=20):
    return {"before": [before] * n, "median": [median] * n, "setpoint": [0.05] * low + [setpoint] * (n - low),
            "median_removed": removed, "setpoint_removed": removed}


def _obs(healthy=1.0, healthy_removed=0.0, standard=(0.03, 0.77, 0.78), comparator=(0.25, 0.67, 0.71),
         removed=(0.049, 0.03), low=0, over=0, equivalent=True, n=20):
    cells = {}
    for nn, k, p, tau in sp.CELLS:
        h = _arm(1.0, 1.0, healthy, removed=healthy_removed)
        h["max_contrast"] = [1.37] * (n - over) + [1.40] * over
        cells[f"{nn}/{k}/{p:g}"] = {
            "n": nn, "k": k, "p": p, "tau": tau, "setpoint": [1.38] * n, "median_threshold": 1.397,
            "healthy": h, "standard": _arm(*standard, removed=removed[0], low=low),
            "comparator": _arm(*comparator, removed=removed[1], low=low)}
    return {"cells": cells, "equivalent": equivalent}


def test_the_probe_pattern_passes():
    out = sp.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_touching_a_healthy_store_fails_g1():
    assert not sp.evaluate(_obs(healthy=0.95))["bars"]["G1"]
    assert not sp.evaluate(_obs(healthy_removed=0.01))["bars"]["G1"]


def test_a_short_repair_fails_g2():
    assert not sp.evaluate(_obs(standard=(0.03, 0.6, 0.5)))["bars"]["G2"]


def test_a_short_reach_fails_g3():
    assert not sp.evaluate(_obs(comparator=(0.25, 0.4, 0.4)))["bars"]["G3"]


def test_well_below_the_reference_rule_fails_g4():
    assert not sp.evaluate(_obs(comparator=(0.25, 0.7, 0.6)))["bars"]["G4"]


def test_collapsed_brains_left_fail_g5():
    assert not sp.evaluate(_obs(low=3))["bars"]["G5"]


def test_a_heavy_sleep_fails_g6():
    assert not sp.evaluate(_obs(removed=(0.12, 0.03)))["bars"]["G6"]
    assert not sp.evaluate(_obs(removed=(0.05, 0.06)))["bars"]["G6"]


def test_a_healthy_brain_past_its_set_point_fails_g7():
    assert not sp.evaluate(_obs(over=1))["bars"]["G7"]


def test_a_non_equivalent_loop_fails_everything():
    assert not any(sp.evaluate(_obs(equivalent=False))["bars"].values())


def test_the_cells_and_brains_are_new():
    from research.experiments import memory_comparator as mc
    from research.experiments import memory_lifecycle as lc
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_read_rescue as rre
    from research.experiments import memory_repetition_reach as rr
    from research.experiments import memory_reuse_budget as rb
    from research.experiments import memory_robust_sleep as rs
    from research.experiments import memory_signal_margin as sm
    from research.experiments import memory_sleep as sl
    from research.experiments import memory_write_separation as ws
    used = ({c[:3] for c in ra.CELLS} | {c[:3] for c in rre.CELLS} | {c[:3] for c in rb.CELLS}
            | {c[:3] for c in sm.CELLS} | {c[:3] for c in ws.CELLS} | {c[:3] for c in mc.CELLS}
            | {c[:3] for c in sl.CELLS} | {c[:3] for c in lc.CELLS} | {c[:3] for c in rr.CELLS}
            | {c[:3] for c in rs.CELLS} | {(10000, 75, 0.48), (12000, 80, 0.45), (8000, 60, 0.6)})
    assert not {c[:3] for c in sp.CELLS} & used
    assert min(sp.SEEDS) > max(rs.REFERENCE_SEEDS)
    assert not set(sp.SEEDS) & set(sp.REFERENCE_SEEDS)
