"""Amendment 54's bars (PREREG_refraction_memory.md), on synthetic records: the probe's pattern
passes; a median gate that touches a healthy store fails R1; a short reach fails R2; a median
gate worse than the maximum's fails R3; a heavy sleep fails R4; collapsed brains left fail R5; a
non-equivalent store loop fails everything; the rules read a tensor of contrasts."""
import pytest

from research.experiments import memory_robust_sleep as rs

pytestmark = pytest.mark.requires_torch


def _cell(healthy_q=1.0, healthy_removed=0.0, reach_max=0.37, reach_q=0.62, removed=0.016, low=0, n=20):
    return {"calibration": {"thresholds": {"max": 1.56, "median": 1.43}},
            "healthy": {"before": [1.0] * n, "max": [1.0] * n, "median": [healthy_q] * n,
                        "max_removed": 0.0, "median_removed": healthy_removed},
            "reuse": {"comparator": [0.18] * n, "flags": 0.09, "max": [reach_max] * n,
                      "median": [0.05] * low + [reach_q] * (n - low), "max_removed": 0.002, "median_removed": removed}}


def _obs(equivalent=True, **kw):
    cells = {f"{n}/{k}/{p:g}": {"n": n, "k": k, "p": p, "tau": tau, **_cell(**kw)} for n, k, p, tau in rs.CELLS}
    return {"cells": cells, "equivalent": equivalent}


def test_the_probe_pattern_passes():
    out = rs.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_touching_a_healthy_store_fails_r1():
    assert not rs.evaluate(_obs(healthy_q=0.95))["bars"]["R1"]
    assert not rs.evaluate(_obs(healthy_removed=0.01))["bars"]["R1"]


def test_a_short_reach_fails_r2():
    assert not rs.evaluate(_obs(reach_max=0.3, reach_q=0.4))["bars"]["R2"]


def test_worse_than_the_maximum_fails_r3():
    assert not rs.evaluate(_obs(reach_max=0.66, reach_q=0.62))["bars"]["R3"]


def test_a_heavy_sleep_fails_r4():
    assert not rs.evaluate(_obs(removed=0.03))["bars"]["R4"]


def test_collapsed_brains_left_fail_r5():
    assert not rs.evaluate(_obs(low=3))["bars"]["R5"]


def test_a_non_equivalent_loop_fails_everything():
    assert not any(rs.evaluate(_obs(equivalent=False))["bars"].values())


def test_outlier_brains_move_the_maximum_not_the_median():
    import torch
    B = 20
    base = torch.linspace(1.35, 1.40, 300 * B)            # brain-minor, as dream() records
    rare = base.clone()
    rare[7] = 1.6                                           # one brain, one dream
    often = base.clone().view(-1, B)
    often[::2, 3] = 1.6                                     # one brain that settles half the time
    often[:, 11] += 0.3                                     # and a second outlying brain
    a, b, c = rs.thresholds(base, B), rs.thresholds(rare, B), rs.thresholds(often.reshape(-1), B)
    assert b["max"] - a["max"] > 0.2 and c["max"] - a["max"] > 0.2
    # an outlier can shift the median by one rank among near-equal maxima, never by its own size
    assert abs(b["median"] - a["median"]) < 1e-4 and abs(c["median"] - a["median"]) < 1e-4


def test_the_cells_and_brains_are_new():
    from research.experiments import memory_comparator as mc
    from research.experiments import memory_lifecycle as lc
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_read_rescue as rre
    from research.experiments import memory_repetition_reach as rr
    from research.experiments import memory_reuse_budget as rb
    from research.experiments import memory_signal_margin as sm
    from research.experiments import memory_sleep as sl
    from research.experiments import memory_write_separation as ws
    used = ({c[:3] for c in ra.CELLS} | {c[:3] for c in rre.CELLS} | {c[:3] for c in rb.CELLS}
            | {c[:3] for c in sm.CELLS} | {c[:3] for c in ws.CELLS} | {c[:3] for c in mc.CELLS}
            | {c[:3] for c in sl.CELLS} | {c[:3] for c in lc.CELLS} | {c[:3] for c in rr.CELLS}
            | {(10000, 75, 0.48), (12000, 80, 0.45), (8000, 60, 0.6)})
    assert not {c[:3] for c in rs.CELLS} & used
    assert min(rs.SEEDS) > max(rr.REFERENCE_SEEDS)
    assert not set(rs.SEEDS) & set(rs.REFERENCE_SEEDS)
