"""Amendment 51's bars (PREREG_refraction_memory.md), on synthetic records: a self-limiting sleep
that repairs a collapsed store and spares a healthy one passes; too small a repair fails S1;
erasure at high dose fails S2; harm to a healthy store fails S3; a gate that stays open fails
S4; too many counts removed fails S5; collapsed brains left fail S6; a missing dose fails all."""
import pytest

from research.experiments import memory_sleep as sl

pytestmark = pytest.mark.requires_torch


def _doses(rels, removed=(0.0, 0.03, 0.04, 0.045, 0.045), gated=(0.0, 0.4, 0.1, 0.02, 0.001), n=20, low=0):
    out = []
    for i, (e, r) in enumerate(zip(sl.DOSES, rels)):
        vals = [r] * n
        if e == 300 and low:
            vals = [0.05] * low + [r] * (n - low)
        out.append({"episodes": e, "reliability": vals, "removed": removed[i], "gated_interval": gated[i]})
    return out


def _obs(sick=(0.03, 0.67, 0.76, 0.79, 0.80), well=(1.0, 1.0, 1.0, 1.0, 1.0), removed=None,
         well_removed=0.0, gated=None, low=0, drop=None):
    cells = {}
    for n, k, p, tau in sl.CELLS:
        s = _doses(sick, removed=removed or (0.0, 0.03, 0.04, 0.045, 0.045),
                   gated=gated or (0.0, 0.4, 0.1, 0.02, 0.001), low=low)
        w = _doses(well, removed=(0.0,) * 4 + (well_removed,), gated=(0.0,) * 5)
        if drop is not None:
            s = [d for d in s if d["episodes"] != drop]
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "calibration": {"threshold": 1.4},
                                   "50": {"doses": s}, "10": {"doses": w}}
    return {"cells": cells}


def test_a_self_limiting_repair_passes():
    out = sl.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_too_small_a_repair_fails_s1():
    assert not sl.evaluate(_obs(sick=(0.03, 0.2, 0.3, 0.4, 0.45)))["bars"]["S1"]


def test_erasure_at_high_dose_fails_s2():
    assert not sl.evaluate(_obs(sick=(0.03, 0.67, 0.76, 0.3, 0.0)))["bars"]["S2"]


def test_harm_to_a_healthy_store_fails_s3():
    assert not sl.evaluate(_obs(well=(1.0, 1.0, 1.0, 0.9, 0.8)))["bars"]["S3"]
    assert not sl.evaluate(_obs(well_removed=0.01))["bars"]["S3"]


def test_a_gate_that_stays_open_fails_s4():
    assert not sl.evaluate(_obs(gated=(0.0, 0.4, 0.3, 0.2, 0.1)))["bars"]["S4"]


def test_too_many_counts_removed_fails_s5():
    assert not sl.evaluate(_obs(removed=(0.0, 0.1, 0.2, 0.3, 0.4)))["bars"]["S5"]


def test_collapsed_brains_left_fail_s6():
    assert not sl.evaluate(_obs(low=3))["bars"]["S6"]


def test_a_missing_dose_fails_everything():
    assert not any(sl.evaluate(_obs(drop=1000))["bars"].values())


def test_the_cells_and_brains_are_new():
    from research.experiments import memory_comparator as mc
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_read_rescue as rr
    from research.experiments import memory_reuse_budget as rb
    from research.experiments import memory_signal_margin as sm
    from research.experiments import memory_write_separation as ws
    used = ({c[:3] for c in ra.CELLS} | {c[:3] for c in rr.CELLS} | {c[:3] for c in rb.CELLS}
            | {c[:3] for c in sm.CELLS} | {c[:3] for c in ws.CELLS} | {c[:3] for c in mc.CELLS}
            | {(10000, 75, 0.48), (12000, 80, 0.45), (8000, 60, 0.6), (11000, 80, 0.45)})
    assert not {c[:3] for c in sl.CELLS} & used
    assert not set(sl.SEEDS) & set(sl.REFERENCE_SEEDS)
    assert min(sl.SEEDS) > max(mc.SEEDS)
