"""Amendment 50's bars (PREREG_refraction_memory.md), on synthetic records: a comparator that
rescues the collapse selectively passes; harm to a healthy store fails C1; too small a rescue
fails C2; collapsed brains left fail C3; too many flags fail C4; detections that miss the
oracle's fail C5; captures left fail C6; a non-equivalent store loop fails everything."""
import pytest

from research.experiments import memory_comparator as mc

pytestmark = pytest.mark.requires_torch


def _arm(std, comp, flag=0.02, hit=0.8, captured=0.0, judged=10000, oracle=300):
    return {"standard": {"reliability": list(std), "captured": [0.03] * len(std),
                         "judged": judged, "flag": int(flag * judged), "oracle": oracle, "hit": 0},
            "comparator": {"reliability": list(comp), "captured": [captured] * len(comp),
                           "judged": judged, "flag": int(flag * judged), "oracle": oracle,
                           "hit": int(hit * oracle)}}


def _obs(healthy=1.0, rescued=0.94, low=0, flag50=0.02, flag10=0.001, hit=0.8, captured=0.0, equivalent=True):
    cells = {}
    for n, k, p, tau in mc.CELLS:
        comp50 = [0.05] * low + [rescued] * (20 - low)
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "arms": {
            "10": _arm([1.0] * 20, [healthy] * 20, flag=flag10, captured=captured, oracle=5),
            "50": _arm([0.03] * 20, comp50, flag=flag50, hit=hit, captured=captured),
            "60": _arm([0.0] * 20, [0.8] * 20, flag=flag50, hit=hit, captured=captured)}}
    return {"cells": cells, "equivalent": equivalent}


def test_a_selective_rescue_passes():
    out = mc.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_harm_to_a_healthy_store_fails_c1():
    assert not mc.evaluate(_obs(healthy=0.9))["bars"]["C1"]


def test_too_small_a_rescue_fails_c2():
    assert not mc.evaluate(_obs(rescued=0.6))["bars"]["C2"]


def test_collapsed_brains_left_fail_c3():
    assert not mc.evaluate(_obs(low=3))["bars"]["C3"]


def test_too_many_flags_fail_c4():
    assert not mc.evaluate(_obs(flag50=0.2))["bars"]["C4"]
    assert not mc.evaluate(_obs(flag10=0.05))["bars"]["C4"]


def test_detections_missing_the_oracle_fail_c5():
    assert not mc.evaluate(_obs(hit=0.3))["bars"]["C5"]


def test_captures_left_fail_c6():
    assert not mc.evaluate(_obs(captured=0.01))["bars"]["C6"]


def test_a_non_equivalent_loop_fails_everything():
    assert not any(mc.evaluate(_obs(equivalent=False))["bars"].values())


def test_the_cells_are_new():
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_read_rescue as rr
    from research.experiments import memory_reuse_budget as rb
    from research.experiments import memory_signal_margin as sm
    from research.experiments import memory_write_separation as ws
    used = ({c[:3] for c in ra.CELLS} | {c[:3] for c in rr.CELLS} | {c[:3] for c in rb.CELLS}
            | {c[:3] for c in sm.CELLS} | {c[:3] for c in ws.CELLS}
            | {(10000, 75, 0.48), (12000, 80, 0.45), (8000, 60, 0.6)})
    assert not {c[:3] for c in mc.CELLS} & used
