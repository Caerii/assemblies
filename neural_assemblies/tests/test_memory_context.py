"""The context study's bars (PREREG_refraction_memory.md Amendment 30), on
synthetic records: refraction that codes shared elements apart, a Hebbian area
that merges them, and separation that fades after a gap must pass; shared
codes under refraction fail C1."""
import pytest

from research.experiments import memory_context as mc

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(values):
    return {"keys": SEEDS, "values": values, "mean": sum(values) / len(values)}


def _obs(*, refr_sep=0.0, hebb_sep=0.8, gap_sep=0.3):
    cells = {}
    for n, k, p in mc.CELLS:
        cases = {}
        for tau in mc.TAUS:
            for m in mc.SHARED:
                for gap in mc.GAPS:
                    if tau == "0":
                        sep, right = hebb_sep if m == 16 else 0.2, 5
                    else:
                        sep = refr_sep if gap == 0 else (gap_sep if tau == "33" else 0.1)
                        right = 20 if sep < 0.1 else 14
                    cases[f"{tau}/{m}/{gap}"] = {"tau": tau, "m": m, "gap": gap,
                                                 "separation": _ens([sep] * 20),
                                                 "both_correct": _ens([1.0] * right + [0.0] * (20 - right))}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "cases": cases}
    return {"cells": cells}


def test_refraction_separates_hebbian_merges_recency_fades_passes():
    bars = mc.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_shared_codes_under_refraction_fail_c1():
    assert not mc.evaluate(_obs(refr_sep=0.3))["bars"]["C1"]


def test_a_hebbian_area_that_separates_fails_c2():
    assert not mc.evaluate(_obs(hebb_sep=0.1))["bars"]["C2"]


def test_separation_that_survives_the_gap_fails_c3():
    assert not mc.evaluate(_obs(gap_sep=0.02))["bars"]["C3"]
