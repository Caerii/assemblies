"""The tiling study's bars (PREREG_refraction_memory.md Amendment 28), on
synthetic records: a sequence that tiles the area, breaks at multiples of n/k
and is rescued by a recovering bias must pass; breaks off the deadline fail
D2; a gradual fresh share fails D1; a reset that does not help fails D3."""
import pytest

from research.experiments import memory_tiling as mt

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(values):
    return {"keys": SEEDS, "values": values, "mean": sum(values) / len(values)}


def _obs(*, gradual=False, offset=0.0, reset_ok=True):
    cells = {}
    for spec in mt.plan(mt.CELLS):
        tile, L = spec["tile"], spec["length"]
        if gradual:
            fresh = [max(0.0, 1.0 - e / (2 * tile)) for e in range(L)]
        else:
            fresh = [1.0 if e < tile else 0.0 for e in range(L)]
        breaks = [round((1 + (s % 2)) * tile + offset * tile) for s in SEEDS]
        reset = [L - 1 if reset_ok else breaks[i] for i in range(20)]
        cells[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = dict(
            spec, arms={"refracted": {"reset": None, "fresh_mean": fresh, "break": _ens(breaks)},
                        "reset": {"reset": spec["reset"], "fresh_mean": fresh,
                                  "break": _ens(reset)}})
    return {"cells": cells}


def test_a_tiling_deadline_rescued_by_reset_passes():
    bars = mt.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_breaks_off_the_deadline_fail_d2():
    assert not mt.evaluate(_obs(offset=0.4))["bars"]["D2"]


def test_a_gradual_fresh_share_fails_d1():
    assert not mt.evaluate(_obs(gradual=True))["bars"]["D1"]


def test_a_reset_that_does_not_help_fails_d3():
    assert not mt.evaluate(_obs(reset_ok=False))["bars"]["D3"]


def test_on_deadline_uses_the_larger_of_five_percent_and_two_steps():
    assert mt.on_deadline(35, 33.3) and not mt.on_deadline(38, 33.3)
    assert mt.on_deadline(139, 133.3) and not mt.on_deadline(141, 133.3)
