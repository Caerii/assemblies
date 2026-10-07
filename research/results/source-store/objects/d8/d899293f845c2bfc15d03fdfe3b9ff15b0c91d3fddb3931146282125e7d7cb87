"""The bidirectional study's bars (PREREG_refraction_memory.md Amendment 31),
on synthetic records: forward-only without reverse links, too weak at one
count, both ways from anywhere at two and three counts with LRI, and
directionless without it, must pass; a backward chain at one count fails R2."""
import pytest

from research.experiments import memory_bidirectional as mb

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(v):
    return {"keys": SEEDS, "values": [v] * 20, "mean": v}


def _obs(*, weak_back=0.1, lri_ok=1.0, masked_two=0.05):
    table = {0: dict(forward_masked=1.0, forward_lri=1.0, backward_lri=0.0, middle_backward=0.0, middle_forward=1.0),
             1: dict(forward_masked=0.1, forward_lri=1.0, backward_lri=weak_back, middle_backward=0.1, middle_forward=1.0),
             2: dict(forward_masked=masked_two, forward_lri=lri_ok, backward_lri=lri_ok, middle_backward=lri_ok, middle_forward=lri_ok),
             3: dict(forward_masked=0.02, forward_lri=lri_ok, backward_lri=lri_ok, middle_backward=lri_ok, middle_forward=lri_ok)}
    cells = {}
    for n, k, p in mb.CELLS:
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p,
                                   "reverse": {str(r): {read: _ens(v) for read, v in reads.items()}
                                               for r, reads in table.items()}}
    return {"cells": cells}


def test_lri_steered_bidirectional_recall_passes():
    bars = mb.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_one_count_that_already_works_fails_r2():
    assert not mb.evaluate(_obs(weak_back=0.95))["bars"]["R2"]


def test_lri_that_does_not_steer_fails_r3():
    assert not mb.evaluate(_obs(lri_ok=0.6))["bars"]["R3"]


def test_a_two_way_chain_that_keeps_direction_without_lri_fails_r4():
    assert not mb.evaluate(_obs(masked_two=0.95))["bars"]["R4"]
