"""Amendment 47's bars (PREREG_refraction_memory.md), on synthetic records: a
paired rescue of collapsed brains passes; a brain made worse fails Q1; too few
collapsed brains leave Q2 untested (fails); harm to a healthy memory fails Q3;
an equal gain on the repetition failure fails Q4."""
import pytest

from research.experiments import memory_read_rescue as rr

pytestmark = pytest.mark.requires_torch


def _modes(masked, habit):
    return {"modes": {"masked": list(masked), "habit": list(habit), "strong": list(masked)}}


def _obs(rec_masked=None, rec_habit=None, rep_gain=0.03, healthy=1.0):
    rec_masked = rec_masked or [0.0] * 6 + [0.3] * 4 + [0.95] * 10
    rec_habit = rec_habit or [0.4] * 6 + [0.6] * 4 + [0.98] * 10
    cells = {}
    for n, k, p, tau in rr.CELLS:
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "arms": {
            "recurrence40": _modes(rec_masked, rec_habit), "recurrence50": _modes(rec_masked, rec_habit),
            "repetition": _modes([0.0] * 20, [rep_gain] * 20), "healthy": _modes([1.0] * 20, [healthy] * 20)}}
    return {"cells": cells}


def test_a_paired_rescue_passes():
    out = rr.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_brains_made_worse_fail_q1():
    habit = [0.4] * 4 + [0.0, 0.0] + [0.6] * 4 + [0.98] * 10
    masked = [0.0] * 4 + [0.1, 0.1] + [0.3] * 4 + [0.95] * 10
    assert not rr.evaluate(_obs(rec_masked=masked, rec_habit=habit))["bars"]["Q1"]


def test_too_few_collapsed_brains_leave_q2_untested():
    masked = [0.1] + [0.95] * 19
    habit = [0.6] + [0.98] * 19
    assert not rr.evaluate(_obs(rec_masked=masked, rec_habit=habit))["bars"]["Q2"]


def test_harm_to_a_healthy_memory_fails_q3():
    assert not rr.evaluate(_obs(healthy=0.9))["bars"]["Q3"]


def test_an_equal_gain_on_repetition_fails_q4():
    assert not rr.evaluate(_obs(rep_gain=0.4))["bars"]["Q4"]


def test_the_cells_are_new():
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_reuse_budget as rb
    used = {c[:3] for c in ra.CELLS} | {c[:3] for c in rb.CELLS} | {(10000, 75, 0.48), (12000, 80, 0.45)}
    assert not {c[:3] for c in rr.CELLS} & used
