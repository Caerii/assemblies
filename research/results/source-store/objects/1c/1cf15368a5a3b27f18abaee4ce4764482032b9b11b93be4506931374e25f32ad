"""Amendment 48's bars (PREREG_refraction_memory.md), on synthetic records: brains
on the probes' logistic law pass; a law at another position fails M1; no shift
under adaptation fails M2; rescues ordered against slack fail M3; a rescue of
another size fails M4; too few collapsed brains leave M3 and M4 untested
(failed)."""
import pytest

from research.experiments import memory_signal_margin as sm

pytestmark = pytest.mark.requires_torch

REF = 0.35


def _arm(xs, masked_fit=sm.MASKED_FIT, habit_fit=sm.HABIT_FIT, lift=0.0):
    masked = [sm.logistic(x, *masked_fit) for x in xs]
    habit = [min(1.0, sm.logistic(x, *habit_fit) + lift) for x in xs]
    return {"slack": [x * REF for x in xs], "top1": [0.5] * len(xs), "masked": masked, "habit": habit}


def _obs(spread=(0.35, 0.95), **kw):
    lo, hi = spread
    cells = {}
    for n, k, p, tau in sm.CELLS:
        arms = {str(sm.REFERENCE): {"slack": [REF] * 20, "top1": [1.0] * 20, "masked": [1.0] * 20,
                                    "habit": [1.0] * 20}}
        for i, u in enumerate(sm.USES):
            xs = [lo + (hi - lo) * (i * 20 + j) / (20 * len(sm.USES) - 1) for j in range(20)]
            arms[str(u)] = _arm(xs, **kw)
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "arms": arms}
    return {"cells": cells}


def test_brains_on_the_law_pass():
    out = sm.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_a_law_elsewhere_fails_m1():
    assert not sm.evaluate(_obs(masked_fit=(0.9, 0.04), habit_fit=(0.78, 0.09)))["bars"]["M1"]


def test_no_shift_fails_m2():
    assert not sm.evaluate(_obs(habit_fit=sm.MASKED_FIT))["bars"]["M2"]


def test_rescues_ordered_against_slack_fail_m3():
    mid, w = sm.HABIT_FIT
    assert not sm.evaluate(_obs(habit_fit=(mid, -w)))["bars"]["M3"]


def test_a_rescue_of_another_size_fails_m4():
    assert not sm.evaluate(_obs(lift=0.3))["bars"]["M4"]


def test_too_few_collapsed_brains_leave_m3_m4_untested():
    out = sm.evaluate(_obs(spread=(0.80, 0.95)))
    assert not out["bars"]["M3"] and not out["bars"]["M4"]


def test_the_fit_recovers_a_logistic():
    xs = [0.3 + 0.01 * i for i in range(70)]
    mid, w = sm.fit(xs, [sm.logistic(x, 0.7, 0.05) for x in xs])
    assert abs(mid - 0.7) < 0.002 and abs(w - 0.05) < 0.002


def test_the_cells_are_new():
    from research.experiments import memory_read_adaptation as ra
    from research.experiments import memory_read_rescue as rr
    from research.experiments import memory_reuse_budget as rb
    used = ({c[:3] for c in ra.CELLS} | {c[:3] for c in rr.CELLS} | {c[:3] for c in rb.CELLS}
            | {(10000, 75, 0.48), (12000, 80, 0.45), (8000, 60, 0.6)})
    assert not {c[:3] for c in sm.CELLS} & used
