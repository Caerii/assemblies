"""The memory-study library (research/experiments/memory_lib): cells and their recovery time, the
ledger of registered cells and brains, and bars as data judged by confidence bounds.

The bars are checked the strongest way available: Amendments 53-55's registered bars, re-expressed
as data and evaluated on their RECORDED runs, must reproduce every recorded verdict when judged as
those registrations judged (the bare mean, ``how="mean"``). Judged by the confidence bound they
may differ, and ``BOUND_READINGS`` pins what they give, so the re-reading is on record too."""
from __future__ import annotations

import json
import pathlib

import pytest

from research.experiments import memory_lib as lib
from research.experiments.memory_lib import Bar, Check, below, brains, delta, every, scalar

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "research" / "results" / "runs"


# ----------------------------------------------------------------- cells
def test_tau_follows_amendment_41s_rule_and_refuses_the_gap():
    assert lib.Cell.of(10000, 75, 0.48).tau == 67              # n/k 133: n/k / 2
    assert lib.Cell.of(11600, 80, 0.44).tau == 72              # 72.5, rounded as registered
    assert lib.Cell.of(2000, 60, 0.5).tau == 33                # n/k 33: n/k
    with pytest.raises(ValueError, match="name tau"):
        lib.Cell.of(4000, 60, 0.5)                             # n/k 67: not measured
    assert lib.Cell.of(4000, 60, 0.5, tau=40).tau == 40


def test_registered_cells_match_the_rule_and_their_recorded_betas():
    from research.experiments import memory_setpoint_sleep as sp
    for n, k, p, tau in sp.CELLS:
        c = lib.Cell.of(n, k, p)
        assert c.tau == tau and c.in_regime
    assert lib.Cell.of(10000, 75, 0.48).beta == 0.36474


# ----------------------------------------------------------------- ledger
def test_every_registration_since_37_took_new_cells_and_brains():
    ledger = lib.entries()
    assert {e.amendment for e in ledger} >= set(range(37, 56)) - {47}
    for e in ledger:
        if e.amendment >= lib.NEW_FROM:
            assert lib.check_new(e.cells, e.seeds, e.reference_seeds, e.amendment, ledger) == [], e.module


def test_the_ledger_catches_reuse():
    ledger = lib.entries()
    assert lib.check_new([(10000, 75, 0.48, 67)], range(2000, 2020), amendment=60, ledger=ledger)  # A45's cell
    assert lib.check_new([(9999, 77, 0.5)], range(1000, 1020), amendment=60, ledger=ledger)       # A53's brains
    assert lib.check_new([(9999, 77, 0.5)], range(960, 980), amendment=60, ledger=ledger)         # probe brains
    assert lib.check_new([(9999, 77, 0.5)], range(3000, 3020), range(3010, 3030), amendment=60,
                         ledger=ledger)                                                           # overlap
    assert lib.check_new([(9999, 77, 0.5)], range(3000, 3020), range(3020, 3040), amendment=60,
                         ledger=ledger) == []
    nxt = lib.next_brains(40, ledger)
    assert lib.check_new([(9999, 77, 0.5)], nxt[:20], nxt[20:], ledger=ledger) == []


# ----------------------------------------------------------------- bars on synthetic cells
def _cell(values, other=None):
    return {"x": list(values), "y": list(other or values), "n": 0.01}


def test_a_mean_that_clears_the_bar_inside_its_interval_fails_by_the_bound():
    noisy = [0.2, 0.9, 0.5, 0.8, 0.4, 0.7, 0.3, 0.6, 0.55, 0.65]          # mean 0.56, wide
    bar = Bar("X1", "REACH", "x >= 0.5", (Check(brains("x"), ">=", 0.5),))
    assert bar.judge({"c": _cell(noisy)}, "mean")["pass"]
    assert not bar.judge({"c": _cell(noisy)}, "bound")["pass"]
    tight = [0.6, 0.61, 0.59, 0.6, 0.62, 0.58, 0.6, 0.61, 0.6, 0.59]
    assert bar.judge({"c": _cell(tight)}, "bound")["pass"]


def test_readings_and_their_judges():
    cell = {"a": [0.9, 0.8, 0.85, 0.1], "b": [0.5, 0.5, 0.5, 0.5], "s": 0.03}
    assert Check(below("a", 0.2), "<=", 1).judge(cell)["pass"]
    assert not Check(below("a", 0.2), "<=", 0).judge(cell)["pass"]
    assert Check(scalar("s"), "<=", 0.05).judge(cell)["pass"]
    assert not Check(every("a", "b"), ">", 0).judge(cell)["pass"]        # one brain below
    r = Check(delta("a", "b"), ">=", -1).judge(cell, "mean")
    assert r["pass"] and abs(r["mean"] - (0.65 / 4 - 0.0)) < 0.2
    two = {"c1": cell, "c2": dict(cell, s=0.09)}
    assert not Bar("Z", "BOTH CELLS", "", (Check(scalar("s"), "<=", 0.05),)).judge(two)["pass"]


# ----------------------------------------------------------------- registered bars, re-expressed
def _p4(cell):
    """Amendment 53's P4: the b = 3 flag rate >= max(2%, 10x the random rate), as one margin"""
    from research.experiments.memory_lib.bars import Reading
    f3, fr = cell["b3"]["flags"], cell["random"]["flags"]
    return Reading("scalar", f3 - max(0.02, 10 * fr), "b3 flags - max(0.02, 10 x random flags)")


A53 = (
    Bar("P1", "THE EDGE", "", (Check(brains("b3/standard"), "<=", 0.25),)),
    Bar("P2", "RESCUE", "", (Check(brains("b3/both"), ">=", 0.75),)),
    Bar("P3", "COMPOSITION", "", (Check(delta("b3/both", "b3/comparator"), ">=", 0.0),
                                  Check(delta("b3/both", "b3/sleep"), ">=", 0.15))),
    Bar("P4", "THE COMPARATOR SEES REPETITION", "", (Check(_p4, ">=", 0.0),)),
    Bar("P5", "HARMLESS AND SPARING", "", (Check(brains("random/both"), ">=", 0.98),
                                           Check(brains("b5/both"), ">=", 0.95),
                                           *(Check(scalar(f"{a}/both_removed"), "<=", 0.02)
                                             for a in ("b3", "b5", "random")))),
    Bar("P6", "NO COLLAPSED BRAINS", "", (Check(below("b3/both", 0.2), "<=", 2),
                                          Check(below("b5/both", 0.2), "<=", 2))),
    Bar("P7", "NOT AN IMMEDIATE MERGE", "", (Check(brains("b3/pair_overlap"), "<=", 0.2),)),
)
A54 = (
    Bar("R1", "SAFE", "", (Check(brains("healthy/median"), ">=", 0.99),
                           Check(scalar("healthy/median_removed"), "<=", 0.001))),
    Bar("R2", "REACH", "", (Check(brains("reuse/median"), ">=", 0.45),)),
    Bar("R3", "NEVER WORSE", "", (Check(delta("reuse/median", "reuse/max"), ">=", -0.02),)),
    Bar("R4", "SPARING", "", (Check(scalar("reuse/median_removed"), "<=", 0.02),)),
    Bar("R5", "NO COLLAPSED BRAINS", "", (Check(below("reuse/median", 0.2), "<=", 2),)),
)
A55 = (
    Bar("G1", "SAFE", "", (Check(brains("healthy/setpoint"), ">=", 0.99),
                           Check(scalar("healthy/setpoint_removed"), "<=", 0.001))),
    Bar("G2", "REPAIR", "", (Check(brains("standard/setpoint"), ">=", 0.6),)),
    Bar("G3", "LIFECYCLE REACH", "", (Check(brains("comparator/setpoint"), ">=", 0.45),)),
    Bar("G4", "AS GOOD AS THE REFERENCE", "", (Check(delta("standard/setpoint", "standard/median"), ">=", -0.05),
                                               Check(delta("comparator/setpoint", "comparator/median"), ">=", -0.05))),
    Bar("G5", "NO COLLAPSED BRAINS", "", (Check(below("standard/setpoint", 0.2), "<=", 2),
                                          Check(below("comparator/setpoint", 0.2), "<=", 2))),
    Bar("G6", "FRUGAL", "", (Check(scalar("standard/setpoint_removed"), "<=", 0.10),
                             Check(scalar("comparator/setpoint_removed"), "<=", 0.05))),
    Bar("G7", "THE MECHANISM", "", (Check(every("healthy/max_contrast", "setpoint"), "<", 0.0),)),
)
REGISTERED = {
    "memory.repetition_reach/repetition-reach-20261009": (A53, "memory_repetition_reach"),
    "memory.robust_sleep/robust-sleep-20261009": (A54, "memory_robust_sleep"),
    "memory.setpoint_sleep/setpoint-sleep-20261010": (A55, "memory_setpoint_sleep"),
}
#: judged by the confidence bound instead of the bare mean (a post-hoc re-reading; the recorded
#: verdicts stand). Every verdict survives: each pass clears its bound, and A55's G7 fails either
#: way. The narrowest: A53's P3 at (8500, 64, 0.52), both - comparator +0.031 +/- 0.019, lower
#: bound +0.012 against 0.
BOUND_READINGS = {
    "memory.repetition_reach/repetition-reach-20261009": {"P1": True, "P2": True, "P3": True, "P4": True,
                                                         "P5": True, "P6": True, "P7": True},
    "memory.robust_sleep/robust-sleep-20261009": {"R1": True, "R2": True, "R3": True, "R4": True, "R5": True},
    "memory.setpoint_sleep/setpoint-sleep-20261010": {"G1": True, "G2": True, "G3": True, "G4": True,
                                                     "G5": True, "G6": True, "G7": False},
}


@pytest.mark.parametrize("run", sorted(REGISTERED))
def test_registered_bars_as_data_reproduce_the_recorded_verdicts(run):
    import importlib
    bars, module = REGISTERED[run]
    record = json.loads((RUNS / run / "results.json").read_text(encoding="utf-8"))
    obs = record["observations"]
    recorded = importlib.import_module(f"research.experiments.{module}").evaluate(obs)["bars"]
    cells = obs["cells"]
    as_mean = {k: v["pass"] for k, v in lib.evaluate(bars, cells, "mean").items()}
    assert as_mean == recorded
    as_bound = {k: v["pass"] for k, v in lib.evaluate(bars, cells, "bound").items()}
    assert as_bound == BOUND_READINGS[run]
