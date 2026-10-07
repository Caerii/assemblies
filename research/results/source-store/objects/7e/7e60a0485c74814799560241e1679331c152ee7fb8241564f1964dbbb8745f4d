"""The robustness study's bars (PREREG_refraction_memory.md Amendment 34), on
synthetic records: a shared element budget near the single-sequence limit,
harmless small noise, a horizon that doubles with the area, and a tolerant cue
must pass; a budget far below the limit fails N1; a horizon that does not
grow fails N3."""
import pytest

from research.experiments import memory_robustness as mr

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(values):
    return {"keys": SEEDS, "values": values, "mean": sum(values) / len(values)}


def _obs(*, budget_frac=1.0, big_horizon=110.0):
    cells = {}
    for (n, k, p) in mr.CELLS:
        limit = budget_frac * mr.A29[(n, k, p)] / mr.SEQ_LEN
        many = {str(M): _ens([1.0 if M <= limit / 1.2 else (0.0 if M >= limit * 1.2 else 0.5)] * 20)
                for M in mr.M_LADDER if M <= 4 * limit}
        horizon = big_horizon if n == 8000 else 42.0
        noise = {"0": _ens([399.0] * 20), "0.05": _ens([399.0] * 20), "0.07": _ens([399.0] * 20),
                 "0.1": _ens([horizon] * 20), "0.2": _ens([5.0] * 20)}
        cue = {"0.25": _ens([1.0] * 20), "0.5": _ens([0.02] * 20)}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "many": many, "noise": noise, "cue": cue}
    return {"cells": cells}


def test_a_robust_memory_passes():
    bars = mr.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_a_budget_far_below_the_limit_fails_n1():
    assert not mr.evaluate(_obs(budget_frac=0.3))["bars"]["N1"]


def test_a_horizon_that_does_not_grow_fails_n3():
    assert not mr.evaluate(_obs(big_horizon=50.0))["bars"]["N3"]
