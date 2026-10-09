"""Amendment 45's bars (PREREG_refraction_memory.md), on synthetic records: a
separable budget with two edges passes; a missing edge fails S1; an interaction
fails S2; a budget whose interior sits on the floor and ceiling leaves S2
UNTESTED (fails) rather than passing it vacuously."""
import pytest

from research.experiments import memory_reuse_budget as rb

pytestmark = pytest.mark.requires_torch


def _f(R):            # repetition term
    return max(0.0, min(1.0, 1.0 - (R - 1.8) / 1.5))


def _g(U):            # recurrence term
    return max(0.0, min(1.0, 1.0 - (U - 20) / 30))


def _obs(f=_f, g=_g, interact=0.0):
    cells = {}
    for n, k, p, tau in rb.CELLS:
        arms = {}
        for u, b in rb.ARMS:
            R = 1.0 if b is None else u / b
            w = f(R) * g(u) - (interact if (u, b) in rb.INTERIOR else 0.0)
            arms[rb.name(u, b)] = {"word": [max(0.0, w)] * 20, "repeats": [R] * 20}
        cells[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": tau, "arms": arms}
    return {"cells": cells}


def test_a_separable_budget_passes():
    out = rb.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_no_recurrence_edge_fails_s1():
    assert not rb.evaluate(_obs(g=lambda U: 1.0))["bars"]["S1"]


def test_an_interaction_fails_s2():
    assert not rb.evaluate(_obs(interact=0.3))["bars"]["S2"]


def test_an_uninformative_interior_leaves_s2_untested():
    step_f = lambda R: 1.0 if R < 1.5 else 0.0          # noqa: E731
    step_g = lambda U: 1.0 if U < 15 else 0.0           # noqa: E731
    out = rb.evaluate(_obs(f=step_f, g=step_g))
    assert not out["bars"]["S2"]
    assert all(len(v) < 2 for v in out["informative"].values())


def test_interpolation_is_piecewise_linear_and_clamped():
    pts = [(1.0, 1.0), (3.0, 0.0)]
    assert rb.interpolate(pts, 2.0) == 0.5 and rb.interpolate(pts, 0.5) == 1.0 and rb.interpolate(pts, 9) == 0.0


def test_the_cells_are_new():
    from research.experiments import memory_reuse_grammar as rg
    from research.experiments import memory_reuse_noise as rn
    used = {c[:3] for c in rg.CELLS} | {c[:3] for c in rn.CELLS} | {(12000, 80, 0.45), (4000, 100, 0.35)}
    assert not {c[:3] for c in rb.CELLS} & used
