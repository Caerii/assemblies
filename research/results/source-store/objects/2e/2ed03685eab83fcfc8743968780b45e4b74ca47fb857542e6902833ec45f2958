"""Amendment 46's bars (PREREG_refraction_memory.md), on synthetic records: the
probe's picture passes; no rescue fails H1; a rescue from the ceiling is not a
rescue (H1); harm to a healthy memory fails H2; a rescue of repetition fails H3;
strong adaptation that does not harm fails H4."""
import pytest

from research.experiments import memory_read_adaptation as ra

pytestmark = pytest.mark.requires_torch


def _obs(rec=(0.43, 0.70, 0.10), rep=(0.0, 0.03, 0.0), hea=(1.0, 0.99, 0.5)):
    def arm(v):
        return {"modes": {m: [x] * 20 for (m, _, _), x in zip(ra.MODES, v)}}
    return {"cells": {f"{n}/{k}/{p:g}": {"n": n, "k": k, "p": p, "tau": tau,
                                          "arms": {"recurrence": arm(rec), "repetition": arm(rep), "healthy": arm(hea)}}
                      for n, k, p, tau in ra.CELLS}}


def test_the_probe_picture_passes():
    out = ra.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_no_rescue_fails_h1():
    assert not ra.evaluate(_obs(rec=(0.43, 0.48, 0.1)))["bars"]["H1"]


def test_a_rescue_from_the_ceiling_is_not_counted():
    assert not ra.evaluate(_obs(rec=(0.90, 1.0, 0.1)))["bars"]["H1"]


def test_harm_to_a_healthy_memory_fails_h2():
    assert not ra.evaluate(_obs(hea=(1.0, 0.9, 0.5)))["bars"]["H2"]


def test_a_rescue_of_repetition_fails_h3():
    assert not ra.evaluate(_obs(rep=(0.0, 0.3, 0.0)))["bars"]["H3"]


def test_strong_adaptation_that_does_not_harm_fails_h4():
    assert not ra.evaluate(_obs(rec=(0.43, 0.70, 0.40)))["bars"]["H4"]


def test_the_cells_are_new():
    from research.experiments import memory_reuse_budget as rb
    from research.experiments import memory_reuse_grammar as rg
    used = {c[:3] for c in rb.CELLS} | {c[:3] for c in rg.CELLS} | {(12000, 80, 0.45), (8000, 60, 0.5)}
    assert not {c[:3] for c in ra.CELLS} & used
