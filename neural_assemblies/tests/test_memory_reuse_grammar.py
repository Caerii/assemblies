"""Amendment 44's bars (PREREG_refraction_memory.md), on synthetic records: the
transition-repetition picture passes; recurrence that costs by itself fails G2
and G3; repetition that does not cost fails G1; tokens that do not merge fail G4.
And the grammar's walks have the registered branching."""
import pytest

from research.experiments import memory_reuse_grammar as rg

pytestmark = pytest.mark.requires_torch


def _obs(word=None, same=None):
    word = word or {"U20/bV": 0.98, "U20/b4": 0.35, "U20/b2": 0.02, "U10/b2": 0.40, "U40/b8": 0.30}
    same = same or {"U20/bV": 0.09, "U20/b4": 0.12, "U20/b2": 0.14, "U10/b2": 0.12, "U40/b8": 0.12}
    return {"cells": {f"{n}/{k}/{p:g}": {"n": n, "k": k, "p": p, "tau": tau, "arms": {
        a: {"word": [word[a]] * 20, "same": [same[a]] * 20} for a in word}} for n, k, p, tau in rg.CELLS}}


def test_the_repetition_picture_passes():
    out = rg.evaluate(_obs())
    assert all(out["bars"].values()), out


def test_recurrence_that_costs_by_itself_fails_g2_and_g3():
    bars = rg.evaluate(_obs(word={"U20/bV": 0.70, "U20/b4": 0.35, "U20/b2": 0.02,
                                  "U10/b2": 0.90, "U40/b8": 0.10}))["bars"]
    assert not bars["G2"] and not bars["G3"]


def test_repetition_that_costs_nothing_fails_g1():
    assert not rg.evaluate(_obs(word={a: 0.97 for a in ("U20/bV", "U20/b4", "U20/b2", "U10/b2", "U40/b8")}))["bars"]["G1"]


def test_tokens_that_do_not_merge_fail_g4():
    assert not rg.evaluate(_obs(same={a: 0.09 for a in ("U20/bV", "U20/b4", "U20/b2", "U10/b2", "U40/b8")}))["bars"]["G4"]


def test_walks_follow_the_grammar():
    w = rg.walks(7, 50, 30, 2, 1)
    succ = {}
    for row in w:
        for a, b in zip(row, row[1:]):
            succ.setdefault(int(a), set()).add(int(b))
    assert max(len(v) for v in succ.values()) <= 2
    iid = rg.walks(7, 50, 30, None, 1)
    assert iid.shape == (50, rg.LENGTH) and len({(int(a), int(b)) for r in iid for a, b in zip(r, r[1:])}) > 300


def test_the_arms_cross_uses_and_repeats():
    R = {a: u / (b or u) for u, b in rg.ARMS for a in [f"U{u}/b{b or 'V'}"]}
    assert R["U10/b2"] == R["U20/b4"] == R["U40/b8"] == 5
    assert {u for u, b in rg.ARMS if b and u / b == 5} == {10, 20, 40}
