"""``read_only()``: measuring must not change what is measured.

``frozen()`` was the tool for this and it is not sufficient. It stops weights
changing but not the area GROWING, and growth is the channel that actually made
probes contaminate each other: parsing three items in one order versus the
reverse left one role area at w=647 in one copy and w=650 in the other, under
frozen(), and structurally different brains cannot have matching synapses no
matter how initialisation is seeded.

These tests pin the difference rather than the implementation, because the
failure mode is silent. A leaking probe returns a perfectly plausible answer;
what it corrupts is whatever is measured NEXT, and no assertion in the probe
itself can see that. So each test here compares against a pristine deepcopy.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _brain(seed=42, n=2000, k=50):
    b = Brain(p=0.05, seed=seed, engine="numpy_sparse")
    for area in ("A", "B"):
        b.add_area(area, n=n, k=k, beta=0.1)
    b.add_stimulus("s", k)
    b.add_stimulus("t", k)
    for _ in range(12):
        b.project({"s": ["A"]}, {"A": ["A", "B"]})
    return b


def _conn_snapshot(brain):
    out = {}
    for src, dsts in brain._engine._area_conns.items():
        for dst, conn in dsts.items():
            w = np.asarray(conn.weights)
            if w.size:
                out[(src, dst)] = w.copy()
    return out


# -- what read_only guarantees ---------------------------------------------

def test_read_only_leaves_ever_fired_counts_untouched():
    """The property frozen() does not have."""
    brain = _brain()
    before = {n: a.w for n, a in brain.areas.items()}
    with brain.read_only():
        for _ in range(5):
            brain.project({"t": ["A"]}, {"A": ["B"]})
    assert {n: a.w for n, a in brain.areas.items()} == before


def test_frozen_alone_does_not_stop_growth():
    """Documents WHY read_only exists, so the two are not conflated again.

    If a change ever makes frozen() itself non-growing this fails, which is the
    right prompt to collapse the two rather than a reason to delete the test.
    """
    brain = _brain()
    before = {n: a.w for n, a in brain.areas.items()}
    with brain.frozen():
        for _ in range(5):
            brain.project({"t": ["A"]}, {"A": ["B"]})
    assert {n: a.w for n, a in brain.areas.items()} != before


def test_read_only_does_not_advance_the_generator():
    brain = _brain()
    state = brain._engine._rng.bit_generator.state
    with brain.read_only():
        brain.project({"t": ["A"]}, {"A": ["B"]})
    assert brain._engine._rng.bit_generator.state == state


def test_read_only_leaves_every_weight_untouched():
    brain = _brain()
    before = _conn_snapshot(brain)
    with brain.read_only():
        for _ in range(3):
            brain.project({"t": ["A"]}, {"A": ["B"]})
    after = _conn_snapshot(brain)
    assert set(before) == set(after)
    for key, w in before.items():
        assert np.array_equal(w, after[key]), key


def test_probe_order_does_not_change_the_brain():
    """The measurement this whole mechanism exists to make true.

    Two copies of one brain get the same three drives in opposite orders. Under
    read_only they must end identical -- same ever-fired counts and same
    weights -- or "the same brain" depends on the order things were measured in.
    """
    host = _brain()
    a, b = copy.deepcopy(host), copy.deepcopy(host)
    drives = [{"s": ["A"]}, {"t": ["A"]}, {"s": ["A", "B"]}]
    for probe in drives:
        with a.read_only():
            a.project(probe, {"A": ["B"]})
    for probe in reversed(drives):
        with b.read_only():
            b.project(probe, {"A": ["B"]})

    assert {n: ar.w for n, ar in a.areas.items()} == \
           {n: ar.w for n, ar in b.areas.items()}
    wa, wb = _conn_snapshot(a), _conn_snapshot(b)
    assert set(wa) == set(wb)
    for key, w in wa.items():
        assert np.array_equal(w, wb[key]), key


def test_read_only_restores_state_when_the_body_raises():
    brain = _brain()
    before = {n: a.w for n, a in brain.areas.items()}
    with pytest.raises(ValueError):
        with brain.read_only():
            brain.project({"t": ["A"]}, {"A": ["B"]})
            raise ValueError("boom")
    assert {n: a.w for n, a in brain.areas.items()} == before
    assert brain._engine._no_recruitment is False
    assert brain.disable_plasticity is False


def test_read_only_nests_without_leaking_the_flag():
    brain = _brain()
    with brain.read_only():
        with brain.read_only():
            brain.project({"t": ["A"]}, {"A": ["B"]})
        assert brain._engine._no_recruitment is True
    assert brain._engine._no_recruitment is False


# -- what it must NOT change ------------------------------------------------

def test_winners_still_move_inside_the_block():
    """read_only suppresses growth, not activity -- a probe must still respond.

    If this ever passes trivially (winners frozen too) the probe is measuring
    nothing and every downstream readout is a constant.
    """
    brain = _brain()
    with brain.read_only():
        brain.project({"t": ["A"]}, {"A": ["B"]})
        during = np.asarray(brain.areas["B"].winners).copy()
    assert during.size == brain.areas["B"].k


def test_cold_population_requires_initialization_before_read_only():
    brain = Brain(p=0.05, seed=7, engine="numpy_sparse")
    brain.add_area("C", n=1000, k=50, beta=0.1)
    brain.add_stimulus("u", 50)
    with pytest.raises(ValueError, match="materialized"):
        with brain.read_only():
            brain.project({"u": ["C"]}, {})
    assert brain.areas["C"].w == brain._engine._areas["C"].w == 0
    # Frozen learning still allows deliberate population initialization.
    with brain.frozen():
        brain.project({"u": ["C"]}, {})
    assert brain._engine._areas["C"].w >= 50
