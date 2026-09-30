"""Tests for explicit-area slot-WTA winner selection."""

from __future__ import annotations

import numpy as np

from neural_assemblies.compute.winner_selection import select_slot_winners


def test_select_slot_winners_picks_highest_slot():
    inputs = np.zeros(40, dtype=np.float32)
    inputs[25:30] = 5.0
    winners = select_slot_winners(inputs, k=10, slot_count=4)
    assert len(winners) == 10
    assert all(20 <= w < 30 for w in winners)


def test_slot_wta_class_projection():
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=0.1, seed=0, engine="numpy_sparse", w_max=1e9)
    brain.add_area("HIGH", 100, 10, 1.0, explicit=True)
    brain.add_area("CLASS", 40, 10, 1.0, explicit=True, slot_count=4)
    brain.areas["HIGH"].winners = np.arange(10, dtype=np.uint32)
    brain._explicit_engine.set_winners("HIGH", brain.areas["HIGH"].winners)
    conn = brain.connectomes["HIGH"]["CLASS"].weights
    conn[:, 20:30] = 2.0
    if brain._explicit_engine is not None:
        brain._explicit_engine._area_conns["HIGH"]["CLASS"].weights = conn
    brain.project({}, {"HIGH": ["CLASS"]})
    winners = brain.areas["CLASS"].winners
    assert len(winners) == 10
    assert int(np.min(winners)) >= 20
    assert int(np.max(winners)) < 30
