"""`get_num_ever_fired()` must survive a winners assignment, and old pickles.

Two separate defects live here, and they pull in opposite directions:

  1. `Area.w` means num-ever-fired between projections, but the `winners`
     SETTER overwrites it with `len(winners)`. An accessor that returned `w`
     reported `k` for any area whose winners had been assigned directly --
     probes, `inhibit_areas`, fixed assemblies -- which reads as a sealed
     area. Hence the `_num_ever_fired` shadow.

  2. That shadow did not exist in checkpoints written before it was added,
     and `checkpoint.py` loads such files from disk. A bare attribute access
     raises `AttributeError` on load; the fix must degrade to `w`.

Fixing (1) the obvious way CAUSES (2), so both are pinned here together.
"""

import pickle

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain


@pytest.fixture
def trained():
    """A brain whose area has recruited strictly more than `k` neurons."""
    b = Brain(p=0.05, seed=7, sampled_recurrence_policy="acknowledged")
    b.add_stimulus("S", 20)
    b.add_area("A", 1000, 20, 0.05)
    for _ in range(12):
        b.project({"S": ["A"]}, {"A": ["A"]})
    area = b.areas["A"]
    # The whole point is a gap between recruitment and k. Without it every
    # assertion below passes for the WRONG reason.
    assert area.get_num_ever_fired() > area.k
    return b, area


def test_recruited_count_is_the_public_semantic_name(trained):
    _b, area = trained
    assert area.recruited_count == area.get_num_ever_fired()


def test_engine_and_area_agree_after_training(trained):
    b, area = trained
    assert area.get_num_ever_fired() == b._engine.get_num_ever_fired("A")


def test_recruitment_survives_inhibit(trained):
    """`inhibit_areas` empties the winners; recruitment is not undone by it."""
    b, area = trained
    expected = area.get_num_ever_fired()

    b.inhibit_areas(["A"])

    assert area.active_count == 0
    assert int(area.w) == 0, "precondition: the setter really does clobber w"
    assert area.get_num_ever_fired() == expected


def test_recruitment_survives_direct_winner_assignment(trained):
    """The `_restore_outer_state` pattern: hand a k-sized set straight in."""
    b, area = trained
    expected = area.get_num_ever_fired()

    area.winners = np.arange(area.k, dtype=np.uint32)

    assert int(area.w) == area.k, "precondition: w now reads k, not recruitment"
    assert area.get_num_ever_fired() == expected


def test_survives_pickle_roundtrip(trained):
    _b, area = trained
    restored = pickle.loads(pickle.dumps(area))
    assert restored.get_num_ever_fired() == area.get_num_ever_fired()


def test_old_checkpoint_without_the_shadow_field_still_loads(trained):
    """An Area pickled BEFORE `_num_ever_fired` existed has no such field.

    It must fall back to `w` -- the value such a checkpoint was recorded
    with -- rather than raising.
    """
    _b, area = trained
    restored = pickle.loads(pickle.dumps(area))
    del restored._num_ever_fired          # what an old checkpoint looks like

    assert restored.get_num_ever_fired() == int(restored.w)


def test_active_count_is_the_other_meaning(trained):
    """`active_count` and `get_num_ever_fired()` must not be interchangeable."""
    _b, area = trained
    assert area.active_count == len(area.winners) == area.k
    assert area.active_count != area.get_num_ever_fired()


def test_population_counts_keep_all_three_meanings(trained):
    b, area = trained
    counts = b.population_counts("A")
    assert counts.active == area.k
    assert counts.ever_fired == area.get_num_ever_fired()
    assert counts.materialized == b._engine.materialized_count("A")

    b.inhibit_areas(["A"])
    cleared = b.population_counts("A")
    assert cleared.active == 0
    assert cleared.ever_fired == counts.ever_fired
    assert cleared.materialized == counts.materialized


def test_dense_population_has_no_materialized_extent():
    b = Brain(p=0.1, seed=3, engine="numpy_explicit")
    b.add_stimulus("S", 10)
    b.add_area("A", 100, 10, 0.05)
    b.project({"S": ["A"]}, {})

    counts = b.population_counts("A")
    assert counts.active == 10
    assert counts.ever_fired == 10
    assert counts.materialized is None


def test_population_counts_reject_unknown_area():
    with pytest.raises(KeyError, match="unknown area 'missing'"):
        Brain(engine="numpy_sparse").population_counts("missing")
