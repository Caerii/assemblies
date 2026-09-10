"""Behavioral obligations for the named-target multi-round schedule."""
import copy

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain


def make_brain(engine="numpy_exact", explicit=False):
    b = Brain(engine=engine, seed=13, p=.2, norm_init=False,
              save_winners=True, save_size=True)
    for name in ("S", "T"):
        b.add_area(name, 100, 10, explicit=explicit)
    for name in ("s", "t"):
        b.add_stimulus(name, 10)
    b.project({"s": ["S"], "t": ["T"]}, {})
    return b


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_sparse", "numpy_explicit"])
def test_rounds_preserve_every_history_entry_and_activation(engine):
    b = make_brain(engine)
    reference = copy.deepcopy(b)
    b.record_activation = reference.record_activation = True
    b.project_rounds("T", {"s": ["T"]}, {"S": ["T"]}, 3)
    for _ in range(3):
        reference.project({"s": ["T"]}, {"S": ["T"]})
    for name in ("S", "T"):
        np.testing.assert_array_equal(b.areas[name].winners, reference.areas[name].winners)
        assert b.areas[name].saved_w == reference.areas[name].saved_w
        np.testing.assert_array_equal(b.areas[name].saved_winners,
                                      reference.areas[name].saved_winners)
    assert b.last_activation_scores == reference.last_activation_scores
    assert b.last_pre_kwta_totals == reference.last_pre_kwta_totals
    assert b.last_pre_kwta_counts == reference.last_pre_kwta_counts


@pytest.mark.parametrize("closed", ["target", "source", "fiber"])
def test_closed_schedule_does_not_execute_or_learn(closed):
    b = make_brain()
    if closed == "target":
        b.inhibit_area("T")
    elif closed == "source":
        b.inhibit_area("S")
    else:
        b.inhibit_fiber("S", "T")
    before = b.areas["T"].winners.copy()
    count = b._engine._projections
    b.project_rounds("T", {}, {"S": ["T"]}, 3)
    assert b._engine._projections == count
    assert not b._engine._area_pot.get(("S", "T"))
    np.testing.assert_array_equal(b.areas["T"].winners, before)


def test_facade_clamp_is_respected_before_the_first_round():
    b = make_brain()
    b.areas["T"].fixed_assembly = True
    before = b.areas["T"].winners.copy()
    b.project_rounds("T", {"s": ["T"]}, {}, 3)
    assert b._engine.is_fixed("T")
    np.testing.assert_array_equal(b.areas["T"].winners, before)


@pytest.mark.parametrize("explicit", [False, True])
def test_named_target_excludes_other_destinations(explicit):
    b = make_brain(explicit=explicit)
    before = b.areas["S"].winners.copy()
    history_length = len(b.areas["S"].saved_w)
    b.project_rounds("T", {"s": ["T", "S"]}, {}, 2)
    np.testing.assert_array_equal(b.areas["S"].winners, before)
    assert len(b.areas["S"].saved_w) == history_length


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
@pytest.mark.parametrize("explicit", [False, True])
def test_invalid_round_count_is_rejected_before_dispatch(rounds, explicit, monkeypatch):
    b = make_brain(explicit=explicit)
    calls = []
    monkeypatch.setattr(b, "project", lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(b._engine, "project_rounds", lambda **kw: calls.append(1))
    with pytest.raises(ValueError, match="positive integer"):
        b.project_rounds("T", {"s": ["T"]}, {}, rounds)
    assert calls == []


@pytest.mark.parametrize("stims,areas", [({}, {}), ({}, {"T": ["T"]})])
def test_empty_resolved_schedule_is_rejected(stims, areas):
    b = make_brain()
    before = b._engine._projections
    with pytest.raises(ValueError, match="no inputs"):
        b.project_rounds("T", stims, areas, 2)
    assert b._engine._projections == before


def test_unknown_excluded_destination_is_not_silently_accepted():
    b = make_brain()
    before = b._engine._projections
    with pytest.raises(IndexError, match="missing"):
        b.project_rounds("T", {"s": ["T", "missing"]}, {}, 2)
    assert b._engine._projections == before


@pytest.mark.parametrize("multiple", [False, True])
def test_empty_source_activity_cannot_reuse_backend_winners(multiple):
    b = make_brain()
    assert len(b._engine.get_winners("S")) == 10
    b.areas["S"].winners = np.array([], dtype=np.uint32)
    if multiple:
        b.project_rounds("T", {}, {"S": ["T"]}, 2)
    else:
        b.project({}, {"S": ["T"]})
    assert len(b._engine.get_winners("S")) == 0
    assert not b._engine._area_pot.get(("S", "T"))


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_sparse", "numpy_explicit"])
def test_rounds_inside_read_only_restore_activity_and_history(engine):
    b = make_brain(engine)
    before = copy.deepcopy(b)
    with b.read_only():
        b.project_rounds("T", {"s": ["T"]}, {"S": ["T"]}, 3)
    for name in ("S", "T"):
        np.testing.assert_array_equal(b.areas[name].winners, before.areas[name].winners)
        np.testing.assert_array_equal(b.areas[name].saved_winners,
                                      before.areas[name].saved_winners)
        assert b.areas[name].saved_w == before.areas[name].saved_w
        assert b._engine.get_num_ever_fired(name) == before._engine.get_num_ever_fired(name)
    # A later learning step must still follow the unobserved trajectory.
    b.project({"s": ["T"]}, {"S": ["T"]})
    before.project({"s": ["T"]}, {"S": ["T"]})
    np.testing.assert_array_equal(b.areas["T"].winners, before.areas["T"].winners)


def test_cold_read_only_rounds_fail_before_source_synchronization():
    b = Brain(engine="numpy_sparse", seed=13)
    b.add_area("S", 100, 10)
    b.add_area("T", 100, 10)
    b.add_stimulus("s", 10)
    b.project({"s": ["S"]}, {})
    backend_before = b._engine.get_winners("S").copy()
    b.areas["S"].winners = np.array([], dtype=np.uint32)
    with b.read_only():
        with pytest.raises(ValueError, match="materializ"):
            b.project_rounds("T", {}, {"S": ["T"]}, 2)
        np.testing.assert_array_equal(b._engine.get_winners("S"), backend_before)
    assert b._engine.materialized_count("T") == 0


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_sparse", "numpy_explicit"])
@pytest.mark.parametrize("learning", [False, True])
def test_engine_repetition_matches_individual_steps(engine, learning):
    b = make_brain(engine)
    reference = copy.deepcopy(b._engine)
    result = b._engine.project_rounds("T", ["s"], ["S"], np.int64(3),
                                      plasticity_enabled=learning, record_activation=True)
    for _ in range(3):
        expected = reference.project_into("T", ["s"], ["S"],
                                           plasticity_enabled=learning, record_activation=True)
    np.testing.assert_array_equal(result.winners, expected.winners)
    assert result.num_ever_fired == expected.num_ever_fired
    assert result.pre_kwta_count == expected.pre_kwta_count
    assert result.pre_kwta_total == expected.pre_kwta_total
    assert result.total_activation == expected.total_activation


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_sparse", "numpy_explicit"])
@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_engine_rejects_invalid_rounds_before_any_step(engine, rounds, monkeypatch):
    b = make_brain(engine)
    calls = []
    monkeypatch.setattr(b._engine, "project_into", lambda *a, **kw: calls.append(1))
    with pytest.raises(ValueError, match="positive integer"):
        b._engine.project_rounds("T", ["s"], ["S"], rounds)
    assert calls == []
