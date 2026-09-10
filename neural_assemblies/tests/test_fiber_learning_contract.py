"""A learning mask keeps drive active while preventing the selected fiber update."""
import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.fixture(params=[False, True])
def brain(request):
    b = Brain(engine="numpy_sparse" if request.param else "numpy_explicit",
              norm_init=False, seed=83)
    for name in ("A", "B", "T"):
        b.add_area(name, 4, 2, .5, explicit=request.param)
    for source in ("A", "B"):
        b.areas[source].winners = [0]
        b.connectomes[source]["T"].weights[:] = 0
    b.connectomes["A"]["T"].weights[0] = [0, 0, 10, 9]
    b.connectomes["B"]["T"].weights[0] = [1, 2, 1, 1]
    return b


def test_mask_preserves_drive_and_blocks_only_selected_learning(brain):
    a = brain.connectomes["A"]["T"].weights.copy()
    b = brain.connectomes["B"]["T"].weights.copy()
    brain.set_fiber_plasticity("A", "T", False)
    brain.project({}, {"A": ["T"], "B": ["T"]})
    assert list(brain.areas["T"].winners) == [2, 3]
    np.testing.assert_array_equal(brain.connectomes["A"]["T"].weights, a)
    b[0, [2, 3]] *= 1.5
    np.testing.assert_array_equal(brain.connectomes["B"]["T"].weights, b)
    brain.set_fiber_plasticity("A", "T", True)
    brain.project({}, {"A": ["T"]})
    assert brain.connectomes["A"]["T"].weights[0, 2] > a[0, 2]


def test_failed_projection_does_not_leave_engine_mask(brain):
    brain.set_fiber_plasticity("A", "T", False)
    with pytest.raises(ValueError):
        brain.project({}, {"A": ["T"]}, external_drive={"T": [1]})
    engine = brain._engine_for(brain.areas["T"])
    assert not getattr(engine, "_suppressed_learning_fibers", ())


def test_backend_without_capability_cannot_silently_ignore_mask(monkeypatch):
    brain = Brain(engine="numpy_sparse", norm_init=False, seed=83)
    brain.add_area("A", 8, 2, .1)
    brain.add_area("T", 8, 2, .1)
    monkeypatch.setattr(brain._engine, "supports_fiber_learning_masks", False)
    brain.set_fiber_plasticity("A", "T", False)
    with pytest.raises(NotImplementedError, match="fiber"):
        brain.project({}, {"A": ["T"]})


def test_nested_scope_and_supervision_share_the_mask(brain):
    engine = brain._engine_for(brain.areas["T"])
    before = brain.connectomes["A"]["T"].weights.copy()
    with engine.suppress_fiber_learning([("A", "T")]):
        assert not engine.fiber_learning_allowed("A", "T")
        with engine.suppress_fiber_learning([("B", "T")]):
            assert not engine.fiber_learning_allowed("A", "T")
            assert not engine.fiber_learning_allowed("B", "T")
            brain.reinforce_connectome("A", "T", [2], beta=1)
        assert engine.fiber_learning_allowed("B", "T")
        assert not engine.fiber_learning_allowed("A", "T")
    assert engine.fiber_learning_allowed("A", "T")
    np.testing.assert_array_equal(before, brain.connectomes["A"]["T"].weights)


def test_stimulus_mask_keeps_drive_without_learning(brain):
    brain.add_stimulus("stim", 4)
    weights = brain.connectomes_by_stimulus["stim"]["T"].weights
    weights[:] = 0
    weights[0] = [0, 0, 10, 9]
    before = weights.copy()
    brain.set_fiber_plasticity("stim", "T", False)
    brain.project({"stim": ["T"]}, {})
    assert list(brain.areas["T"].winners) == [2, 3]
    np.testing.assert_array_equal(weights, before)


def test_ir_cannot_promise_learning_through_an_engine_mask(brain):
    from neural_assemblies.ir.projection import ExplicitRound
    engine = brain._engine_for(brain.areas["T"])
    with engine.suppress_fiber_learning([("A", "T")]):
        with pytest.raises(ValueError, match="learning"):
            ExplicitRound("T", ["A"], True).execute(engine)
