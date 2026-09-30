"""Feedforward inhibition is a validated mechanism, never an ignored kwarg."""

import math

import pytest

from neural_assemblies import Brain, FeedforwardInhibitionConfig
from neural_assemblies.core.engine import create_engine


@pytest.mark.parametrize(
    "probability,weight,error",
    [
        (True, -0.2, "finite real numbers"),
        (math.nan, -0.2, "finite real numbers"),
        (-0.1, -0.2, r"probability must be in \[0, 1\]"),
        (1.1, -0.2, r"probability must be in \[0, 1\]"),
        (0.2, 0.0, "weight must be negative"),
        (0.2, 1.0, "weight must be negative"),
        (0.0, -1.0, "weight is inert"),
    ],
)
def test_invalid_inhibition_laws_reject(probability, weight, error):
    with pytest.raises(ValueError, match=error):
        FeedforwardInhibitionConfig(probability, weight)


def test_exact_engine_receives_one_canonical_configuration():
    brain = Brain(
        engine="numpy_exact",
        p=0.1,
        seed=7,
        norm_init=False,
        inhibitory_prob=0.25,
        inhibitory_weight=-0.75,
    )

    assert brain.feedforward_inhibition == FeedforwardInhibitionConfig(0.25, -0.75)
    assert brain._engine.inhibitory_prob == 0.25
    assert brain._engine.inhibitory_weight == -0.75


def test_supplied_engine_must_match_requested_inhibition():
    from neural_assemblies.core.numpy_engine import NumpyExactEngine

    engine = NumpyExactEngine(
        p=0.1, seed=7, inhibitory_prob=0.25, inhibitory_weight=-0.75
    )
    with pytest.raises(ValueError, match="feedforward inhibition conflicts"):
        Brain(engine=engine, p=0.1, seed=7, norm_init=False)

    brain = Brain(
        engine=engine,
        p=0.1,
        seed=7,
        norm_init=False,
        inhibitory_prob=0.25,
        inhibitory_weight=-0.75,
    )
    assert brain.feedforward_inhibition == FeedforwardInhibitionConfig(0.25, -0.75)


def test_exact_inhibition_reverses_area_to_area_drive():
    """The admitted mechanism must move the observation past its true null."""
    totals = {}
    for probability in (0.0, 1.0):
        inhibition = {"inhibitory_prob": probability}
        if probability:
            inhibition["inhibitory_weight"] = -0.75
        brain = Brain(
            engine="numpy_exact",
            p=1.0,
            seed=7,
            norm_init=False,
            **inhibition,
        )
        brain.add_stimulus("S", 10)
        brain.add_area("A", 100, 10, beta=0.0)
        brain.add_area("B", 100, 10, beta=0.0)
        brain.project({"S": ["A"]}, {})
        brain.record_activation = True
        brain.project({}, {"A": ["B"]})
        totals[probability] = brain.pre_kwta_observation("B").total

    assert totals[0.0] == 1000.0
    assert totals[1.0] == -750.0


def test_sampled_engine_rejects_incomplete_signed_candidate_semantics():
    with pytest.raises(ValueError, match="does not support feedforward inhibition"):
        Brain(engine="numpy_sparse", inhibitory_prob=0.25)

    from neural_assemblies.core.numpy_engine import NumpySparseEngine

    with pytest.raises(TypeError, match="inhibitory_prob"):
        NumpySparseEngine(p=0.1, inhibitory_prob=0.25)


@pytest.mark.parametrize("surface", ["brain", "factory"])
def test_unsupported_dense_inhibition_rejects_before_constructor(monkeypatch, surface):
    import neural_assemblies.core.brain as brain_module
    import neural_assemblies.core.engine as engine_module

    called = False
    engine_module.ensure_engine("numpy_explicit")
    original = engine_module._ENGINE_REGISTRY["numpy_explicit"]

    class RecordingExplicit(original):
        def __init__(self, *args, **kwargs):
            nonlocal called
            called = True
            super().__init__(*args, **kwargs)

    monkeypatch.setitem(
        engine_module._ENGINE_REGISTRY, "numpy_explicit", RecordingExplicit
    )
    with pytest.raises(ValueError, match="does not support feedforward inhibition"):
        if surface == "brain":
            brain_module.Brain(engine="numpy_explicit", inhibitory_prob=0.2)
        else:
            create_engine("numpy_explicit", p=0.1, inhibitory_prob=0.2)
    assert called is False
