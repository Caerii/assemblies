"""Admission contract for recurrence over a lazy sampled connectome."""

import copy
import warnings

import numpy as np
import pytest

from neural_assemblies import SampledRecurrencePolicy
from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import NumpySparseEngine


def _active_brain(policy="warn"):
    brain = Brain(
        engine="numpy_sparse",
        p=.1,
        seed=17,
        norm_init=False,
        sampled_recurrence_policy=policy,
    )
    brain.add_area("A", 200, 10, beta=.1)
    brain.add_stimulus("s", 10)
    brain.project({"s": ["A"]}, {})
    return brain


@pytest.mark.parametrize("value", [None, True, "", "allow", "FORBID"])
def test_invalid_sampled_recurrence_policy_rejects_at_construction(value):
    with pytest.raises(ValueError, match="sampled_recurrence_policy"):
        Brain(engine="numpy_sparse", sampled_recurrence_policy=value)


def test_policy_enum_is_public_and_canonical():
    brain = _active_brain(SampledRecurrencePolicy.ACKNOWLEDGED)
    assert brain.sampled_recurrence_policy is SampledRecurrencePolicy.ACKNOWLEDGED
    assert brain._engine.sampled_recurrence_policy is (
        SampledRecurrencePolicy.ACKNOWLEDGED
    )
    with pytest.raises(AttributeError):
        brain.sampled_recurrence_policy = SampledRecurrencePolicy.FORBID
    with pytest.raises(AttributeError):
        brain._engine.sampled_recurrence_policy = SampledRecurrencePolicy.FORBID


def test_engine_policy_cannot_change_after_area_registration():
    brain = _active_brain("acknowledged")
    with pytest.raises(RuntimeError, match="cannot change"):
        brain._engine._configure_sampled_recurrence_policy("forbid")


def test_legacy_state_without_policy_retains_the_warning_default():
    brain = _active_brain("acknowledged")
    del brain._sampled_recurrence_policy
    del brain._engine._sampled_recurrence_policy
    assert brain.sampled_recurrence_policy is SampledRecurrencePolicy.WARN
    assert brain._engine.sampled_recurrence_policy is SampledRecurrencePolicy.WARN
    with pytest.warns(RuntimeWarning, match="sampled numpy connectome"):
        brain.project({"s": ["A"]}, {"A": ["A"]})


def test_warn_policy_fires_once_and_names_the_escape_hatch():
    brain = _active_brain()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        brain.project({"s": ["A"]}, {"A": ["A"]})
        brain.project({"s": ["A"]}, {"A": ["A"]})
    messages = [
        str(item.message) for item in caught
        if "sampled numpy connectome" in str(item.message)
    ]
    assert len(messages) == 1
    assert "PREREG_sampler_audit.md" in messages[0]
    assert "sampled_recurrence_policy='acknowledged'" in messages[0]


def test_acknowledged_policy_is_silent_for_deliberate_sampled_execution():
    brain = _active_brain("acknowledged")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        brain.project({"s": ["A"]}, {"A": ["A"]})
    assert not any("sampled numpy connectome" in str(item.message) for item in caught)


def test_forbid_policy_rejects_before_rng_or_connectome_mutation():
    brain = _active_brain("forbid")
    engine = brain._engine
    before_rng = copy.deepcopy(engine._rng.bit_generator.state)
    before_winners = engine.get_winners("A").copy()
    before_count = engine.get_num_ever_fired("A")
    before_weights = engine._area_conns["A"]["A"].weights.copy()

    with pytest.raises(RuntimeError, match="forbidden.*still sampled"):
        brain.project({"s": ["A"]}, {"A": ["A"]})

    assert engine._rng.bit_generator.state == before_rng
    np.testing.assert_array_equal(engine.get_winners("A"), before_winners)
    assert engine.get_num_ever_fired("A") == before_count
    np.testing.assert_array_equal(
        engine._area_conns["A"]["A"].weights,
        before_weights,
    )


def test_forbid_policy_admits_a_materialized_area():
    brain = _active_brain("forbid")
    brain.materialize_area("A")
    brain.project({"s": ["A"]}, {"A": ["A"]})
    assert brain._engine.materialized_count("A") == brain.areas["A"].n


@pytest.mark.parametrize("mode", ["fixed", "read-only"])
def test_forbid_policy_admits_recurrence_that_cannot_sample_candidates(mode):
    brain = _active_brain("forbid")
    if mode == "fixed":
        brain.areas["A"].fix_assembly()
        brain.project({"s": ["A"]}, {"A": ["A"]})
    else:
        with brain.read_only():
            brain.project({"s": ["A"]}, {"A": ["A"]})


def test_supplied_sampled_engine_policy_must_match_before_adoption():
    engine = NumpySparseEngine(
        p=.1,
        seed=17,
        norm_init=False,
        sampled_recurrence_policy="acknowledged",
    )
    with pytest.raises(ValueError, match="conflicts with supplied engine"):
        Brain(
            p=.1,
            seed=17,
            norm_init=False,
            engine=engine,
            sampled_recurrence_policy="forbid",
        )
    assert engine.sampled_recurrence_policy is (
        SampledRecurrencePolicy.ACKNOWLEDGED
    )
