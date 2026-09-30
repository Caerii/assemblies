"""Executable identity for backend choices that change scientific meaning."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from neural_assemblies import (
    ArithmeticMode,
    Brain,
    CandidateDomain,
    ConnectomeMode,
    ModelSemantics,
    NormalizationMode,
    PlasticityRule,
    StimulusDriveLaw,
    TieBreakRule,
)
from neural_assemblies.core.numpy_engine import NumpyExactEngine


def _brain(engine, **kwargs):
    return Brain(engine=engine, norm_init=False, **kwargs)


def test_semantics_are_immutable_and_round_trip_through_wire_mapping():
    semantics = _brain("numpy_exact").model_semantics
    assert ModelSemantics.normalize(semantics.to_dict()) == semantics
    with pytest.raises(FrozenInstanceError):
        semantics.connectome = ConnectomeMode.LAZY_CONTENT_ADDRESSED


@pytest.mark.parametrize(
    "value, error",
    [
        (None, TypeError),
        ({}, ValueError),
        ({"connectome": "lazy-content-addressed"}, ValueError),
    ],
)
def test_incomplete_semantics_never_default_silently(value, error):
    with pytest.raises(error, match="model_semantics"):
        ModelSemantics.normalize(value)


def test_unknown_mapping_field_rejects():
    raw = _brain("numpy_exact").model_semantics.to_dict()
    raw["engine"] = "numpy_exact"
    with pytest.raises(ValueError, match="unknown.*engine"):
        ModelSemantics.normalize(raw)
    raw[1] = "also unknown"
    with pytest.raises(ValueError, match="unknown"):
        ModelSemantics.normalize(raw)


@pytest.mark.parametrize("ceiling", [True, 0, -1, np.nan, np.inf, "20"])
def test_invalid_weight_ceiling_rejects(ceiling):
    raw = _brain("numpy_exact").model_semantics.to_dict()
    raw["weight_ceiling"] = ceiling
    with pytest.raises(ValueError, match="weight_ceiling"):
        ModelSemantics.normalize(raw)


def test_plasticity_rule_and_ceiling_must_agree():
    raw = _brain("numpy_exact").model_semantics.to_dict()
    raw["plasticity"] = "multiplicative-unbounded"
    with pytest.raises(ValueError, match="unbounded.*None"):
        ModelSemantics.normalize(raw)


def test_cpu_engine_profiles_name_their_real_semantic_differences():
    sampled = _brain("numpy_sparse").model_semantics
    dense = _brain("numpy_explicit").model_semantics
    hashed = _brain("numpy_exact").model_semantics

    assert sampled.connectome is ConnectomeMode.LAZY_CONTENT_ADDRESSED
    assert sampled.candidate_domain is (
        CandidateDomain.MATERIALIZED_PLUS_ORDER_STATISTICS
    )
    assert sampled.stimulus_drive is (
        StimulusDriveLaw.LAZY_CONDITIONED_AFFERENT_COUNT
    )
    assert sampled.default_tie_break is TieBreakRule.PARTITION_ORDER

    assert dense.connectome is ConnectomeMode.FIXED_DENSE_CONTENT_ADDRESSED
    assert hashed.connectome is ConnectomeMode.FIXED_HASH_REGENERATED
    for fixed in (dense, hashed):
        assert fixed.candidate_domain is CandidateDomain.ALL_NEURONS
        assert fixed.stimulus_drive is (
            StimulusDriveLaw.FIXED_BERNOULLI_AFFERENT_COUNT
        )
        assert fixed.default_tie_break is TieBreakRule.LOWEST_NEURON_ID


def test_requested_semantics_accept_exact_match_and_reject_one_field_drift():
    expected = _brain("numpy_exact").model_semantics
    assert _brain("numpy_exact", model_semantics=expected).model_semantics == expected
    assert _brain(
        "numpy_exact", model_semantics=expected.to_dict()
    ).model_semantics == expected

    wrong = replace(expected, default_tie_break=TieBreakRule.BACKEND_TOPK_ORDER)
    with pytest.raises(
        ValueError,
        match="default_tie_break.*backend-topk-order.*lowest-neuron-id",
    ):
        _brain("numpy_exact", model_semantics=wrong)


def test_normalization_and_arithmetic_are_part_of_identity():
    raw = _brain("numpy_exact").model_semantics
    normalized = Brain(engine="numpy_exact", norm_init=True).model_semantics
    assert raw.normalization is NormalizationMode.NONE
    assert normalized.normalization is NormalizationMode.INVERSE_INDEGREE

    exact64 = NumpyExactEngine(p=.1, dtype=np.float64)
    assert exact64.describe_model_semantics().arithmetic is ArithmeticMode.FLOAT64
    with pytest.raises(ValueError, match="float32 or float64"):
        NumpyExactEngine(p=.1, dtype=np.float16)


def test_weight_ceiling_is_part_of_model_identity():
    clipped = Brain(engine="numpy_exact", norm_init=False, w_max=20.0)
    unbounded = Brain(engine="numpy_exact", norm_init=False, w_max=None)
    assert clipped.model_semantics.plasticity is (
        PlasticityRule.MULTIPLICATIVE_CLIPPED
    )
    assert unbounded.model_semantics.plasticity is (
        PlasticityRule.MULTIPLICATIVE_UNBOUNDED
    )
    assert clipped.model_semantics.weight_ceiling == 20.0
    assert unbounded.model_semantics.weight_ceiling is None
    different_clip = Brain(engine="numpy_exact", norm_init=False, w_max=7.0)
    with pytest.raises(ValueError, match="weight_ceiling.*20.0.*7.0"):
        Brain(
            engine="numpy_exact", norm_init=False, w_max=7.0,
            model_semantics=clipped.model_semantics,
        )
    assert different_clip.model_semantics.weight_ceiling == 7.0


def test_legacy_brain_reconstructs_semantics_from_its_engine():
    brain = _brain("numpy_exact")
    del brain._model_semantics
    assert brain.model_semantics == brain._engine.describe_model_semantics()


def test_stream_addressed_compatibility_switch_is_visible(monkeypatch):
    expected = _brain("numpy_sparse").model_semantics
    monkeypatch.setenv("ASSEMBLIES_STREAM_INIT", "1")
    semantics = _brain("numpy_sparse").model_semantics
    assert semantics.connectome is ConnectomeMode.LAZY_STREAM_ADDRESSED
    with pytest.raises(ValueError, match="connectome.*lazy-content-addressed"):
        _brain("numpy_sparse", model_semantics=expected)
