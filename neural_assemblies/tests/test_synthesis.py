"""Smoke tests for AC synthesis scorecard."""

from __future__ import annotations

import pytest

from neural_assemblies.programs.colt_mnist_forward_completion import encode_and_predict
from neural_assemblies.programs.colt_mnist_synthesis import audit_bundle
from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_recurrent_bundle


@pytest.fixture
def smoke_kw():
    return dict(seed=42, n_examples=10, use_cache=False)


def test_encode_and_predict_runs(smoke_kw):
    clear_ventral_bundle_cache()
    b = load_recurrent_bundle(**smoke_kw)
    pred, hv, meta = encode_and_predict(b, b.examples[3, 0])
    assert 0 <= pred <= 9
    assert hv.shape[0] == b.brain.areas["HIGH"].n
    assert meta["head"] in ("discriminative", "generative", "auto")


def test_synthesis_audit_smoke(smoke_kw):
    clear_ventral_bundle_cache()
    b = load_recurrent_bundle(**smoke_kw)
    card = audit_bundle(b, label="recurrent", seed=42)
    assert card.discriminative_accuracy >= 0.45
    assert card.pattern_complete_recovery >= 0.05
    assert card.narrative


def test_emergent_absence_training(smoke_kw):
    clear_ventral_bundle_cache()
    b = load_recurrent_bundle(absence_exposure_prob=0.2, **smoke_kw)
    assert b.parameters.get("absence_exposure_prob") == 0.2
    assert b.parameters.get("generative_prototypes") is not None
    card = audit_bundle(b, label="emergent", seed=42)
    assert card.forward_center_band_overlap >= 0.0
    assert card.generative_head_available
