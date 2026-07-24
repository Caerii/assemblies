"""Smoke tests for ventral push phase 2."""

import pytest

from neural_assemblies.programs.colt_mnist_ventral_push import (
    evaluate_bundle_accuracy,
    run_ventral_push_phase2,
)
from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_recurrent_bundle


@pytest.fixture
def smoke_kw():
    return {"seed": 42, "n_examples": 8, "k": 200}


def test_evaluate_bundle_accuracy(smoke_kw):
    clear_ventral_bundle_cache()
    bundle = load_recurrent_bundle(use_cache=False, **smoke_kw)
    _, mean_acc, conf, other = evaluate_bundle_accuracy(bundle)
    assert mean_acc > 0.5
    assert conf >= 0.0
    assert other >= 0.0


def test_ventral_push_phase2_runs(smoke_kw):
    clear_ventral_bundle_cache()
    rows = run_ventral_push_phase2(
        learn_assembly_epochs=3,
        pair_passes=2,
        **smoke_kw,
    )
    assert len(rows) >= 4
    assert rows[0].name == "recurrent_baseline"
    assert max(r.mean_accuracy for r in rows) > 0.5
