"""Smoke tests for H9 diagnostic."""

import pytest

from neural_assemblies.programs.colt_mnist_h9_diagnostic import run_h9_diagnostic
from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache


@pytest.fixture
def smoke_kw():
    return {"seed": 42, "n_examples": 10}


def test_h9_diagnostic_runs(smoke_kw):
    clear_ventral_bundle_cache()
    result = run_h9_diagnostic(**smoke_kw)
    assert result.h9_verdict in ("supported", "partial", "falsified", "inconclusive")
    assert len(result.streams) >= 4
    names = {s.name for s in result.streams}
    assert "recurrent_ventral" in names
    assert "merge_halves" in names


def test_h9_merge_has_lower_or_higher_overlap_than_baseline(smoke_kw):
    clear_ventral_bundle_cache()
    result = run_h9_diagnostic(**smoke_kw)
    baseline = next(s for s in result.streams if s.name == "recurrent_ventral")
    merge = next(s for s in result.streams if s.name == "merge_halves")
    assert baseline.mean_confused_overlap >= 0.0
    assert merge.mean_confused_overlap >= 0.0
    assert "mean_confused" in result.pair_deltas.get("merge_halves_vs_recurrent", {})
