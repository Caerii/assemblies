"""A missing measurement or an absent bar must never pass a golden."""

import pytest

from neural_assemblies.parity.executors import verify_against_golden


@pytest.mark.parametrize("golden", [
    {},
    {"metrics": {"score": 1.0}},
    {"expected": {"score": 1.0}},
    {"thresholds": {"score_min": 0.8}},
    {"thresholds": {"score_max": 0.8}},
    {"thresholds": {"accepted": True}},
    {"thresholds": {"unknown_rule": 0.8}},
    {"metrics": {"score": 1.0}, "thresholds": {"metrics_match_tolerance": 0.1}},
])
def test_incomplete_observation_cannot_pass(golden):
    passed, diffs = verify_against_golden("fixture", {}, golden)
    assert not passed
    assert diffs


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, True])
@pytest.mark.parametrize("suffix", ["min", "max"])
def test_invalid_bound_measurement_fails(value, suffix):
    assert not verify_against_golden(
        "fixture", {"score": value}, {"thresholds": {f"score_{suffix}": 0.8}}
    )[0]


def test_legacy_expected_bound_is_evaluated_and_can_fail():
    golden = {"expected": {"score_min": 0.8}, "thresholds": {"score_min": 0.8}}
    assert verify_against_golden("fixture", {"score": 0.9}, golden)[0]
    assert not verify_against_golden("fixture", {"score": 0.7}, golden)[0]


def test_declared_golden_tolerance_is_evaluated():
    golden = {"metrics": {"score": 0.9},
              "thresholds": {"score_min": 0.5, "metrics_match_tolerance": 0.05}}
    assert verify_against_golden("fixture", {"score": 0.9}, golden)[0]
    assert not verify_against_golden("fixture", {"score": 0.6}, golden)[0]


def test_scaling_comparator_rejects_missing_and_nonfinite_measurements():
    from neural_assemblies.parity.runner import _verify_pnas_scaling

    golden = {"regimes": {"small": {"project_persistence": 1.0}}}
    assert _verify_pnas_scaling({"small": {"project_persistence": 1.0}}, golden)[0]
    assert not _verify_pnas_scaling({}, golden)[0]
    assert not _verify_pnas_scaling({"small": {"project_persistence": float("nan")}}, golden)[0]
    assert not _verify_pnas_scaling({}, {})[0]
