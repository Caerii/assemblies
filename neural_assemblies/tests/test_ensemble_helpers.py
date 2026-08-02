"""Guards that encode four measurement errors made in one session.

Three had the same shape -- a number that looked like a result was a mechanism
that never ran. These tests exist so the guards cannot be quietly removed.
"""
import pytest

from neural_assemblies.diagnostics import (Ensemble, compare_arms, ensemble,
                                           paired_delta)

SEEDS = list(range(42, 52))


def test_identical_arms_raise():
    """The dead-fiber case: two arms, one computation, four-decimal agreement."""
    with pytest.raises(ValueError, match="IDENTICAL"):
        compare_arms({"intervention": lambda s: float(s),
                      "control": lambda s: float(s)}, SEEDS)


def test_genuinely_different_arms_pass():
    out = compare_arms({"a": lambda s: float(s),
                        "b": lambda s: float(s) + 1.0}, SEEDS)
    assert out["b"].mean - out["a"].mean == pytest.approx(1.0)


def test_single_seed_is_refused():
    """A single-seed before/after is not a measurement."""
    with pytest.raises(ValueError, match="confidence interval"):
        ensemble(lambda s: 1.0, [42])


def test_beats_uses_the_confidence_bound_not_the_mean():
    """The next-token failure: a point estimate that happens to clear a bar.

    Mean 0.0904 against chance 0.0900 "beats" it only if you read the point
    estimate. The ensemble mean was AT chance.
    """
    vals = [0.1165, 0.1041, 0.0909, 0.0745, 0.0551, 0.0732,
            0.1020, 0.0991, 0.1220, 0.0850]
    e = ensemble(lambda s: vals[s - 42], SEEDS, "next-token")
    assert e.mean > 0.0900, "precondition: the POINT estimate clears chance"
    assert not e.beats(0.0900), "but the confidence bound must not"
    assert e.indistinguishable_from(0.0900)


def test_paired_delta_uses_per_seed_differences():
    """Comparing a difference against ONE arm's sd understates it by ~sqrt(2)."""
    a = ensemble(lambda s: 0.20 + (s % 5) * 0.01, SEEDS, "a")
    b = ensemble(lambda s: 0.10 + (s % 5) * 0.01, SEEDS, "b")
    d = paired_delta(a, b)
    assert d.mean == pytest.approx(0.10)
    # Correlated arms: the paired difference is far tighter than either arm.
    assert d.ci < a.ci


def test_paired_delta_requires_matching_seeds():
    a = ensemble(lambda s: 1.0 * s, SEEDS, "a")
    b = ensemble(lambda s: 1.0 * s, SEEDS[:5], "b")
    with pytest.raises(ValueError, match="same seeds"):
        paired_delta(a, b)
