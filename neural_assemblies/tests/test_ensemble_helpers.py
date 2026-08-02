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


# --------------------------------------------------------------------------
# load_audit -- the correction to "paired comparisons survive the sampler"
# --------------------------------------------------------------------------

class TestLoadAudit:
    """Both cases are CONSTRUCTED, because a flag that never says no has no
    measured power. The true positive is the A/B whose answer is already known
    from `research/notes/recurrence_ceiling_on_exact_drive.md` (norm_init on vs
    off reads an 8.0x capacity gain on the sampler and 1.0x on exact drive);
    the true negative is the same protocol at a different seed, which must NOT
    be flagged or the check is just noise.
    """

    N, K, P, BETA, M, ROUNDS = 1000, 50, 0.05, 0.10, 8, 6

    def _built(self, norm_init, seed=42, engine="numpy_sparse"):
        from neural_assemblies.assembly_calculus.ops import project
        from neural_assemblies.core.brain import Brain

        b = Brain(p=self.P, seed=seed, norm_init=norm_init, engine=engine)
        b.add_area("L", self.N, self.K, beta=self.BETA)
        for m in range(self.M):
            b.add_stimulus(f"w{m}", self.K)
        for m in range(self.M):
            project(b, f"w{m}", "L", rounds=self.ROUNDS, recurrent=True)
        return b

    def test_flags_the_ab_that_is_known_to_be_confounded(self):
        from neural_assemblies.diagnostics import load_audit

        gaps = load_audit({"norm_on": self._built(True),
                           "norm_off": self._built(False)})
        assert gaps and gaps[0].area == "L"
        assert gaps[0].confounded(), (
            f"norm_init on/off left the arms at loads "
            f"{gaps[0].by_arm} (gap {gaps[0].gap:.3f}) and was NOT flagged; "
            f"this is the case the check exists for")

    def test_does_not_flag_a_genuine_null(self):
        from neural_assemblies.diagnostics import load_audit

        gaps = load_audit({"seed42": self._built(True, seed=42),
                           "seed7": self._built(True, seed=7)})
        assert not gaps[0].confounded(), (
            f"two seeds of the SAME protocol were flagged as load-confounded "
            f"(gap {gaps[0].gap:.3f}); the threshold is below seed noise and "
            f"the check would flag everything")

    def test_exact_engine_has_no_sampler_to_confound(self):
        """Same arms, engine with no invented drive -> reported, never flagged."""
        from neural_assemblies.diagnostics import load_audit

        gaps = load_audit({"norm_on": self._built(True, engine="numpy_exact"),
                           "norm_off": self._built(False, engine="numpy_exact")})
        assert not gaps[0].sampler_bearing
        assert not gaps[0].confounded()

    def test_one_arm_is_not_a_comparison(self):
        import pytest as _pytest

        from neural_assemblies.diagnostics import load_audit
        with _pytest.raises(ValueError, match="at least two"):
            load_audit({"only": self._built(True)})
