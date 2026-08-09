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


# --------------------------------------------------------------------------
# gain_stability -- the confound that has now bitten three times
# --------------------------------------------------------------------------

class TestGainStability:
    """Synthetic on purpose. The real check costs ~20 minutes of engine time;
    what needs pinning here is the DECISION RULE, and a closed-form effect
    makes the true-positive and true-negative cases exact rather than sampled.
    The measured case it encodes is in
    `research/notes/ceiling_n_scaling_on_exact_drive.md`.
    """

    #: The measured capacity exponent at three gains (2026-08-02). The design
    #: held beta and T fixed while n varied, which is the known-broken one.
    MEASURED = {1.34: 1.55, 1.77: 1.65, 2.99: 0.87}
    FLOOR = 0.20      #: measured estimator scatter, not a chosen threshold

    def test_flags_the_case_that_was_actually_an_artifact(self):
        from neural_assemblies.diagnostics import gain_stability

        gs = gain_stability(lambda g: self.MEASURED[g], tuple(self.MEASURED),
                            noise_floor=self.FLOOR, label="capacity exponent")
        assert gs.confounded and not gs.inconclusive, str(gs)
        assert gs.spread == pytest.approx(0.78, abs=1e-9)

    def test_does_not_flag_a_gain_independent_effect(self):
        """Without this the check could flag everything and mean nothing."""
        from neural_assemblies.diagnostics import gain_stability

        gs = gain_stability(lambda g: 1.42 + 0.02 * (g > 2), (1.34, 1.77, 2.99),
                            noise_floor=self.FLOOR, label="stable effect")
        assert not gs.confounded, str(gs)

    def test_a_result_near_the_noise_floor_is_inconclusive_either_way(self):
        """The first real run said 'not confounded' at 0.29 vs a guessed 0.30.
        A margin that thin is a coin flip and must not read as a verdict."""
        from neural_assemblies.diagnostics import gain_stability

        gs = gain_stability(lambda g: {1.0: 0.00, 2.0: 0.21}[g], (1.0, 2.0),
                            noise_floor=self.FLOOR)
        assert gs.inconclusive, str(gs)
        assert "INCONCLUSIVE" in str(gs)

    def test_refuses_to_run_without_a_measured_noise_floor(self):
        from neural_assemblies.diagnostics import gain_stability

        with pytest.raises(ValueError, match="noise_floor must be measured"):
            gain_stability(lambda g: 1.0, (1.0, 2.0), noise_floor=0.0)

    def test_one_gain_is_not_a_stability_check(self):
        from neural_assemblies.diagnostics import gain_stability

        with pytest.raises(ValueError, match="at least two gains"):
            gain_stability(lambda g: 1.0, (1.0,), noise_floor=0.2)


class TestVerifyProbe:
    """`verify_probe` refuses an instrument that cannot produce both answers.

    The three cases below are not hypothetical -- they are the three probes
    written in one evening, all on the same object, all reading exactly 1.000
    for three DIFFERENT reasons. Two were caught by disbelieving a round
    number; the third shipped, and a published result had to be retracted.
    """

    def test_accepts_a_probe_that_separates(self):
        from neural_assemblies.diagnostics import verify_probe
        c = verify_probe(lambda: 0.9, lambda: 0.1, label="sane")
        assert c.discriminating and c.separation == pytest.approx(0.8)

    def test_refuses_the_index_tie_break_shape(self):
        """Untrained weights all tie, so k-WTA returns the same set either way."""
        from neural_assemblies.diagnostics import verify_probe
        with pytest.raises(AssertionError, match="cannot discriminate"):
            verify_probe(lambda: 1.0, lambda: 1.0, label="tie-break")

    def test_refuses_the_saturated_shape(self):
        """0.846 against 0.840 is a difference between two ceilings."""
        from neural_assemblies.diagnostics import verify_probe
        with pytest.raises(AssertionError, match="cannot discriminate"):
            verify_probe(lambda: 0.846, lambda: 0.840, label="saturated")

    def test_refuses_an_inverted_probe(self):
        """Reading LOW where it must read HIGH is worse than not separating."""
        from neural_assemblies.diagnostics import verify_probe
        with pytest.raises(AssertionError, match="cannot discriminate"):
            verify_probe(lambda: 0.1, lambda: 0.9, label="inverted")

    def test_the_separation_bar_is_caller_supplied(self):
        """A real effect smaller than the blunt default must still be statable
        -- but stated BEFORE the data, for the reason gain_stability's
        noise_floor is mandatory."""
        from neural_assemblies.diagnostics import verify_probe
        c = verify_probe(lambda: 0.55, lambda: 0.50, min_separation=0.04,
                         label="fine-grained")
        assert c.discriminating

    def test_it_reports_both_readings_for_the_log(self):
        from neural_assemblies.diagnostics import verify_probe
        c = verify_probe(lambda: 0.7, lambda: 0.2, label="probe")
        assert "0.7000" in str(c) and "0.2000" in str(c)


class TestRankStatistics:
    """Promoted from research/experiments/overlap_ceiling.py (#149 literate
    pass) after six experiments imported them from an experiment file.
    Pinned: tie handling, a hand-checkable coefficient, and NaN-on-constant
    (an OUTCOME to report, never to filter -- the undefinedness lesson)."""

    def test_rankdata_averages_ties(self):
        from neural_assemblies.diagnostics import rankdata
        assert rankdata([3, 1, 4, 1, 5]).tolist() == [3.0, 1.5, 4.0, 1.5, 5.0]

    def test_spearman_hand_value(self):
        from neural_assemblies.diagnostics import spearman
        # One adjacent transposition in each half of a 5-permutation:
        # rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 6*4/120 = 0.8
        assert abs(spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) - 0.8) < 1e-12

    def test_spearman_nan_on_constant_input(self):
        import math
        from neural_assemblies.diagnostics import spearman
        assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))

    def test_partial_spearman_removes_the_confound(self):
        from neural_assemblies.diagnostics import partial_spearman, spearman
        # y == z: controlling for z must destroy the raw correlation. The
        # residuals are fp noise, not exact zeros, so the result is a small
        # spurious number (measured -0.027 here), not 0.0 and not NaN --
        # pin "far below raw", the claim actually used by E11's analysis.
        x = [1, 2, 3, 4, 5]
        z = [2, 1, 4, 3, 5]
        assert spearman(x, z) > 0.7
        assert abs(partial_spearman(x, z, z)) < 0.1

    def test_experiment_reexport_is_the_same_object(self):
        """`from overlap_ceiling import spearman` must keep reproducing the
        committed E-series scripts -- via re-export, not a second copy."""
        import importlib.util
        import os
        from neural_assemblies import diagnostics
        path = os.path.join(os.path.dirname(diagnostics.__file__), "..",
                            "research", "experiments", "overlap_ceiling.py")
        if not os.path.exists(path):
            pytest.skip("research/ not present in this checkout")
        spec = importlib.util.spec_from_file_location("_oc_reexport", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.spearman is diagnostics.spearman
        assert mod.partial_spearman is diagnostics.partial_spearman
