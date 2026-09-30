"""Score the ORDERING, not the scale.

Why `diagnostics.separation` exists, and what it is guarding against.

THE FAILURE IT REPLACES. This repo's ERP package scored its grammatical /
violation contrast with Cohen's d computed on
``p600_excess = max(0, v - baseline.p600_median)``, where the baseline IS the
grammatical median. The null arm was therefore clipped against itself onto an
exact 0.0 floor with almost no variance, and Cohen's d divides by a pooled sd
dominated by that floor. Measured across four encodings of an IDENTICAL
ordering::

    grows,     raw p600        AUC 1.000    Cohen's d  2.241
    read_only, raw p600        AUC 1.000    Cohen's d  4.691
    grows,     clipped excess  AUC 1.000    Cohen's d  2.826
    read_only, clipped excess  AUC 1.000    Cohen's d 24.754

An 11x range on the same separation. Reducing measurement noise inflated d
without the effect growing at all.

THE DEEPER REASON, and why a rank statistic is the right default here. P600 was
originally unbounded post-k-WTA churn (grammatical 0.12 vs violation 5.24) and
was later replaced with an energy deficit bounded in [0,1] (0.989 vs 0.995).
Every absolute threshold in the package silently became meaningless -- the
"empirical" P600 margin still resolves to a hardcoded 0.076 against a maximum
observed excess of 0.0064, so the violation detector cannot fire. AUC is
invariant under every monotone transform, so it would have survived that
redefinition untouched.
"""

from __future__ import annotations

import math

import pytest

from neural_assemblies.diagnostics import separation


class TestInvarianceIsThePoint:

    def test_auc_is_unchanged_by_a_monotone_transform(self):
        """The property the whole design rests on: rescaling, shifting, or
        redefining the quantity cannot move a rank statistic."""
        lo = [0.9881, 0.9930, 0.9881]
        hi = [0.9935, 0.9951, 0.9946]
        base = separation(hi, lo, "raw").auc
        for f in (lambda v: v * 1000,          # rescale
                  lambda v: v - 0.99,          # shift
                  lambda v: v ** 3,            # nonlinear, monotone
                  lambda v: math.log(v)):
            assert separation([f(v) for v in hi],
                              [f(v) for v in lo], "t").auc == base

    def test_auc_survives_the_clipping_that_broke_cohens_d(self):
        """Clipping the null arm at its own median is monotone, so AUC does not
        move -- which is exactly why it is the right statistic here."""
        lo_raw, hi_raw = [0.9881, 0.9930, 0.9881], [0.9935, 0.9951, 0.9946]
        med = 0.9914
        clip = lambda v: max(0.0, v - med)          # noqa: E731
        assert separation(hi_raw, lo_raw, "raw").auc == pytest.approx(
            separation([clip(v) for v in hi_raw],
                       [clip(v) for v in lo_raw], "clipped").auc)


class TestSemantics:

    def test_null_is_one_half(self):
        assert separation([1, 2, 3], [1, 2, 3], "same").auc == 0.5

    def test_perfect_and_inverted(self):
        assert separation([4, 5, 6], [1, 2, 3], "hi").auc == 1.0
        assert separation([1, 2, 3], [4, 5, 6], "lo").auc == 0.0

    def test_ties_score_one_half(self):
        assert separation([1.0], [1.0], "tie").auc == 0.5

    def test_one_misordered_pair_out_of_nine(self):
        """The resolution limit is worth seeing: with 3 vs 3 samples the AUC
        can only take values in steps of 1/9, so 0.889 IS one bad pair."""
        s = separation([2, 3, 4], [1, 2.5, 1.5], "coarse")
        assert s.auc == pytest.approx(8 / 9, abs=1e-9)


class TestSaturationIsReportedSeparately:

    def test_perfect_ordering_can_still_be_saturated(self):
        """AUC deliberately ignores magnitude, so `span` carries it. A perfect
        ordering across 0.8% of the range is BOTH facts and neither should hide
        the other -- this is the real ERP p600 case."""
        s = separation([0.9953, 0.9952, 0.9949],
                       [0.9879, 0.9911, 0.9879], "p600")
        assert s.perfect
        assert s.saturated
        assert s.span < 0.01

    def test_a_healthy_metric_is_not_flagged(self):
        s = separation([0.9, 0.8, 0.85], [0.2, 0.1, 0.3], "healthy")
        assert s.perfect
        assert not s.saturated

    def test_unbounded_quantities_report_no_span(self):
        s = separation([5.2, 5.4], [0.1, 0.2], "churn", bounded=False)
        assert math.isnan(s.span)
        assert not s.saturated, "a NaN span must not read as saturated"


class TestRefusesAnEmptyArm:

    @pytest.mark.parametrize("hi,lo", [([], [1, 2]), ([1, 2], []), ([], [])])
    def test_empty_arm_raises_rather_than_scoring_zero(self, hi, lo):
        """An empty condition is a protocol failure. Returning 0.0 or 0.5 would
        make a run that collected no samples look like a measured null."""
        with pytest.raises(ValueError, match="BOTH conditions"):
            separation(hi, lo, "empty")


class TestErpReportCarriesIt:

    def test_calibration_reports_auc_and_span(self, forked_parser):
        from neural_assemblies.assembly_calculus.emergent.evaluation import (
            calibrate_erp_thresholds,
        )
        sep = calibrate_erp_thresholds(
            forked_parser("SENTENCES", seed=11)).separation
        for key in ("p600_auc", "p600_span", "n400_auc", "n400_span"):
            assert key in sep, f"{key} missing -- the honest statistic is not reported"
        assert 0.0 <= sep["p600_auc"] <= 1.0
