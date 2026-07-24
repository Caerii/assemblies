"""Empirical ERP calibration tests (composed-ERP conditions on emergent parser)."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent.evaluation import (
    calibrate_erp_thresholds,
)

N, K = 3000, 30


class TestErpCalibration:
    # These tests all mutate their parser (calibration writes thresholds), so
    # they take independent forks rather than the shared cached object. The
    # underlying curriculum training is still paid once per session.
    def test_calibration_separates_category_violation_from_grammatical(
        self, forked_parser,
    ):
        parser = forked_parser("SENTENCES", seed=42)
        report = calibrate_erp_thresholds(parser)
        assert report.readiness.p600_ready
        assert report.thresholds.source == "empirical"

        gram = report.by_label.get("grammatical", {})
        catv = report.by_label.get("category_violation", {})
        assert gram.get("n", 0) >= 2
        assert catv.get("n", 0) >= 2

        assert catv["p600_excess_median"] > gram["p600_excess_median"]
        assert report.separation.get("p600_cohens_d", 0) > 0.3

    def test_tuned_thresholds_flag_catviol_not_grammatical(self, forked_parser):
        parser = forked_parser("TWO_WORD", seed=7)
        report = calibrate_erp_thresholds(parser)
        gram_wobbly = sum(
            1 for s in report.samples
            if s.label == "grammatical" and s.violation and s.violation.wobbly
        )
        catv_wobbly = sum(
            1 for s in report.samples
            if s.label == "category_violation" and s.violation and s.violation.wobbly
        )
        assert gram_wobbly <= catv_wobbly

    def test_fast_calibration_preserves_separation(self, forked_parser):
        # Two SEPARATE forks on purpose: calibrating one must not contaminate
        # the other, since the whole point is comparing full against fast on
        # identically-trained parsers.
        parser = forked_parser("SENTENCES", seed=42)
        full = calibrate_erp_thresholds(parser, fast=False)
        parser2 = forked_parser("SENTENCES", seed=42)
        fast = calibrate_erp_thresholds(parser2, fast=True)
        assert fast.separation.get("p600_cohens_d", 0) > 0.3
        assert full.separation.get("p600_cohens_d", 0) > 0.3
        assert fast.thresholds.p600_excess_margin == full.thresholds.p600_excess_margin
        assert fast.thresholds.n400_excess_margin == full.thresholds.n400_excess_margin
