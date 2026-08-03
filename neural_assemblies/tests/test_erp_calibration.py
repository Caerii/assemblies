"""Empirical ERP calibration tests (composed-ERP conditions on emergent parser).

ASSERTED ON AUC, NOT COHEN'S D, and the reason is measured rather than stylistic.

Two fresh trainings, SAME seed, backbone cache disabled, separate processes::

    run 1   grammatical [0.9874, 0.9938, 0.9874]   d = 1.452   AUC = 0.889
    run 2   grammatical [0.9872, 0.9941, 0.9872]   d = 1.291   AUC = 0.889

Training is not reproducible across processes (#80), so Cohen's d moves 11%
between runs of identical code -- while the rank statistic is IDENTICAL. A
threshold on d is therefore a threshold on the run, not on the model, and
`d > 0.3` had been failing and passing depending on which other tests ran first.

d is also not an effect size here even when it is stable: it is computed on
`p600_excess = max(0, v - grammatical_median)`, which clips the NULL arm against
its own median onto a 0.0 floor, so reducing measurement noise inflates d
without the effect growing. Across four encodings of one IDENTICAL ordering it
spanned 2.241 to 24.754 while AUC stayed 1.000. See
research/notes/erp_metric_is_clipped.md and `diagnostics.separation`.

WHY THE THRESHOLD IS ONLY "ABOVE CHANCE". Measured p600 AUC at SENTENCES depth:

    seed 42  0.667      seed 11  1.000      seed 12  0.889      seed 13  1.000

so the floor across seeds is 0.667, and with n=3 per arm the AUC granularity is
1/9 -- 0.667 is a single misordered pair. A tighter bound would be pinning one
realization of a quantity whose spread is one granularity step.

NOT ASSERTED AT TWO_WORD DEPTH, deliberately: seed 7 measures AUC 0.000, fully
INVERTED (d = -3.200). The separation is a SENTENCES-depth result and claiming
it earlier would be false. `test_tuned_thresholds_flag_catviol_not_grammatical`
runs there and asserts only the wobbly counts, which is why it is unaffected.
"""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent.evaluation import (
    calibrate_erp_thresholds,
)

N, K = 3000, 30

#: Null for a rank statistic. Violations must out-score grammatical more often
#: than not; see the module docstring for why nothing tighter is asserted.
CHANCE = 0.5


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
        assert report.separation["p600_auc"] > CHANCE, (
            f"p600 AUC {report.separation['p600_auc']:.3f} is not above chance "
            f"-- violations do not out-score grammatical")

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
        assert fast.separation["p600_auc"] > CHANCE
        assert full.separation["p600_auc"] > CHANCE

        # THE THRESHOLDS ARE THE EXACT CLAIM, and they are what "preserves"
        # means operationally: the fast path must derive the same gates.
        assert fast.thresholds.p600_excess_margin == full.thresholds.p600_excess_margin
        assert fast.thresholds.n400_excess_margin == full.thresholds.n400_excess_margin

        # DELIBERATELY NOT ASSERTED: that fast's AUC tracks full's. It is the
        # comparison the test name suggests, and at n=3 per arm the sample
        # cannot support it -- AUC granularity is 1/9 = 0.111 and the measured
        # fast-minus-full delta over three fresh trainings was 0.000, 0.000,
        # -0.167, i.e. 1.5 granularity steps. A bound at 1/9 failed on the third
        # run; a bound loose enough to survive (2/9) would admit any degradation
        # worth catching. The comparison needs more samples per condition, not a
        # wider threshold. Tracked with #80.
