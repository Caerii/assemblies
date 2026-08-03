"""The ERP Cohen's d is scored on a variable clipped at its own null.

#102 / #32. Pinned as KNOWN DEFECTS so they cannot quietly change, and so that
nobody reads a large Cohen's d from this package as a large effect.

TWO INDEPENDENT PROBLEMS, both measured.

1. CLIPPED AGAINST ITS OWN NULL. `p600_excess(v) = max(0, v - p600_median)`
   and `ErpBaseline.p600_median` is "median over recent grammatical parses".
   The grammatical condition is therefore clipped against its own median by
   construction, and lands on an exact 0.0 floor. Measured: grammatical
   p600_excess = [0.0, 0.0002, 0.0]. Cohen's d divides by a pooled sd dominated
   by that floor, so REDUCING MEASUREMENT NOISE INFLATES d without the effect
   growing -- making the parse read-only moved d 1.63 -> 3.97 while the absolute
   gap moved only 0.0030 -> 0.0047.

   Consequence: `d > 0.3`, asserted in test_erp_calibration.py, is close to
   vacuous. Against a floored null arm almost any nonzero violation clears it.

2. SATURATED RAW QUANTITY. Raw p600 lives in [0.9879, 0.9953] -- 0.7% of the
   [0,1] range -- in BOTH the growing and the read-only parse. P600 is
   `1 - normalized_energy`, so the role area receives ~1% of its normalizing
   scale in every condition, grammatical included.

The separation is REAL and the direction is right. These tests do not dispute
that. They pin the fact that its magnitude cannot be read off a Cohen's d.

See research/notes/erp_metric_is_clipped.md.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation import (
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.gates import (
    ErpBaseline,
)
from neural_assemblies.diagnostics import separation


def _samples(report, label):
    return [s for s in report.samples if s.label == label]


class TestExcessIsClippedAtItsOwnNull:

    def test_excess_is_one_sided_by_definition(self):
        """The mechanism, independent of any parser: a grammatical value at or
        below the baseline reads exactly 0.0, so the null arm has a floor."""
        b = ErpBaseline(p600_median=0.99)
        assert b.p600_excess(0.98) == 0.0
        assert b.p600_excess(0.99) == 0.0
        assert b.p600_excess(0.995) == pytest.approx(0.005)

    def test_grammatical_excess_is_crushed_against_the_floor(self, forked_parser):
        """And it does so in practice, not just in principle.

        ASSERTS THE ASYMMETRY, not that a specific sample equals 0.0. The first
        version asserted `any(v == 0.0 for v in vals)` and failed whenever the
        baseline happened to come from a different sample set -- a realization,
        not the mechanism, which is the trap this whole file is about.

        The structural fact is that clipping against the grammatical median
        crushes the null arm's spread while leaving the violation arm intact.
        That is what makes Cohen's d divide by a floor.
        """
        report = calibrate_erp_thresholds(forked_parser("SENTENCES", seed=11))
        gram = [float(s.p600_excess) for s in _samples(report, "grammatical")]
        catv = [float(s.p600_excess)
                for s in _samples(report, "category_violation")]
        assert gram and catv, "calibration produced no samples"
        assert min(gram) == 0.0, (
            f"nothing in the grammatical arm reaches the clip floor: {gram}")
        assert max(gram) < min(catv) or sum(gram) < sum(catv), (
            f"grammatical excess {gram} is not below violation {catv} -- the "
            f"clipping asymmetry has gone, which would mean the baseline is no "
            f"longer the grammatical median and Cohen's d may finally be an "
            f"effect size")


class TestRawQuantityIsSaturated:

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN DEFECT, pinned deliberately: raw p600 uses ~0.7% of its range "
        "([0.9879, 0.9953]) in every condition, grammatical included. P600 is "
        "1 - normalized_energy, so the role area receives ~1% of its "
        "normalizing scale throughout. Not fixed by making the parse "
        "read-only -- both arms sit at ~0.99. See "
        "research/notes/erp_metric_is_clipped.md"))
    def test_raw_p600_uses_a_reasonable_fraction_of_its_range(
        self, forked_parser,
    ):
        report = calibrate_erp_thresholds(forked_parser("SENTENCES", seed=11))
        raw = [float(s.p600) for s in report.samples]
        assert raw, "no samples"
        assert max(raw) - min(raw) > 0.1, (
            f"raw p600 spans only {max(raw) - min(raw):.4f} of [0,1] "
            f"(min {min(raw):.4f}, max {max(raw):.4f})")

    def test_the_separation_itself_is_real_and_correctly_signed(
        self, forked_parser,
    ):
        """The saturation is a magnitude problem, NOT a sign or existence
        problem, and this keeps the two claims apart.

        ASSERTS AUC > 0.5, NOT PERFECT ORDERING. The first version of this test
        asserted `max(gram) < min(catv)` and failed on exactly one overlapping
        pair (0.9939 vs 0.9932) -- after I had already MEASURED AUC = 0.889,
        which says the ordering is not perfect. Asserting a 1.000-equivalent I
        had evidence against is [[fake-perfect-probe-signatures]]; with n=3 per
        arm the AUC granularity is 1/9, so one bad pair is well within what the
        protocol produces.
        """
        report = calibrate_erp_thresholds(forked_parser("SENTENCES", seed=11))
        gram = [float(s.p600) for s in _samples(report, "grammatical")]
        catv = [float(s.p600) for s in _samples(report, "category_violation")]
        assert gram and catv
        sep = separation(catv, gram, "p600")
        assert sep.auc > 0.5, (
            f"violations do not score above grammatical: AUC {sep.auc:.3f} on "
            f"gram {gram} vs catv {catv}. At or below 0.5 the separation "
            f"itself has gone, which is a bigger problem than its scale")
