"""A constant that outlives its quantity must fail loudly.

TIER D of research/plans/TYPE_SAFETY_PROGRAM.md, tested on the real defect that
motivated it.

THE CASE ON RECORD. `P600_EXCESS_MARGIN = 0.152` was calibrated against an
unbounded cumulative P600 (grammatical ~0.12, violation ~5.24). The quantity was
later replaced with `1 - normalized_energy`, bounded in [0,1] and living at
0.989 vs 0.995. The constant was not revisited. Measured result: the margin
resolves to 0.076 against a maximum observed excess of 0.0064 -- 11.9x -- so the
violation detector has never fired on any seed, while the calibration reports
`source="empirical"`.

Nothing connected the constant to the quantity, so nothing could notice. These
tests pin that both halves of that connection now exist and both are able to
fail.
"""

from __future__ import annotations

import pytest

from neural_assemblies.core.metric import (
    Metric,
    MetricRangeError,
    Threshold,
    warn_if_unreachable,
)

# The two P600 definitions, as they actually were.
P600_V1 = Metric("p600", lo=0.0, hi=10.0, version=1,
                 definition="unbounded cumulative post-kWTA churn")
P600_V2 = Metric("p600", lo=0.0, hi=1.0, version=2,
                 definition="1 - normalized pre-kWTA energy, bounded [0,1]")

# Values measured on the real parser, seed 11, fresh cache.
OBSERVED_EXCESS = [0.0, 0.0002, 0.0, 0.0044, 0.0043, 0.0040]
OBSERVED_RAW = [0.9879, 0.9911, 0.9879, 0.9953, 0.9952, 0.9949]


class TestARedefinitionBreaksItsConstants:

    def test_threshold_pinned_to_an_old_version_refuses_to_construct(self):
        """The mechanism: bump the metric version and every stale constant
        fails at import rather than silently gating nothing."""
        with pytest.raises(MetricRangeError, match="quantity changed"):
            Threshold("p600_excess_margin", 0.152,
                      for_metric=P600_V2, metric_version=1)

    def test_the_same_constant_is_fine_against_the_metric_it_was_made_for(self):
        t = Threshold("p600_excess_margin", 0.152,
                      for_metric=P600_V1, metric_version=1)
        assert t.value == 0.152

    def test_the_message_says_what_to_do(self):
        with pytest.raises(MetricRangeError, match="Re-derive it"):
            Threshold("m", 0.152, for_metric=P600_V2, metric_version=1)


class TestUnreachableThresholdIsDetectable:

    def test_the_real_dead_threshold_is_flagged(self):
        """0.076 against a max observed excess of 0.0044 -- the live defect."""
        t = Threshold("p600_excess_margin", 0.076, for_metric=P600_V2,
                      metric_version=2)
        msg = t.audit(OBSERVED_EXCESS, where="calibration")
        assert msg is not None
        assert "cannot fire" in msg
        assert "17.3x" in msg or "x the LARGEST" in msg

    def test_a_reachable_threshold_is_silent(self):
        """The check must be able to stay quiet, or it is not evidence."""
        t = Threshold("p600_excess_margin", 0.002, for_metric=P600_V2,
                      metric_version=2)
        assert t.audit(OBSERVED_EXCESS) is None

    def test_all_zero_observations_are_called_out_separately(self):
        """A different failure from 'threshold too high': nothing is being
        produced at all, which is a dead mechanism upstream."""
        t = Threshold("m", 0.05, for_metric=P600_V2, metric_version=2)
        msg = t.audit([0.0, 0.0, 0.0])
        assert msg is not None and "every observation" in msg

    def test_no_observations_is_not_a_verdict(self):
        t = Threshold("m", 0.05, for_metric=P600_V2, metric_version=2)
        assert t.audit([]) is None

    def test_warn_variant_does_not_raise(self):
        """Call sites during warm-up must not explode; a threshold can be
        legitimately unreachable before its mechanism is ready."""
        t = Threshold("m", 0.076, for_metric=P600_V2, metric_version=2)
        with pytest.warns(RuntimeWarning, match="cannot fire"):
            warn_if_unreachable(t, OBSERVED_EXCESS)


class TestRangeIsCheckedWhereValuesAreProduced:

    def test_out_of_range_observations_raise(self):
        """If P600 v2 ever produces 5.24 again, the quantity reverted and every
        downstream statistic is describing the wrong thing."""
        with pytest.raises(MetricRangeError, match="observations fall outside"):
            P600_V2.check([0.99, 5.24], where="adapters")

    def test_in_range_observations_pass(self):
        P600_V2.check(OBSERVED_RAW, where="adapters")

    def test_the_message_carries_the_definition(self):
        with pytest.raises(MetricRangeError, match="bounded \\[0,1\\]"):
            P600_V2.check([5.24])


class TestOccupancyExposesSaturation:

    def test_the_real_p600_reads_as_saturated(self):
        """0.7% of [0,1]. Invisible to any statistic that standardises by the
        observed spread -- which is exactly how Cohen's d reached 24.754 on a
        difference of 0.004."""
        occ = P600_V2.occupancy(OBSERVED_RAW)
        assert occ < 0.01, f"expected a sliver, got {occ:.4f}"

    def test_a_healthy_metric_occupies_its_range(self):
        assert P600_V2.occupancy([0.1, 0.9]) == pytest.approx(0.8)

    def test_empty_is_nan_not_zero(self):
        """Zero occupancy would read as 'maximally saturated' for a metric that
        simply has no data yet."""
        assert P600_V2.occupancy([]) != P600_V2.occupancy([0.5, 0.5])
