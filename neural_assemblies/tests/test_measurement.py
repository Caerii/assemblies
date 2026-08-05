"""`Measured` must make an undefined reading UNUSABLE, not merely flagged.

The value of this type is entirely in what it REFUSES. A version that stored
`defined` and let `float()` succeed anyway would be the existing opt-in
`.trustworthy` convention with extra ceremony -- and that convention is what the
P600 and phrase-stability defects walked straight past. So these tests assert
the refusals, not the storage.
"""
from __future__ import annotations

import math

import pytest

from neural_assemblies.core.measurement import (
    Measured,
    UndefinedMeasurement,
    defined_values,
)


class TestDefinedReadingsBehaveLikeFloats:

    def test_float_and_arithmetic_pass_through(self):
        m = Measured.of(0.25)
        assert float(m) == 0.25
        assert m + 1 == 1.25
        assert m * 4 == 1.0
        assert m < 1.0 and m > 0.0

    def test_of_is_explicit_about_wrapping(self):
        assert Measured.of(3).value == 3.0
        assert Measured.of(3).defined


class TestUndefinedReadingsRefuseToBeNumbers:
    """Each of these is a real defect shape from this repo."""

    def test_float_raises_with_the_reason(self):
        m = Measured.undefined("VP has no self-fiber")
        with pytest.raises(UndefinedMeasurement, match="no self-fiber"):
            float(m)

    def test_comparison_raises_rather_than_answering(self):
        """`energy < threshold` on a dead fiber is the P600 bug verbatim.

        Python would happily answer this for a bare float, which is how a
        constant 1.0 passed as a measurement for months.
        """
        m = Measured.undefined("fiber never materialized")
        with pytest.raises(UndefinedMeasurement):
            _ = m < 0.5
        with pytest.raises(UndefinedMeasurement):
            _ = m >= 0.5

    def test_arithmetic_raises_in_both_directions(self):
        m = Measured.undefined("no phrase areas were active")
        with pytest.raises(UndefinedMeasurement):
            _ = m + 1.0
        with pytest.raises(UndefinedMeasurement):
            _ = 1.0 - m

    def test_value_is_nan_so_a_bypass_degrades_loudly(self):
        """Belt and braces: reading `.value` directly still cannot look real.

        0.0 and 1.0 are the dangerous defaults because both read as findings at
        an end of the range. NaN reads as broken, which is the correct
        impression when the type has been circumvented.
        """
        assert math.isnan(Measured.undefined("x").value)

    def test_why_is_carried_not_just_a_flag(self):
        m = Measured.undefined("pool 30 <= k 30, no contest possible")
        assert "pool 30" in m.why
        assert "UNDEFINED" in str(m)


class TestTheSanctionedEscape:

    def test_or_else_makes_the_default_visible(self):
        assert Measured.undefined("dead fiber").or_else(1.0) == 1.0
        assert Measured.of(0.4).or_else(1.0) == 0.4


class TestAggregation:

    def test_defined_values_drops_undefined_rather_than_averaging_them(self):
        """Averaging a mixture is the aggregate form of the same defect."""
        ms = [Measured.of(1.0), Measured.undefined("dead"), Measured.of(3.0)]
        assert defined_values(ms) == [1.0, 3.0]

    def test_caller_can_see_how_many_were_dropped(self):
        """A mean over 2 of 9 probes is a different claim from a mean over 9."""
        ms = [Measured.of(1.0)] + [Measured.undefined("dead")] * 8
        kept = defined_values(ms)
        assert len(kept) == 1 and len(ms) == 9
