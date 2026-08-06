"""`area_health` must not invent a number for a quantity it did not measure.

THE DEFECT THESE PIN. Every quantity on `AreaHealth` used to default to NaN,
and NaN comparisons return False rather than raising. Two consequences, both
measured before the fix:

  1. `margin` was recorded ONLY when the runner-up assembly had NONZERO overlap
     with the live read. Zero overlap is the BEST possible separation, so the
     best case was the one that produced no margin at all: three disjoint
     assemblies with exact re-cue gave `accuracy 1.0000` and `margin nan`,
     printed as an OK verdict. Any harness averaging margins across seeds was
     therefore averaging only the seeds where separation was WORSE.
  2. `nan > ch + 0.2` is False, so an accuracy that was never measured produced
     the verdict "not discriminable" -- a claim about the AREA -- from a fact
     about the MEASUREMENT.

Each test asserts in BOTH directions where a direction exists: that the
undefined case is undefined AND that the defined case is still defined with the
value it had before. A guard that rejects everything passes a one-sided test.
"""
import math

import numpy as np
import pytest

from neural_assemblies import diagnostics
from neural_assemblies.core.measurement import Measured, UndefinedMeasurement


class _Area:
    n, k = 1000, 10


class _Brain:
    def __init__(self):
        self.areas = {"A": _Area()}


def _health(stored, live_for=None, monkeypatch=None):
    """Run area_health with the live read stubbed to `live_for[key]`."""
    state = {"key": None}
    if live_for is not None:
        monkeypatch.setattr(diagnostics, "read_assembly",
                            lambda brain, area: live_for[state["key"]])
        cues = {key: (lambda key=key: state.__setitem__("key", key))
                for key in stored}
    else:
        cues = None
    return diagnostics.area_health(_Brain(), "A", stored, cues)


DISJOINT = {
    "a": np.arange(0, 10),
    "b": np.arange(100, 110),
    "c": np.arange(200, 210),
}


# --------------------------------------------------------------------------
# margin: the best case must not be the undefined one
# --------------------------------------------------------------------------

def test_perfect_separation_gives_an_unbounded_margin_not_an_undefined_one(
        monkeypatch):
    """best/0 with best > 0 IS infinity. Reporting NaN dropped it from means."""
    h = _health(DISJOINT, DISJOINT, monkeypatch)

    assert h.margin.defined, (
        "perfect separation must yield a DEFINED margin; NaN here is what let "
        "the best seeds be silently dropped from the mean")
    assert math.isinf(float(h.margin))
    assert h.unbounded_margins == len(DISJOINT)
    assert h.unmatched_reads == 0


def test_imperfect_separation_still_gives_the_finite_ratio_it_always_did(
        monkeypatch):
    """The other direction: a real runner-up still produces the real number."""
    leaky = {
        "a": np.concatenate([np.arange(0, 7), np.arange(100, 103)]),
        "b": np.concatenate([np.arange(100, 107), np.arange(200, 203)]),
        "c": np.concatenate([np.arange(200, 207), np.arange(0, 3)]),
    }
    h = _health(DISJOINT, leaky, monkeypatch)

    assert h.margin.defined
    assert float(h.margin) == pytest.approx(7 / 3)   # unchanged by the fix
    assert h.unbounded_margins == 0


def test_a_single_stored_item_leaves_the_margin_undefined_and_untrustworthy(
        monkeypatch):
    """One item has no runner-up at all -- genuinely undefined, not infinite."""
    solo = {"a": np.arange(0, 10)}
    h = _health(solo, solo, monkeypatch)

    assert not h.margin.defined
    assert "runner-up" in h.margin.why
    assert not h.trustworthy, (
        "an unmeasurable margin is a fact about the MEASUREMENT, so it must "
        "fail the verdict that `.trustworthy` keys on rather than pass it")


def test_an_undefined_margin_cannot_be_used_as_a_number(monkeypatch):
    """The whole point of the type: forgetting to check raises."""
    solo = {"a": np.arange(0, 10)}
    h = _health(solo, solo, monkeypatch)

    with pytest.raises(UndefinedMeasurement):
        float(h.margin)
    with pytest.raises(UndefinedMeasurement):
        _ = h.margin < 1.2


# --------------------------------------------------------------------------
# the dead-probe guard must still fire -- this is the regression that matters
# --------------------------------------------------------------------------

def test_constant_reads_are_flagged_even_when_the_margin_looks_excellent(
        monkeypatch):
    """The case the margin-based check could never see.

    Every cue returns the same assembly, but the STORED assemblies are
    distinct -- so the runner-up overlap is zero and the margin is UNBOUNDED,
    the best-looking value in the range. Keyed on the margin (as it was, and as
    NaN made it before that), this dead probe passes. Keyed on the reads
    themselves, it cannot.
    """
    same = np.arange(0, 10)
    h = _health(DISJOINT, {key: same for key in DISJOINT}, monkeypatch)

    assert math.isinf(float(h.margin)), "the margin alone looks perfect here"
    assert not h.trustworthy, "the dead-probe verdict must still fire"
    assert any(v.label == "live probe" and not v.ok and "SAME assembly" in v.detail
               for v in h.verdicts)


def test_the_classic_dead_probe_is_still_flagged(monkeypatch):
    """The other direction: stored ALSO degenerate, margin exactly 1.00."""
    same = np.arange(0, 10)
    degenerate = {key: same for key in ("a", "b", "c")}
    h = _health(degenerate, {key: same for key in degenerate}, monkeypatch)

    assert not h.trustworthy
    assert any(v.label == "live probe" and not v.ok for v in h.verdicts)


def test_a_healthy_probe_is_not_flagged_as_dead(monkeypatch):
    """And the guard stays quiet when reads genuinely differ."""
    h = _health(DISJOINT, DISJOINT, monkeypatch)

    assert h.trustworthy
    assert all(v.ok for v in h.verdicts if v.label == "live probe")


# --------------------------------------------------------------------------
# distinctness, with no cues at all
# --------------------------------------------------------------------------

def test_spread_is_undefined_with_fewer_than_two_assemblies():
    """One assembly has no pair. A spread of 0.0 would be the HEALTHIEST read."""
    assert not diagnostics._spread([np.arange(0, 10)]).defined
    assert not diagnostics._spread([]).defined
    two = diagnostics._spread([np.arange(0, 10), np.arange(100, 110)])
    assert two.defined and float(two) == pytest.approx(0.0)


def test_an_empty_area_reports_no_distinctness_verdict_rather_than_a_failing_one():
    """Nothing stored is not the same as stored-and-indistinguishable."""
    h = diagnostics.area_health(_Brain(), "A", {})

    assert not h.distinct_frac.defined
    assert not h.spread.defined
    assert not any(v.label in ("distinct", "no duplicates") for v in h.verdicts), (
        "an empty area must not be judged on distinctness it never had")


def test_distinctness_is_still_judged_when_it_can_be_measured():
    """The other direction, without cues: real assemblies still get verdicts."""
    h = diagnostics.area_health(_Brain(), "A", DISJOINT)

    assert h.spread.defined and float(h.spread) == pytest.approx(0.0)
    assert h.distinct_frac.defined and float(h.distinct_frac) == pytest.approx(1.0)
    assert not h.collapsed
    labels = {v.label for v in h.verdicts}
    assert {"distinct", "no duplicates"} <= labels


# --------------------------------------------------------------------------
# rendering must never raise, and must never print a number it does not have
# --------------------------------------------------------------------------

def test_format_report_renders_an_undefined_quantity_as_n_a(monkeypatch):
    solo = {"a": np.arange(0, 10)}
    text = diagnostics.format_report([_health(solo, solo, monkeypatch)])

    assert "n/a" in text
    assert "nan" not in text.lower()


def test_formatting_an_undefined_measured_raises_rather_than_printing_nan():
    """`f"{m:.4f}"` must not quietly render an undefined value."""
    assert f"{Measured.of(0.25):.4f}" == "0.2500"
    with pytest.raises(UndefinedMeasurement):
        f"{Measured.undefined('no fiber'):.4f}"
