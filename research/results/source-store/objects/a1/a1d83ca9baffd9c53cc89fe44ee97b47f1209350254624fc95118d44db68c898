"""A metric declares its range. A constant declares its metric.

TIER D of research/plans/TYPE_SAFETY_PROGRAM.md.

THE DEFECT THIS PREVENTS, which is on record and still live. `gates.py` defines

    P600_EXCESS_MARGIN = 0.152

which was correct: it was calibrated against an UNBOUNDED cumulative P600 whose
grammatical null was ~0.12 and whose violation case was ~5.24, so a margin of
0.152 sat sensibly between them.

`adapters.py` then replaced the quantity, for a good reason -- post-k-WTA churn
reverses sign under ``norm_init``. P600 became ``1 - normalized_energy``, bounded
in [0,1], living at 0.989 vs 0.995.

**The quantity was replaced. The constant was not.** Measured consequence: the
margin resolves to 0.076 against a maximum observed excess of 0.0064, so the
violation detector is 11.9x from ever firing and has never fired on any seed --
while the calibration reports ``source="empirical"``.

Nothing in the codebase connected the constant to the quantity, so nothing could
notice. That is the gap this module closes: a `Metric` carries its expected
range, a `Threshold` is declared against a metric, and observations are checked
against the declaration at the point they are produced.

WHY RUNTIME AND NOT TYPES. This venv has no pip and no type checker, so a static
annotation would enforce nothing. More importantly a static type cannot express
"0.152 is in range for THIS quantity" -- that is a value property, and it has to
be checked against real data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence


class MetricRangeError(ValueError):
    """Observations fall outside the range their metric declares."""


@dataclass(frozen=True)
class Metric:
    """A named quantity, its expected range, and what a change means.

    ``version`` is the part that does the work. Bump it when the QUANTITY
    changes -- not when its implementation is optimised, but when what the
    number MEANS changes. Every `Threshold` pinned to an older version then
    fails loudly instead of silently describing a quantity that no longer
    exists.
    """

    name: str
    lo: float
    hi: float
    version: int = 1
    #: What the number means, in one line. Shown in every failure message,
    #: because "p600 out of range" is not actionable and "p600 is
    #: 1 - normalized_energy, bounded [0,1]" is.
    definition: str = ""

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= float(value) <= self.hi

    def check(self, values: Sequence[float], *, where: str = "") -> None:
        """Raise if any observation falls outside the declared range.

        Called where values are PRODUCED, so a metric that has quietly changed
        scale is caught at its source rather than three modules downstream in a
        statistic that still returns a number.
        """
        bad = [float(v) for v in values if not self.contains(v)]
        if bad:
            raise MetricRangeError(
                f"{self.name} declares range [{self.lo}, {self.hi}] "
                f"({self.definition or 'no definition given'}) but "
                f"{len(bad)} of {len(values)} observations fall outside it: "
                f"{bad[:5]}{'...' if len(bad) > 5 else ''}"
                f"{f' at {where}' if where else ''}. Either the observation is "
                f"wrong or the metric was redefined -- if redefined, bump "
                f"Metric.version and revisit every Threshold pinned to it.")

    def occupancy(self, values: Sequence[float]) -> float:
        """Fraction of the declared range the observations actually span.

        A metric using a sliver of its range is saturated, and that is invisible
        in any statistic that standardises by the observed spread -- Cohen's d
        on a variable occupying 0.7% of its range read 24.754 on a difference of
        0.004. See `diagnostics.separation`.
        """
        vals = [float(v) for v in values]
        if not vals or self.width <= 0:
            return float("nan")
        return (max(vals) - min(vals)) / self.width


@dataclass(frozen=True)
class Threshold:
    """A constant that only means something relative to a metric.

    `for_metric` and `metric_version` are what make a stale constant detectable.
    A threshold sitting far outside the range of everything the metric can
    produce is not a conservative threshold, it is a dead one.
    """

    name: str
    value: float
    for_metric: Metric
    metric_version: int = 1
    #: Multiple of the observed spread beyond which the threshold is
    #: unreachable in practice rather than merely strict.
    unreachable_factor: float = 3.0

    def __post_init__(self) -> None:
        if self.metric_version != self.for_metric.version:
            raise MetricRangeError(
                f"threshold {self.name!r} = {self.value} was calibrated against "
                f"{self.for_metric.name} v{self.metric_version}, but that "
                f"metric is now v{self.for_metric.version} "
                f"({self.for_metric.definition}). The quantity changed and the "
                f"constant did not. Re-derive it, then update metric_version.")

    def audit(self, observations: Sequence[float], *,
              where: str = "") -> Optional[str]:
        """Is this threshold reachable by the values actually seen?

        Returns a description of the problem, or None. Deliberately NOT an
        exception: a threshold can be legitimately unreachable early in training
        when the mechanism it gates is not ready yet. It becomes a defect only
        when it stays unreachable, which is what the caller decides.
        """
        vals = [abs(float(v)) for v in observations]
        if not vals:
            return None
        peak = max(vals)
        if peak <= 0:
            return (f"{self.name} = {self.value} but every observation of "
                    f"{self.for_metric.name} is zero -- nothing can reach it")
        ratio = abs(self.value) / peak
        if ratio > self.unreachable_factor:
            return (f"{self.name} = {self.value} is {ratio:.1f}x the LARGEST "
                    f"observed {self.for_metric.name} ({peak:.4g})"
                    f"{f' at {where}' if where else ''}, so the mechanism it "
                    f"gates cannot fire. This is the signature of a constant "
                    f"calibrated against a different scale -- check whether "
                    f"the metric was redefined under it.")
        return None


def warn_if_unreachable(threshold: Threshold, observations: Sequence[float],
                        *, where: str = "") -> None:
    """`Threshold.audit` as a warning, for call sites that should not raise."""
    msg = threshold.audit(observations, where=where)
    if msg:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
