"""Separation: score the ORDERING of matched and mismatched pairs, not the scale.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass




# --------------------------------------------------------------------------
# Separation -- score the ORDERING, not the scale
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Separation:
    """How well two conditions separate, measured by ORDERING not by scale."""

    label: str
    auc: float                    #: P(high > low), ties at 0.5. Null is 0.5.
    n_high: int
    n_low: int
    #: Range the raw values occupy. A metric using a sliver of its range is
    #: saturated however large its Cohen's d, and that is worth seeing.
    span: float = float("nan")

    @property
    def perfect(self) -> bool:
        return self.auc >= 1.0

    @property
    def saturated(self) -> bool:
        return self.span == self.span and self.span < 0.05

    def __str__(self) -> str:
        flag = "  SATURATED" if self.saturated else ""
        return (f"{self.label}: AUC {self.auc:.3f} "
                f"(n={self.n_high}/{self.n_low}), span {self.span:.4f}{flag}")


def separation(high, low, label: str = "separation",
               bounded: bool = True) -> Separation:
    """Rank separation of two conditions. PREFER THIS TO COHEN'S D.

    WHY. Cohen's d divides by a pooled standard deviation, so it measures the
    REPRESENTATION as much as the effect. Measured on one ERP contrast, four
    encodings of an IDENTICAL ordering::

        grows,     raw p600        AUC 1.000    Cohen's d  2.241
        read_only, raw p600        AUC 1.000    Cohen's d  4.691
        grows,     clipped excess  AUC 1.000    Cohen's d  2.826
        read_only, clipped excess  AUC 1.000    Cohen's d 24.754

    An 11x range on the same separation. The 24.754 comes from
    ``max(0, v - grammatical_median)``, which clips the NULL arm against its own
    median so it lands on an exact 0.0 floor with almost no variance -- after
    which any reduction in measurement noise inflates d without the effect
    growing at all.

    AUC is invariant under every monotone transform, so it is unchanged by
    clipping, by rescaling, and -- the case that actually happened here -- by
    REDEFINING THE UNDERLYING QUANTITY. This package replaced P600 (unbounded
    post-k-WTA churn, grammatical 0.12 vs violation 5.24) with an energy deficit
    bounded in [0,1] (0.989 vs 0.995). Every absolute threshold silently became
    meaningless; a rank statistic would have survived untouched.

    `span` is reported alongside BECAUSE AUC deliberately ignores magnitude: a
    perfect ordering across 0.7% of the range is perfect ordering AND a
    saturated metric, and both facts matter. Pass ``bounded=False`` when the
    quantity has no natural box.

    Args:
        high: values from the condition expected to score HIGHER.
        low: the null / baseline condition.
        bounded: whether the raw quantity has a natural range worth reporting.
    """
    hi = [float(v) for v in high]
    lo = [float(v) for v in low]
    if not hi or not lo:
        raise ValueError(
            f"separation({label!r}) needs values in BOTH conditions, got "
            f"{len(hi)} high and {len(lo)} low. An empty arm is a protocol "
            f"failure, not a separation of zero.")
    wins = sum(1.0 if a > b else (0.5 if a == b else 0.0)
               for a in hi for b in lo)
    everything = hi + lo
    span = (max(everything) - min(everything)) if bounded else float("nan")
    return Separation(label=label, auc=wins / (len(hi) * len(lo)),
                      n_high=len(hi), n_low=len(lo), span=span)
