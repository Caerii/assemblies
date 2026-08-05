"""A measurement that knows whether it is defined, and refuses to be used if not.

THE PROBLEM. This substrate is TOTALIZING: there is no bottom. k-WTA always
returns k winners, a projection through a fiber with zero synapses still yields
an assembly, and an unknown word still parses. So a function that has nothing to
report cannot signal that -- it must invent a float. The inventions are always
plausible, and always at an end of the range where they look like a finding:

  * ``_self_recurrent_energy`` returns 0.0 for "no energy" AND for "the fiber
    does not exist". The P600 violation arm read a CONSTANT 1.0 for months
    because VP has no self-fiber, and every magnitude published off it was
    measured with one arm dead.
  * ``phrase_stability`` returns 1.0 for "perfectly stable" AND for "there were
    no phrase areas to measure".
  * an out-of-vocabulary word yields ``p600 0.0000, stability 1.0000`` -- the
    degenerate no-parse, indistinguishable from data. A third of one ERP
    condition was silently that for months.

WHY NOT JUST ``.trustworthy``. That convention already exists on
``parse_errors.Stability`` and ``diagnostics.AreaHealth``, and it is the right
idea -- this is its generalization, not a competitor. But it is OPT-IN: the
caller has to remember to read it, and the failures above are precisely the
cases where nobody did. ``Measured`` closes that by making the undefined value
UNUSABLE: ``float()``, comparison, and arithmetic all raise. You cannot forget
to check, because forgetting is an exception rather than a number.

    energy = self_recurrent_energy(brain, "VP")
    if energy.defined:
        report(float(energy))          # explicit, checked
    else:
        skip(energy.why)               # names the reason

    float(Measured.undefined("VP has no self-fiber"))   # raises, loudly

NaN WOULD NOT DO. NaN propagates silently through means, comparisons return
False rather than raising, and `np.mean` of a NaN-containing array is NaN --
which then reads as "the metric is broken" rather than "this quantity was never
defined here". The undefined value still CARRIES NaN so that a bypass degrades
to NaN rather than to a plausible number, but the type is what does the work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class UndefinedMeasurement(RuntimeError):
    """Raised when an undefined `Measured` is used as a number.

    Carries the reason, because "undefined" alone sends the reader back to the
    substrate to work out which of the several no-op paths they hit.
    """


@dataclass(frozen=True)
class Measured:
    """A float that may be UNDEFINED, and says why.

    Attributes:
        value: the number, when ``defined``. NaN otherwise, so that any path
            which bypasses the type still degrades to NaN and not to 0.0 or 1.0
            -- the two values that read as findings.
        defined: whether ``value`` means anything.
        why: the reason it does not, phrased as the missing precondition
            ("VP has no self-fiber"), not as a restatement ("undefined").
        detail: optional structured context for diagnostics (pool sizes, area
            names) that a caller may want without parsing ``why``.
    """

    value: float
    defined: bool = True
    why: str = ""
    detail: Optional[dict] = None

    @classmethod
    def undefined(cls, why: str, **detail) -> "Measured":
        """The quantity does not exist here. ``why`` names the precondition."""
        return cls(float("nan"), False, why, detail or None)

    @classmethod
    def of(cls, value: float) -> "Measured":
        """A defined measurement. Explicit constructor so wrapping is visible."""
        return cls(float(value), True, "")

    def __float__(self) -> float:
        if not self.defined:
            raise UndefinedMeasurement(
                f"measurement is undefined: {self.why}. Check `.defined` "
                f"before use, or call `.or_else(fallback)` and say in the "
                f"caller what the fallback means."
            )
        return float(self.value)

    def or_else(self, fallback: float) -> float:
        """Value if defined, else *fallback* -- an EXPLICIT decision.

        This is the sanctioned escape. It differs from the old behaviour only
        in being visible at the call site: the reader can see that a default
        was chosen and ask whether it is the right one.
        """
        return float(self.value) if self.defined else float(fallback)

    # Comparisons and arithmetic route through __float__ ON PURPOSE, so that
    # `energy < threshold` on an undefined value raises instead of quietly
    # answering. That comparison is the exact shape of the P600 bug.
    def __lt__(self, other) -> bool: return float(self) < float(other)
    def __le__(self, other) -> bool: return float(self) <= float(other)
    def __gt__(self, other) -> bool: return float(self) > float(other)
    def __ge__(self, other) -> bool: return float(self) >= float(other)
    def __add__(self, other) -> float: return float(self) + float(other)
    def __radd__(self, other) -> float: return float(other) + float(self)
    def __sub__(self, other) -> float: return float(self) - float(other)
    def __rsub__(self, other) -> float: return float(other) - float(self)
    def __mul__(self, other) -> float: return float(self) * float(other)
    def __rmul__(self, other) -> float: return float(other) * float(self)

    def __str__(self) -> str:  # pragma: no cover - display only
        if not self.defined:
            return f"UNDEFINED ({self.why})"
        return f"{self.value:.6f}"


def defined_values(measurements) -> list:
    """The defined values only, for aggregating over a set of probes.

    Averaging a mixture of defined and undefined readings is the aggregate form
    of the same defect -- an undefined 0.0 pulls a mean toward a finding. Use
    this and REPORT how many were dropped; a mean over 2 of 9 probes is a
    different claim from a mean over 9.
    """
    return [float(m) for m in measurements if m.defined]
