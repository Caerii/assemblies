"""Probe validation: a measurement that cannot vary is not a measurement.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass




# --------------------------------------------------------------------------
# Probe validation -- a measurement that cannot vary is not a measurement
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeCheck:
    """Whether a probe can tell apart two cases it MUST tell apart."""

    label: str
    high: float
    low: float

    @property
    def separation(self) -> float:
        return self.high - self.low

    @property
    def discriminating(self) -> bool:
        return self.separation > 0.0

    def __str__(self) -> str:
        return (f"{self.label}: high {self.high:.4f}, low {self.low:.4f}, "
                f"separation {self.separation:+.4f}")


def verify_probe(reads_high, reads_low, *, min_separation: float = 0.2,
                 label: str = "probe") -> ProbeCheck:
    """Refuse a probe until it has produced BOTH answers on demand.

    `reads_high` and `reads_low` are zero-argument callables returning a float,
    on cases chosen so the probe MUST read high on one and low on the other.
    Raises unless they separate by `min_separation`.

    WHY THIS EXISTS.  This substrate is totalizing: every operation succeeds and
    returns a well-formed value. k-WTA always returns exactly k winners; a fixed
    area always returns its winners; a tie always resolves by index; `read_only`
    always returns the frozen set. Nothing can report "I did nothing". So a
    broken probe does not raise -- it returns a plausible number, and the number
    is usually 0.000 or 1.000, which reads as a clean result.

    Three probes written in a single evening, all on the same object, all
    reading exactly 1.000 for three DIFFERENT reasons:

      1. driving SYNTAX from MOOD alone -- on untrained weights every candidate
         ties and the deterministic index tie-break returns identical winners;
      2. isolating with `read_only()` -- which freezes the winners, so every arm
         got the same stale set;
      3. a PHON -> LEX -> PHON round trip with PHON still pinned by `activate`
         -- a fixed target short-circuits `project_into`, so the trip never
         travelled.

    Two of those were caught by disbelieving a round number. The third was not
    caught at all: it shipped, and a published "conserved budget" had to be
    retracted. Vigilance is not the control; this is.

    Cheap, and it has never yet failed to pay: the first real use took a
    saturated stability probe (0.846 in both arms) and an untrained control
    (0.06-0.18) and made the saturation obvious in one line.

    The `min_separation` default is deliberately blunt. If a probe's true effect
    is smaller than 0.2, pass the value you can justify -- but pass it before
    seeing the data, for the reason `gain_stability.noise_floor` is mandatory.
    """
    hi, lo = float(reads_high()), float(reads_low())
    check = ProbeCheck(label, hi, lo)
    if check.separation < min_separation:
        raise AssertionError(
            f"{label} cannot discriminate: reads {hi:.4f} where it must read "
            f"HIGH and {lo:.4f} where it must read LOW (separation "
            f"{check.separation:+.4f} < {min_separation}). Fix the probe "
            f"before trusting any number it produces -- an instrument that "
            f"returns the same value either way is not measuring the thing it "
            f"is named after.")
    return check
