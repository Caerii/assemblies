"""Gain stability: when a sweep measures the boundary instead of the substrate.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Dict, Sequence)




# --------------------------------------------------------------------------
# Gain stability -- when a sweep measures the boundary instead of the substrate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GainStability:
    """One effect, re-measured at several absolute gains."""

    label: str
    by_gain: Dict[float, float]
    noise_floor: float          #: the ESTIMATOR's own reproducibility

    @property
    def spread(self) -> float:
        v = list(self.by_gain.values())
        return (max(v) - min(v)) if v else 0.0

    @property
    def confounded(self) -> bool:
        """The effect moves with gain by more than the estimator's scatter."""
        return self.spread > self.noise_floor

    @property
    def inconclusive(self) -> bool:
        """Too close to the noise floor for either verdict to mean anything."""
        return abs(self.spread - self.noise_floor) < 0.4 * self.noise_floor

    @property
    def sign_reverses(self) -> bool:
        v = list(self.by_gain.values())
        return bool(v) and min(v) < 0 < max(v)

    def __str__(self) -> str:
        vals = "  ".join(f"g={g:g}:{v:+.3f}" for g, v in
                         sorted(self.by_gain.items()))
        if self.inconclusive:
            tag = "INCONCLUSIVE"
        elif self.confounded:
            tag = "CONFOUNDED  "
        else:
            tag = "gain-stable "
        return (f"[{tag}] {self.label}: {vals}   spread {self.spread:.3f} "
                f"vs floor {self.noise_floor:.3f}")


def gain_stability(run, gains: Sequence[float], noise_floor: float,
                   label: str = "effect") -> GainStability:
    """Re-measure one effect at several absolute gains. Does it survive?

    WHY THIS EXISTS. `critical-load-alpha-star` states the rule:

        Whenever `g_c` depends on an axis, comparing along that axis at fixed
        ABSOLUTE gain is confounded.

    That rule has now been violated three times in this repo -- an n sweep, a
    k sweep, and (2026-08-02) a lexicon-capacity n sweep that read
    `M_max ~ n^1.70` on three resolved crossings with two doublings agreeing to
    7%. Every internal check passed and the result was still an artifact:
    re-measured at beta = 0.05 / 0.10 / 0.20 the exponent read
    1.55 / 1.65 / **0.87**, going SUB-linear at high gain
    (`research/notes/memory/ceiling_n_scaling_on_exact_drive.md`).

    Knowing `g_c` is not required to detect this, which is the point. If an
    effect is a property of the substrate it is the same at every gain; if it
    is a reading of how far a fixed gain sits from a moving boundary, it moves.
    So: run the whole comparison at several gains and look at the spread.

    `noise_floor` IS MANDATORY AND HAS NO DEFAULT. The first run of that check
    reported "not confounded" at spread 0.29 against a threshold of 0.30 that
    had been picked in advance -- a coin flip dressed as a verdict. The floor
    must be the MEASURED reproducibility of your own estimator under choices
    that should not matter (grid spacing, seed set, refinement depth). For the
    capacity ceiling it was 0.20, measured by estimating the same configuration
    on a factor-2 and a x1.25 grid and finding m_star 32% apart.

    Args:
        run: ``run(gain) -> float``. The EFFECT SIZE, not a raw reading -- a
            fitted exponent, a difference between arms, a ratio. Must be the
            same quantity at every gain.
        gains: absolute gains to sweep. Three spanning ~2x is usually enough.
        noise_floor: measured scatter of `run` under irrelevant choices.
        label: shown in `str()`.

    Returns:
        `GainStability`. Check `.inconclusive` BEFORE reading `.confounded`.
    """
    if len(gains) < 2:
        raise ValueError("gain_stability needs at least two gains")
    if not noise_floor > 0:
        raise ValueError(
            "noise_floor must be measured and positive. Estimate the SAME "
            "configuration under a choice that should not matter (a different "
            "grid, a different seed set) and use the spread. A threshold "
            "picked without it cannot distinguish an effect from your "
            "estimator's scatter -- which is how this check first returned a "
            "false negative at 0.29 vs 0.30.")
    return GainStability(label=label,
                         by_gain={float(g): float(run(g)) for g in gains},
                         noise_floor=float(noise_floor))
