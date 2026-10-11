"""Ensembles over seeds: mean and confidence interval, paired deltas, arm comparison.
The safe path is the short one.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass
from typing import (Any, Dict, Optional, Sequence,
                    Tuple)




# --------------------------------------------------------------------------
# Ensembles and arm comparison
#
# These exist because of four measurement errors made in one session, three
# of which had the SAME shape: a number looked like a result and was actually
# a mechanism that never ran.
#
#   * a candidate arm and its control returned IDENTICAL values because the
#     fiber under test was never materialised (zero drive, k winners anyway);
#   * a "regression" of 1.68 sd was acted on as real when the seed-to-seed
#     spread covered it;
#   * a test asserted "above chance" from ONE seed for a quantity whose
#     ensemble mean was AT chance;
#   * a readout scored the unigram baseline with nothing learned, because
#     `sorted()` broke ties in an order that correlated with the answer.
#
# The point of these helpers is that the safe path is the short one.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ensemble:
    """A measurement over seeds. Never a point estimate.

    The assembly calculus is a claim about ENSEMBLES -- `G(n,p)` is one draw
    and nothing scientific may depend on which draw you got. A single-seed
    before/after is not a measurement, so this is what a result looks like.
    """

    label: str
    values: Tuple[float, ...]
    mean: float
    ci: float          # half-width of the 95% interval
    keys: Optional[Tuple[Any, ...]] = None  # seed/cell identity, in values order

    @property
    def low(self) -> float:
        return self.mean - self.ci

    @property
    def high(self) -> float:
        return self.mean + self.ci

    def beats(self, threshold: float) -> bool:
        """Strictly above `threshold` -- judged by the CONFIDENCE BOUND.

        Not `mean > threshold`. A point estimate that happens to clear a bar
        is exactly what produced the bogus "next-token beats chance" claim.
        """
        return self.low > threshold

    def indistinguishable_from(self, threshold: float) -> bool:
        return self.low <= threshold <= self.high

    def __str__(self) -> str:
        return (f"{self.label}: {self.mean:.4f} +/- {self.ci:.4f} "
                f"(n={len(self.values)}, "
                f"{min(self.values):.4f}..{max(self.values):.4f})")


def ensemble(run, seeds: Sequence[int], label: str = "arm") -> Ensemble:
    """Run `run(seed) -> float` over `seeds` and summarise as mean +/- 95% CI.

    Args:
        run: callable taking a seed and returning a scalar.
        seeds: at least 3; fewer cannot support an interval.
        label: shown in `str()`.
    """
    seeds = list(seeds)
    if len(seeds) < 3:
        raise ValueError(
            f"{len(seeds)} seeds cannot support a confidence interval. "
            f"Seed-to-seed sd is routinely as large as the effects measured "
            f"here, so 1-2 seeds is a draw, not a measurement.")
    _validate_ensemble_keys(seeds, len(seeds))
    return ensemble_from_values([float(run(s)) for s in seeds], label,
                                keys=seeds)


def _validate_ensemble_keys(keys: Sequence[Any], count: int) -> None:
    if len(keys) != count:
        raise ValueError("ensemble keys must match the number of values")
    try:
        unique = set(keys)
    except TypeError as exc:
        raise ValueError("ensemble keys must be hashable seed/cell identities") from exc
    if len(unique) != len(keys):
        raise ValueError("ensemble keys must be unique; duplicate seeds are not independent replicates")


def _t_interval(vals: Sequence[float]) -> float:
    """Half-width of the two-sided 95% Student-t interval of the mean.

    The critical value comes from the t distribution at n - 1 degrees of
    freedom for EVERY n. Until 2026-09-09 a table stopped at n = 12 and
    fell back to the normal 1.96 above it, which made every twenty-seed
    interval 6.4% too narrow (2.093 is the right value at n = 20); an
    external review caught it. No adopted verdict flips under the
    correction (each registered bar was cleared by more than that margin),
    but the intervals printed before that date are narrower than stated.
    """
    from scipy.stats import t as _t
    n = len(vals)
    return float(_t.ppf(0.975, n - 1)) * statistics.stdev(vals) / n ** 0.5


def ensemble_from_values(values: Sequence[float], label: str = "arm",
                         keys: Optional[Sequence[Any]] = None) -> Ensemble:
    """Summarise ALREADY-COMPUTED per-seed values as mean +/- 95% CI.

    THIS EXISTS FOR PARALLEL RUNNERS. `ensemble` takes a callable and drives
    the seeds itself, which a process pool cannot do -- the cells are computed
    elsewhere and come back as a list. Without this the caller reaches for
    `statistics.mean` and a hand-rolled interval, which is the exact pattern
    `test_methodology_ratchet` exists to stop, so the sanctioned path has to
    cover the parallel case too or the ratchet just pushes work off a cliff.

    `keys` records unique seed/cell identities for pairing and error messages.
    Without keys, values can only be paired positionally with another unkeyed ensemble.
    """
    vals = [float(v) for v in values]
    if len(vals) < 3:
        raise ValueError(
            f"{len(vals)} seeds cannot support a confidence interval. "
            f"Seed-to-seed sd is routinely as large as the effects measured "
            f"here, so 1-2 seeds is a draw, not a measurement.")
    identities = tuple(keys) if keys is not None else None
    keys = list(identities) if identities is not None else list(range(len(vals)))
    _validate_ensemble_keys(keys, len(vals))
    bad = [s for s, v in zip(keys, vals, strict=True) if math.isnan(v)]
    if bad:
        # Refuse LOUDLY rather than let statistics.stdev die with a cryptic
        # AttributeError deep in the fraction machinery (it cost two
        # analysis iterations in one night). NaN values are usually an
        # undefined per-seed statistic (e.g. a correlation over a
        # constant-outcome seed) -- and dropping them silently is exactly
        # the bias the undefinedness-correlates-with-outcome lesson warns
        # about, so the caller must decide what a NaN seed MEANS.
        raise ValueError(
            f"ensemble '{label}': NaN from seeds {bad}. A NaN usually "
            f"means the per-seed statistic is UNDEFINED there (constant "
            f"outcomes, empty selection). Handle those seeds explicitly "
            f"-- do not silently filter them.")
    infinite = [s for s, v in zip(keys, vals, strict=True) if not math.isfinite(v)]
    if infinite:
        raise ValueError(f"ensemble '{label}': non-finite values from seeds {infinite}")
    mean = statistics.mean(vals)
    ci = _t_interval(vals)
    return Ensemble(label, tuple(vals), mean, ci, identities)


def compare_arms(arms: Dict[str, Any], seeds: Sequence[int],
                 strict: bool = True) -> Dict[str, Ensemble]:
    """Run several arms on the SAME seeds and refuse to return silent no-ops.

    THE GUARD IS THE POINT. If two arms produce bit-identical values on every
    seed they did not run different computations, however different their
    configuration looked. That is not a finding of "no effect" -- it is a dead
    pathway, a flag that never reached the engine, or two names for one code
    path, and it is the single most common way this codebase produces a
    confident wrong answer.

    It happened here: closing a fiber during training left it unmaterialised,
    so re-opening it at readout projected through a connectome that was never
    grown. The "intervention" arm and its control agreed to four decimals
    across ten seeds, which reads as a clean negative result and was in fact
    the control measured twice.

    Args:
        arms: ``{name: run}``, each ``run(seed) -> float``.
        seeds: shared across arms so differences are paired.
        strict: raise on identical arms. Set False only when duplication is
            genuinely expected, and say why at the call site.

    Raises:
        ValueError: if two arms are identical on every seed and `strict`.
    """
    out = {name: ensemble(run, seeds, name) for name, run in arms.items()}
    names = list(out)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if out[a].values == out[b].values:
                msg = (f"arms {a!r} and {b!r} returned IDENTICAL values on all "
                       f"{len(seeds)} seeds -- they are not two arms. Check "
                       f"that the intervention reached the engine and that "
                       f"the fiber it targets was ever materialised "
                       f"(see fiber_census); a never-grown connectome carries "
                       f"zero drive and still returns k winners.")
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return out


def paired_delta(a: Ensemble, b: Ensemble, label: str = "delta") -> Ensemble:
    """Per-seed difference `a - b`, which is what an A/B actually asks.

    Comparing two independent CIs is not the same test and is less powerful;
    and comparing a difference against a SINGLE arm's sd understates the
    spread by ~sqrt(2), which is how a 1.49-sd difference got reported as
    2.10 sd here.
    """
    if len(a.values) != len(b.values):
        raise ValueError("paired_delta needs the same seeds in both arms")
    if a.keys != b.keys:
        raise ValueError("paired_delta needs the same seed keys in the same order in both arms")
    diffs = [x - y for x, y in zip(a.values, b.values, strict=True)]
    return ensemble_from_values(diffs, label, keys=a.keys)
