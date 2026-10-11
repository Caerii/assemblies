"""Load matching: an A/B on the sampler is not an A/B when the arms carry different
loads.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Dict, List, Optional)

import numpy as np



# --------------------------------------------------------------------------
# Load matching -- when an A/B on the sampler is not an A/B
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadGap:
    """Per-area load across the arms of one comparison."""

    area: str
    by_arm: Dict[str, float]      # arm -> w / n, the fraction ever fired
    engine: str
    threshold: float = 0.05       # threshold captured by load_audit

    @property
    def gap(self) -> float:
        vals = list(self.by_arm.values())
        return max(vals) - min(vals) if vals else 0.0

    @property
    def sampler_bearing(self) -> bool:
        """Only engines that INVENT a drive can have a load-dependent error."""
        return self.engine not in ("numpy_exact", "numpy_explicit")

    @property
    def cross_engine(self) -> bool:
        """True when the arms ran on DIFFERENT engines.

        Then this is not an A/B at all and `confounded` is meaningless: a gap
        measures how much the two engines' recruitment diverges, which is a
        useful number and a different claim. Kept separate because calling an
        engine comparison "confounded" is precisely the two-meanings-one-name
        error this module exists to prevent ([[same-name-two-meanings]]).
        """
        return self.engine == "mixed"

    def confounded(self, threshold: Optional[float] = None) -> bool:
        """Did two arms fail to share the sampler's error?

        ``load_audit`` captures its decision threshold on each result, so a
        later call without an override evaluates the same protocol that
        produced the result. Pass ``threshold`` to ask a different question.
        """
        limit = self.threshold if threshold is None else threshold
        return (self.sampler_bearing and not self.cross_engine
                and self.gap > limit)

    def __str__(self) -> str:
        arms = "  ".join(f"{a}={v:.3f}" for a, v in sorted(self.by_arm.items()))
        if self.cross_engine:
            tag = "engines differ"
        elif not self.sampler_bearing:
            tag = "n/a (exact)   "
        else:
            tag = "CONFOUNDED    " if self.confounded() else "ok            "
        return f"[{tag}] {self.area:<12} load {arms}   gap {self.gap:.3f}"


def area_load(brain) -> Dict[str, float]:
    """Per-area load `w / n` -- the fraction of neurons that have EVER fired.

    Read through `get_num_ever_fired`, not `area.w`. Those are two different
    quantities that shared a name, and reading the wrong one produced a 1.0000
    that the model does not produce ([[same-name-two-meanings]]).
    """
    out = {}
    for name, area in brain.areas.items():
        eng = brain.engine_for(name)
        try:
            w = int(eng.get_num_ever_fired(name))
        except (KeyError, AttributeError):
            continue
        out[name] = w / float(area.n) if area.n else 0.0
    return out


def load_audit(brains: Dict[str, Any], threshold: float = 0.05
               ) -> List[LoadGap]:
    """Do the arms of this comparison sit at the SAME area load?

    WHY THIS EXISTS, and it is a correction to a rule this module used to
    state. `arbitrate`'s docstring said absolute overlaps were suspect on the
    sampler but PAIRED comparisons survived, because both arms share the engine
    and therefore share its error. They share the engine. They share the error
    only if they share the LOAD -- and the sampler's error is a steep function
    of load (disjoint-input overlap 0.90 at low load, 0.19 once the area fills;
    `research/notes/substrate/graded_similarity_and_sampler_load.md`).

    Measured counterexample: norm_init on vs off under recurrence reads an 8.0x
    capacity gain on `numpy_sparse` and 1.0x on `numpy_exact`
    (`research/notes/memory/recurrence_ceiling_on_exact_drive.md`). norm_init changes
    which neurons win, so it changes how fast the area recruits, so the two
    arms carried DIFFERENT sampler errors and the difference between them was
    partly the difference between two errors.

    So the question "is this A/B safe?" is not answered by reasoning about the
    manipulation -- it is MEASURED here. Any manipulation that leaves the arms
    at the same load is fine whatever it does; any that separates them is
    suspect however innocuous it looks.

    A flag is NOT a verdict that the result is wrong. It says the two arms did
    not share the instrument's error, so the comparison should be re-run on
    `numpy_exact` (or `arms=("exact", "sampled")` via `arbitrate`) before the
    magnitude is quoted. On exact/explicit engines there is no sampler, so
    every gap is reported as `n/a`.

    !!! AND AN UNFLAGGED PAIR IS **NOT CLEARED**. THIS RANKS; IT DOES NOT FILTER.
    Measured 2026-08-02 (`task90_load_screen_sensitivity.py`), the specificity
    of this check is ZERO in the one case that could be constructed. Two arms
    differing ONLY in readout -- training bit-identical, load gap 0.000, so the
    screen passes -- gave:

        numpy_sparse   acc 1.0000 vs 1.0000    delta +0.0000
        numpy_exact    acc 0.7292 vs 0.9948    delta -0.2656

    The sampler reported NO effect where the substrate has a large one. Not a
    magnitude error: a missed effect entirely. Matching load matches the
    RECRUITMENT channel of the sampler's error and evidently there is another,
    which at high occupancy (load 0.999 there) reports perfect retrieval on an
    area that has already degraded.

    So: use this to decide WHAT TO RE-RUN FIRST. Nothing short of actually
    re-running on exact drive clears an A/B.

    Args:
        brains: ``{arm_name: Brain}``, each AFTER its protocol has run.
        threshold: load difference above which an arm pair is flagged.

    Returns:
        One `LoadGap` per area present in every arm, worst gap first.
    """
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    if len(brains) < 2:
        raise ValueError("load_audit compares arms; give it at least two")
    loads = {arm: area_load(b) for arm, b in brains.items()}
    shared = set.intersection(*(set(d) for d in loads.values()))
    engines = {getattr(b, "engine_name", "?") for b in brains.values()}
    engine = engines.pop() if len(engines) == 1 else "mixed"
    gaps = [LoadGap(area=a, engine=engine, threshold=float(threshold),
                    by_arm={arm: loads[arm][a] for arm in brains})
            for a in sorted(shared)]
    return sorted(gaps, key=lambda g: -g.gap)
