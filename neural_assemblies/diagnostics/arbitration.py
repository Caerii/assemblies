"""Arbitration: run a protocol on an engine that does not sample, and compare.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Dict, Sequence)

import numpy as np



# --------------------------------------------------------------------------
# Arbitration: ask an engine that does not sample
# --------------------------------------------------------------------------

#: Arms `arbitrate` can run, in order of how much they approximate.
#:
#:   explicit      full n x n weights, drive computed exactly, NO candidate
#:                 sampling anywhere. Ground truth, bounded by O(n^2) memory.
#:   materialized  the sparse engine with `materialize_area` called on every
#:                 area, so `w == n`, the sampler offers zero candidates
#:                 (`k_eff = min(k, max(0, n-w-1)) == 0`) and k-WTA sees exact
#:                 drive. Same answers as `explicit`, at O(n^2 * p) memory.
#:   exact         `numpy_exact`: the drive is recomputed from a content-addressed
#:                 hash instead of stored, so it is exact with NO n^2 term. This
#:                 is the arm that lets a protocol be arbitrated at the n the
#:                 science actually uses, rather than at a shrunken n.
#:   sampled       the normal sparse engine. The only arm that invents a drive
#:                 for neurons that have not fired.
ARBITER_ARMS = ("explicit", "materialized", "exact", "sampled")


@dataclass(frozen=True)
class Arm:
    """How to build a `Brain` for one arbiter arm.

    WHY THIS IS AN OBJECT AND NOT A BOOL. The original signature was
    ``build(explicit: bool)``, which encoded "which arm" as "is it the explicit
    one" -- fine with two arms, wrong with four, and it forced every caller to
    grow an `if` when a new engine arrived. An arm now says what it needs and
    the caller splats it:

        def build(arm):
            b = Brain(p=P, seed=0, **arm.brain_kwargs)
            b.add_area("A", n, k, beta, **arm.area_kwargs)
            ...
            return b

    Written that way, a caller supports every present and future arm without
    naming any of them.
    """
    name: str
    brain_kwargs: Dict[str, object]   # splat into `Brain(...)`
    area_kwargs: Dict[str, object]    # splat into `brain.add_area(...)`

    @property
    def explicit(self) -> bool:
        """True for the `explicit` arm, for callers that must still branch."""
        return self.name == "explicit"


#: The concrete build recipe per arm. `materialized` builds like `sampled` and
#: is then materialised by `arbitrate` AFTER the brain exists, which is why its
#: kwargs are identical to `sampled`'s and not a third engine.
_ARM_SPECS: Dict[str, Arm] = {
    "explicit":     Arm("explicit", {}, {"explicit": True}),
    "materialized": Arm("materialized", {}, {}),
    "exact":        Arm("exact", {"engine": "numpy_exact"}, {}),
    "sampled":      Arm("sampled", {}, {}),
}


def arm_spec(name: str) -> Arm:
    """The `Arm` for *name*; raises on an unknown arm."""
    try:
        return _ARM_SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown arm {name!r}; expected one of {ARBITER_ARMS}") from None


@dataclass
class Arbitration:
    """One protocol, measured on each arm with ONE extractor."""
    label: str
    # Arms may be scalar metrics or per-item sequences; the extractor owns
    # that protocol choice, so keep the container honest rather than forcing
    # an incorrect homogeneous numeric type here.
    by_arm: Dict[str, Any]

    def ratio(self, arm: str = "sampled", truth: str = "explicit"):
        """`arm / truth`, elementwise for sequences, else scalar."""
        a, t = self.by_arm.get(arm), self.by_arm.get(truth)
        if a is None or t is None:
            return None
        if np.isscalar(a) and np.isscalar(t):
            return float(a) / float(t) if t else float("nan")
        if np.isscalar(a) or np.isscalar(t) or len(a) != len(t):
            raise ValueError("arbitration ratios require equally sized sequences")
        return [float(x) / float(y) if y else float("nan")
                for x, y in zip(a, t, strict=True)]

    def __str__(self) -> str:
        def fmt(v):
            if v is None:
                return "  (not run)"
            try:
                return " ".join(f"{float(x):>8.3f}" for x in v)
            except TypeError:
                return f"{float(v):>8.3f}"
        w = max(len(a) for a in self.by_arm) if self.by_arm else 0
        return "\n".join([f"  {self.label}"] +
                         [f"    {a:<{w}}  {fmt(self.by_arm[a])}"
                          for a in ARBITER_ARMS if a in self.by_arm])


def arbitrate(build, measure, arms: Sequence[str] = ARBITER_ARMS,
              label: str = "protocol") -> Arbitration:
    """Run one protocol on several engines and read it with ONE extractor.

    WHY THIS EXISTS.  The sparse engine invents a drive for neurons that have
    not fired (`sample_new_winner_inputs`); the explicit engine does not. So
    whenever a sparse-engine number looks wrong, the question "is this the
    substrate or the sampler?" is answerable -- but only by running the same
    protocol on an engine that does not share the approximation.

    Measured 2026-07-31 on the merge protocol, support in units of k:

        n, k        explicit    materialized    sampled
        1000, 32       6.1          5.9           8.9
        2000, 45       6.1          7.2           8.2
        4000, 63       6.9          7.2          10.8

    `materialized` tracks `explicit`; `sampled` runs systematically high. That
    is the sampler's accuracy cost, and it is the first thing to rule out.

    THE SHARED EXTRACTOR IS THE POINT, not a convenience. Hand-built arbiter
    harnesses read `area.w` on the sparse side and distinct-winners-over-the-run
    on the explicit side -- two different quantities with the same informal
    name -- and reported the sampler gap as 2-4x when it is ~1.4x. `measure`
    runs unchanged against every arm precisely so that cannot happen. See
    [[same-name-two-meanings]].

    Args:
        build: ``build(arm: Arm) -> Brain``. Construct the brain and run the
            protocol, splatting `arm.brain_kwargs` into `Brain(...)` and
            `arm.area_kwargs` into every `add_area(...)`. Must be
            side-effect-free across calls (reseed inside).
        measure: ``measure(brain) -> value``. Scalar or sequence. Applied
            IDENTICALLY to every arm.
        arms: subset of `ARBITER_ARMS`.
        label: shown in `str(...)`.

    Returns:
        `Arbitration`; `.by_arm[arm]` and `.ratio()`.

    ON CHOOSING n. `materialized` needs `n^2 * p` floats per area fiber and
    `explicit` needs `n^2`, so those two arms force `n` down -- and when you
    shrink `n`, preserve `k*p` (the expected afferent count), NOT `p`; see the
    module docstring of `research/literature/parity/pnas2020_paper_claims.py`
    for why copying `p` to a smaller `n` destroys the dynamics. The `exact`
    arm has no such term, so a protocol that only needs truth-vs-sampler can
    run ``arms=("exact", "sampled")`` at full size. Include `explicit` at a
    small `n` as well when you want to check that `exact` and the ground truth
    still agree.
    """
    out: Dict[str, object] = {}
    for arm in arms:
        spec = arm_spec(arm)
        brain = build(spec)
        if arm == "materialized":
            for name, area in list(brain.areas.items()):
                eng = brain.engine_for(name)
                if hasattr(eng, "materialize_area"):
                    eng.materialize_area(name)
        out[arm] = measure(brain)
    return Arbitration(label=label, by_arm=out)


def arbitrate_prebuilt(run, arms: Sequence[str] = ARBITER_ARMS,
                       label: str = "protocol") -> Arbitration:
    """`arbitrate` for protocols that must materialize BEFORE they run.

    `arbitrate` materializes after `build` returns, which is correct when the
    protocol is what `build` executed. When the protocol must see a fully
    materialized area from its first projection, pass ``run(arm: Arm) -> value``
    and do the materialization inside it.
    """
    return Arbitration(label=label,
                       by_arm={a: run(arm_spec(a)) for a in arms})
