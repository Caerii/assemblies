"""Is a configuration in the regime the laws were measured in?

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (Dict, List, Mapping, Optional, Sequence)


from neural_assemblies.core.measurement import Measured


@dataclass
class Regime:
    """Whether one area receives enough afferents to select winners reliably.

    The sequence theorems (Dabagia, Papadimitriou & Vempala) all assume
    ``k*p >= 3 ln n``: a target neuron must receive on the order of ``3 ln n``
    synapses FROM THE DRIVING ASSEMBLY, or the gap between the k-th and
    (k+1)-th candidate is too small for winner selection to be repeatable.

    MEASURED, and it is a cliff rather than a slope. On the mod-3 FSM
    (`research/experiments/seq_a1_exactness_sweep.py`), sweeping the state
    area's afferent count across its floor of 3 ln 500 = 18.6:

        state kp   steps recovering EXACTLY   correct trajectories
           14              4/100                    4/10
           21             80/100                   10/10
           28            100/100                   10/10

    Below the floor recovery is essentially never exact and the machine fails;
    above it the same organ runs 2000 steps without a single error.

    THE CONDITION IS PER-AREA, which is the trap this exists to catch. The A1
    run that failed had its ARC comfortably above floor (28 vs 25.6) and its
    STATE below (14 vs 18.6), because the reference's density was copied
    without checking each area separately. An organ is in-regime only when
    every area in it is.

    ``afferents`` counts ``k_source * p`` per incoming fiber, so it is governed
    by the SOURCE's assembly size, not the target's -- raising a starved area's
    own ``k`` does nothing for it. ``k`` and ``p`` are NOT interchangeable ways
    to reach a regime even though only their product appears in the floor:
    raising ``k`` spends capacity (``M_max ~ 1.15 n/k``, [[AC-CAP]]) and forces
    ``n`` up with it, so the two need setting independently.

    WHY A DRIVE MAP IS REQUIRED FOR A VERDICT. Topology cannot tell you what
    co-fires. `add_stimulus` wires the new stimulus to EVERY existing area and
    `add_area` wires every existing stimulus to the new area, so a mod-3 FSM
    reads as 13 incoming fibers on both of its areas when only two ever fire
    together. Summing those gives 182 against a floor of 18.6 and calls a
    starved area healthy -- the same over-reporting `fiber_census` warns about.
    So `driven` says which sources actually fire together, and without it this
    reports per-fiber numbers and judges on the strongest single fiber.

    Theory: [[SEQ-REGIME]]. Measurement: [[SEQ-REGIME-CLIFF]].
    """

    area: str
    n: int
    k: int
    afferents: Dict[str, float]     # source (area or stimulus) -> k_src * p
    floor: float
    driven: Optional[Sequence[str]] = None

    @property
    def total(self) -> Measured:
        """Afferents delivered by the sources that co-fire.

        UNDEFINED without a drive map, rather than a sum over every fiber that
        happens to exist -- see the class docstring. `Measured` so an undefined
        total prints as "n/a" instead of silently becoming a number.
        """
        if self.driven is None:
            return Measured.undefined("no drive map given")
        return Measured.of(sum(self.afferents.get(s, 0.0)
                               for s in self.driven))

    @property
    def largest(self) -> float:
        """Afferents from the single strongest fiber."""
        return float(max(self.afferents.values())) if self.afferents else 0.0

    @property
    def in_regime(self) -> bool:
        """Whether this area clears its floor.

        Judged on the co-firing total when a drive map is given, and otherwise
        on the strongest single fiber -- the conservative reading, since an
        area driven by one fiber at a time is only in-regime if that one fiber
        clears the floor.
        """
        total = self.total
        return (total.value >= self.floor if total.defined
                else self.largest >= self.floor)

    def __str__(self) -> str:
        tag = "ok  " if self.in_regime else "LOW "
        shown = (self.driven if self.driven is not None
                 else sorted(self.afferents, key=lambda s: -self.afferents[s])[:4])
        detail = ", ".join(f"{s}={self.afferents.get(s, 0.0):.0f}" for s in shown)
        if self.driven is None and len(self.afferents) > len(shown):
            detail += f", +{len(self.afferents) - len(shown)} unused"
        total = self.total
        amount = (f"{total.value:6.1f}" if total.defined
                  else f"{self.largest:6.1f} (max single)")
        return (f"[{tag}] {self.area:<18} n={self.n:<7d} k={self.k:<5d} "
                f"afferent kp {amount} vs floor {self.floor:5.1f}  ({detail})")


def regime_audit(brain, driven: Optional[Mapping[str, Sequence[str]]] = None,
                 p: Optional[float] = None, warn: bool = True) -> List[Regime]:
    """Per-area afferent count against the ``k*p >= 3 ln n`` floor [[SEQ-REGIME]].

    Run this on any organ BEFORE concluding that a mechanism does not work.
    Several of this project's null results were recorded at ``kp`` an order of
    magnitude below floor, where the theory predicts failure regardless of the
    mechanism under test -- a null there is not evidence about the mechanism.

    ``driven`` maps a target area to the sources that FIRE TOGETHER into it,
    e.g. ``{"ARC": ["STATE", "sym_4"], "STATE": ["ARC"]}``. Supply it whenever
    a verdict is wanted: without it the sum over every existing fiber is
    meaningless, because the engine wires every stimulus to every area (a mod-3
    FSM reads as 13 incoming fibers where 2 fire). Omitted, each area is judged
    on its strongest single fiber and the total prints as "n/a".

    Counts stimulus fibers as well as area fibers, because a conjunction is
    routinely driven by one of each and counting only areas halves the number
    that matters.

    ``p`` overrides the brain's global density for the calculation; per-fiber
    overrides (`Brain.add_connectivity`) are read from the engine when it
    exposes them. This reads DECLARED densities, not realized ones, so it is a
    topology check like `pricing_exposure` -- pair it with `fiber_census` if
    you also need to know a fiber is actually delivering.
    """
    base_p = float(p if p is not None else getattr(brain, "p", 0.0) or 0.0)
    out: List[Regime] = []
    for name, area in brain.areas.items():
        try:
            eng = brain.engine_for(name)
        except Exception:                                    # noqa: BLE001
            continue
        n = int(getattr(area, "n", 0) or 0)
        if n <= 1:
            continue
        per_fiber = getattr(eng, "_fiber_p", {}) or {}
        afferents: Dict[str, float] = {}

        for src_name, per_dst in getattr(eng, "_area_conns", {}).items():
            if per_dst.get(name) is None:
                continue
            src = brain.areas.get(src_name)
            if src is None:
                continue
            fp = float(per_fiber.get((src_name, name), base_p))
            afferents[src_name] = int(getattr(src, "k", 0) or 0) * fp

        for stim_name, per_dst in getattr(eng, "_stim_conns", {}).items():
            if per_dst.get(name) is None:
                continue
            stim = getattr(eng, "_stimuli", {}).get(stim_name)
            if stim is None:
                continue
            fp = float(per_fiber.get((stim_name, name), base_p))
            afferents[stim_name] = int(getattr(stim, "size", 0) or 0) * fp

        if afferents:
            out.append(Regime(name, n, int(getattr(area, "k", 0) or 0),
                              afferents, 3.0 * math.log(n),
                              None if driven is None else list(driven.get(name, ()))))

    # WARN, do not merely report. This function has been called by four
    # experiments for months and only ever PRINTED a table, which at the
    # bottom of a run log is indistinguishable from silence: the S5 organ
    # printed `kp 28.0 vs floor 29.7` on every single run and the violation
    # went unacted-on until a study was built to explain the resulting
    # defects, and a later recurrence study ran at kp 2.5 against a floor of
    # 22.8 -- 9.1x below -- and read its own null as a fact about the
    # mechanism. Both are exactly the mistake the docstring above warns about.
    # A RuntimeWarning reaches stderr, survives into logs, and can be promoted
    # to an error with -W; a printed row cannot do any of those.
    if warn:
        # An area the drive map does not mention is NOT being driven in this
        # protocol, so it has nothing to be in or out of regime about.
        # Flagging it reports a dead fiber for every bystander area and
        # trains the reader to ignore the warning -- which is precisely the
        # failure this warning exists to correct.
        bad = [r for r in out if not r.in_regime
               and (r.driven is None or len(r.driven) > 0)]
        if bad:
            import warnings
            def _row(r):
                kp = float(r.total) if r.total.defined else r.largest
                # A zero-afferent area is not "10^10 x below floor" -- it is
                # a fiber that delivers NOTHING, which is a different defect
                # ([[silent-no-op-dead-fibers]]) and should read as one.
                ratio = (f"{r.floor / kp:.1f}x below" if kp > 0
                         else "NO AFFERENTS -- dead fiber, not a regime miss")
                return f"{r.area}: kp={kp:.1f} vs floor {r.floor:.1f} ({ratio})"
            rows = "; ".join(_row(r) for r in bad)
            warnings.warn(
                f"OUT OF REGIME [[SEQ-REGIME]]: {rows}. The sequence theorems "
                f"assume k*p >= 3 ln n; below it winner selection is not "
                f"repeatable and the theory predicts failure REGARDLESS of the "
                f"mechanism under test -- a null measured here is not evidence "
                f"about that mechanism. Raise the fiber density "
                f"(`Brain.add_connectivity`) or k, or lower n. Pass "
                f"warn=False to silence once you have decided out-of-regime is "
                f"what you meant to measure.",
                RuntimeWarning, stacklevel=2)
    return out


def require_regime(brain, driven=None, p=None):
    """`regime_audit` as a HARD GATE: raise unless every area clears its floor.

    For experiments that should not start at all out of regime. The warning
    emitted by `regime_audit` is the right default -- plenty of legitimate work
    is deliberately below floor -- but a registered study whose conclusion
    would be void out of regime should say so in code rather than in a note
    nobody re-reads.

    Returns the audit rows on success so the caller can still print them.
    """
    rows = regime_audit(brain, driven, p, warn=False)
    bad = [r for r in rows if not r.in_regime
           and (r.driven is None or len(r.driven) > 0)]
    if bad:
        detail = "; ".join(
            f"{r.area}: kp="
            f"{float(r.total) if r.total.defined else r.largest:.1f}"
            f" vs floor {r.floor:.1f}" for r in bad)
        raise RuntimeError(
            f"refusing to run OUT OF REGIME [[SEQ-REGIME]]: {detail}. "
            f"A null here is not evidence about the mechanism under test.")
    return rows
