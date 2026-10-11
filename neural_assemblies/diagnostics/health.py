"""Area health: has an area collapsed its items into one, is a probe dead, is
stability being mistaken for identity.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import (List, Mapping, Optional)


from neural_assemblies.core.measurement import Measured

from .readout import _spread, assembly_overlap, read_assembly
from .verdict import Verdict


# --------------------------------------------------------------------------
# Area health
# --------------------------------------------------------------------------

#: The default for every quantity that only exists once cues are supplied.
#: Shared because `Measured` is frozen, and named because "no cues" is a
#: DIFFERENT reason from any of the ones the measurement itself can produce.
_NO_CUES = Measured.undefined(
    "no cues were supplied, so only distinctness was measured")


@dataclass
class AreaHealth:
    """Whether an area can still tell its occupants apart.

    Every quantity is a `Measured`, not a float. These four used to default to
    NaN, which is the ⊥-invention this module exists to catch: `nan > ch + 0.2`
    is False, so an accuracy that was never measured produced the verdict
    "not discriminable" -- a claim about the AREA -- from a fact about the
    MEASUREMENT. See `margin` for the case where the best possible result was
    the one that went undefined.
    """

    area: str
    n_items: int
    floor: Measured                 #: chance pairwise overlap, k/n
    spread: Measured                #: measured mean pairwise overlap
    distinct_frac: Measured = _NO_CUES  #: unique assemblies / items stored
    accuracy: Measured = _NO_CUES   #: rank-1 identity, if cues were supplied
    margin: Measured = _NO_CUES     #: best match / second best
    identity: Measured = _NO_CUES   #: re-cue overlap with what was stored

    #: Probes whose runner-up overlapped ZERO -- separation so clean the ratio
    #: is unbounded. Counted separately because it is the good outcome, and
    #: because it used to be silently dropped from the margin mean.
    unbounded_margins: int = 0
    #: Probes whose live read overlapped NOTHING stored, so best/second is 0/0.
    unmatched_reads: int = 0
    verdicts: List[Verdict] = field(default_factory=list)

    @property
    def collapsed(self) -> bool:
        """True when the area can no longer tell its occupants apart.

        Counts duplicates as collapse, not just high overlap. Partial collapse
        -- most items sharing a few assemblies while the rest stay clean --
        leaves mean pairwise overlap at the floor, so keying this on ``spread``
        alone would call a mostly-destroyed area healthy.
        """
        return any(v.label in ("distinct", "no duplicates") and not v.ok
                   for v in self.verdicts)

    @property
    def trustworthy(self) -> bool:
        """False when a verdict says the MEASUREMENT is suspect, not the area."""
        return not any(v.label in ("live probe", "index space") and not v.ok
                       for v in self.verdicts)


def area_health(brain, area: str, stored: Mapping,
                cues: Optional[Mapping] = None,
                chance: Optional[float] = None) -> AreaHealth:
    """Can *area* still distinguish the assemblies stored in it?

    Args:
        brain: the live Brain.
        area: area name.
        stored: {key: assembly} of neuron IDs, e.g. from ``read_assembly``.
        cues: optional {key: callable} that re-presents item ``key`` and leaves
            *area* holding whatever it retrieves. When given, accuracy, margin
            and identity are measured; without it only distinctness is.
        chance: rank-1 chance level. Defaults to 1/len(stored).

    Returns:
        AreaHealth. Check ``.trustworthy`` BEFORE reading the numbers -- a
        failed measurement and a failed area look identical otherwise, which is
        the whole reason this returns verdicts.
    """
    items = list(stored)
    k = int(getattr(brain.areas[area], "k", 0) or 0)
    n = int(getattr(brain.areas[area], "n", 0) or 0)
    floor = (Measured.of(k / n) if n else
             Measured.undefined("area reports n=0, so chance overlap has no "
                                "denominator", area=area, k=k))
    spread = _spread(stored.values())

    uniq = len({tuple(sorted(int(x) for x in a)) for a in stored.values()})
    frac = (Measured.of(uniq / len(items)) if items else
            Measured.undefined("nothing was stored in this area, so there is "
                               "no distinctness to measure", area=area))

    h = AreaHealth(area=area, n_items=len(items), floor=floor, spread=spread,
                   distinct_frac=frac)

    # DISTINCTNESS. Two-thirds of the way to the floor is a generous bar; the
    # collapses measured were 0.5-1.0 against floors of 0.01-0.05, so this
    # separates cleanly rather than splitting hairs.
    #
    # BOTH operands must be defined. `spread` is undefined for a single stored
    # item, and the old `floor == floor` guard only covered the floor -- so a
    # one-item area compared NaN against a real floor, got False, and was
    # reported as NOT distinct.
    if floor.defined and spread.defined:
        h.verdicts.append(Verdict(
            float(spread) < max(3 * float(floor), float(floor) + 0.05),
            "distinct",
            f"pairwise overlap {spread:.4f} vs floor {floor:.4f}"))

    # DUPLICATES, which the spread bar above cannot see. Mean pairwise overlap
    # is dominated by the pairs that DIFFER, so items landing on a handful of
    # shared assemblies barely move it: 256 items on ~58 assemblies makes only
    # ~1.3% of pairs identical, which leaves the mean at the floor. An area can
    # therefore lose most of its capacity and still pass "distinct".
    #
    # Duplicates are the failure that actually destroys a read -- two items
    # with the SAME assembly are unrecoverable no matter how well separated
    # everything else is -- so they get their own verdict rather than a
    # footnote on this one.
    if frac.defined:
        h.verdicts.append(Verdict(
            float(frac) >= 0.9, "no duplicates",
            f"{uniq}/{len(items)} assemblies unique ({frac:.3f})"))

    if cues is None:
        return h

    # MARGIN HAS THREE CASES, and the old code collapsed two of them into NaN.
    # `sims[1][0] > 0` was the only branch that recorded anything, so a probe
    # whose runner-up overlapped ZERO -- the BEST possible separation -- was
    # dropped from the mean entirely. Every margin this module has ever reported
    # was therefore an average over the IMPERFECT probes only: downward-biased by
    # construction, in every seed. With every probe perfect, `margins` came out
    # empty and the margin read NaN, which the dead-probe check cannot fire on
    # (`abs(nan - 1.0) < 1e-9` is False) and which `overnight_characterization`
    # filtered out by hand. Measured directly: three disjoint assemblies with
    # exact re-cue give accuracy 1.0000 and `margin nan`, printed as an OK
    # verdict.
    #
    # Unbounded is not undefined. best/0 with best > 0 IS infinity, and saying
    # so makes the mean infinite -- loud, and impossible to publish by accident.
    # Only 0/0, where the live read overlapped nothing stored at all, is
    # genuinely undefined.
    hits, margins, idents = 0, [], []
    unbounded = unmatched = 0
    read_signatures = set()
    for key in items:
        cues[key]()
        live = read_assembly(brain, area)
        read_signatures.add(tuple(sorted(int(x) for x in live)))
        sims = sorted(((assembly_overlap(live, a), j)
                       for j, a in stored.items()), reverse=True)
        hits += sims[0][1] == key
        idents.append(assembly_overlap(live, stored[key]))
        if len(sims) > 1:
            best, runner_up = sims[0][0], sims[1][0]
            if best <= 0:
                unmatched += 1
            elif runner_up > 0:
                margins.append(best / runner_up)
            else:
                unbounded += 1
                margins.append(float("inf"))

    h.unbounded_margins, h.unmatched_reads = unbounded, unmatched
    h.accuracy = (Measured.of(hits / len(items)) if items else
                  Measured.undefined("nothing was stored, so there is no "
                                     "rank-1 identity to score", area=area))
    h.identity = (Measured.of(statistics.mean(idents)) if idents else
                  Measured.undefined("no cue produced a read to compare with "
                                     "what was stored", area=area))
    h.margin = (Measured.of(statistics.mean(margins)) if margins else
                Measured.undefined(
                    "only one item is stored, so there is no runner-up"
                    if len(items) < 2 else
                    "every live read overlapped nothing stored, so best/second "
                    "is 0/0", area=area, items=len(items), unmatched=unmatched))
    ch = chance if chance is not None else (1.0 / len(items) if items else 0.0)

    # DEAD PROBE. Exactly chance with a unit margin means every read returned
    # the same thing -- usually recruitment blocked under read_only() in an
    # area that was never materialised. This is a claim about the MEASUREMENT,
    # so it is checked before anything is concluded about the area.
    #
    # It is stated as a check on DEFINED values only. An undefined margin
    # cannot fire it -- `abs(nan - 1.0) < 1e-9` is False -- so leaving the old
    # arithmetic in place meant an unmeasurable probe was certified live.
    #
    # CONSTANT READS ARE CHECKED DIRECTLY, not inferred from the margin. The
    # margin only reaches 1.00 when the STORED assemblies are degenerate too;
    # with constant reads against DISTINCT stored items the runner-up overlap
    # is zero, so the margin is unbounded (and used to be NaN) and the guard
    # missed the very thing it was written to catch. The reads are in hand, so
    # ask them.
    constant_reads = len(read_signatures) == 1 and len(items) > 1
    if constant_reads:
        h.verdicts.append(Verdict(
            False, "live probe",
            f"all {len(items)} cues returned the SAME assembly -- the probe is "
            f"not reading area state. Materialise the area OUTSIDE the probe "
            f"before measuring"))
    elif h.margin.defined and h.accuracy.defined:
        dead = (abs(float(h.margin) - 1.0) < 1e-9) or (
            float(h.margin) < 1.001 and abs(float(h.accuracy) - ch) < 1e-9)
        h.verdicts.append(Verdict(
            not dead, "live probe",
            "margin is exactly 1.00 and accuracy is exactly chance -- every "
            "read returned the same assembly. Materialise the area OUTSIDE "
            "the probe before measuring" if dead else f"margin {h.margin:.2f}x"))
    else:
        # An unmeasurable margin is a fact about the MEASUREMENT, so it fails
        # the same verdict that `.trustworthy` keys on rather than passing it.
        h.verdicts.append(Verdict(
            False, "live probe",
            f"margin is undefined: {h.margin.why}"
            if not h.margin.defined else
            f"accuracy is undefined: {h.accuracy.why}"))

    if h.accuracy.defined:
        h.verdicts.append(Verdict(
            float(h.accuracy) > ch + 0.2, "discriminable",
            f"rank-1 {h.accuracy:.4f} vs chance {ch:.4f}"))

    # STABILITY IS NOT DISCRIMINABILITY. High identity with chance accuracy is
    # the signature of a collapsed area: every item stably returns THE one
    # assembly. Reported as its own verdict because reading identity alone is
    # exactly how this was missed.
    if (h.identity.defined and h.accuracy.defined
            and float(h.identity) > 0.5 and float(h.accuracy) < ch + 0.2):
        h.verdicts.append(Verdict(
            False, "stability != identity",
            f"re-cue overlap {h.identity:.4f} looks healthy but rank-1 is "
            f"{h.accuracy:.4f} -- items are STABLE and INDISTINGUISHABLE"))

    # A margin this thin is one step from failing even while accuracy is high.
    if (h.margin.defined and h.accuracy.defined
            and float(h.margin) < 1.2 and float(h.accuracy) > 0.9):
        h.verdicts.append(Verdict(
            False, "margin thin",
            f"accuracy {h.accuracy:.4f} rests on a {h.margin:.2f}x margin; "
            f"treat as about to fail, not as passing"))

    return h
