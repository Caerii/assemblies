"""Runtime diagnostics for assemblies: is this measurement trustworthy?

WHY THIS MODULE EXISTS
----------------------
On 2026-07-28/29 a research session produced eight experiments, six of which
refuted the one before them. Not one of the six was refuted by new theory --
each was refuted by a CONTROL that cost almost nothing to add and had simply
not been there. The failures fell into four repeatable shapes:

  1. COLLAPSE UPSTREAM. An area holding many items merges them into one, and
     everything downstream then reads exactly chance. Six hours were spent on
     "composition fails at depth" before a spread column showed the PARENTS had
     become a single assembly before composition ever ran.
  2. DEAD PROBE. Reading under `read_only()` from an area that was never
     materialised returns the same degenerate winners for every input, so
     accuracy is EXACTLY chance and margin is EXACTLY 1.00. Twice mistaken for
     a negative result.
  3. WRONG INDEX SPACE. `area.winners` holds compact engine indices;
     `ops._snap` returns stable neuron IDs. Comparing across them is silently
     at chance.
  4. STABILITY MISTAKEN FOR DISCRIMINABILITY. After collapse every item still
     re-cues to high overlap with what was stored, because it returns THE
     collapsed assembly. A probe reading only self-overlap reports success at
     0.7763 while rank-1 identity is 0.0143.

Every one of those is mechanically detectable. This module detects them. The
organising idea is that a diagnostic should answer "can I believe this number?",
not merely "what is this number?" -- so the functions here return verdicts and
reasons, not just floats.

WHAT THIS IS NOT
----------------
Not a metrics library. `research/experiments/_substrate.py` covers reading,
similarity and probing for experiments. This is for interrogating a LIVE brain,
including mid-training, and is safe to call from production code.
"""

from __future__ import annotations

import itertools
import math
import statistics
import warnings
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.core.measurement import Measured

__all__ = [
    "Verdict", "AreaHealth", "DriveBreakdown", "FiberState",
    "PricingExposure",
    "read_assembly", "assembly_overlap",
    "area_health", "drive_breakdown", "recurrence_audit", "collapse_scan",
    "fiber_census", "pricing_exposure",
    "format_report",
    "Arbitration", "arbitrate", "arbitrate_prebuilt", "ARBITER_ARMS",
    "Arm", "arm_spec",
    "Ensemble", "ensemble", "compare_arms", "paired_delta",
    "LoadGap", "area_load", "load_audit",
    "GainStability", "gain_stability",
    "Separation", "separation",
    "ProbeCheck", "verify_probe",
]


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

@dataclass
class Verdict:
    """A judgement plus the reason for it. The reason is the point.

    A bare boolean sends the reader back to the code to find out what was
    checked; every failure this module exists to catch was originally missed by
    someone reading a number without its provenance.
    """

    ok: bool
    label: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        mark = "OK  " if self.ok else "WARN"
        return f"[{mark}] {self.label}{(': ' + self.detail) if self.detail else ''}"


# --------------------------------------------------------------------------
# The one sanctioned readout
# --------------------------------------------------------------------------

def read_assembly(brain, area: str) -> NeuronIds:
    """Current assembly in *area* as STABLE NEURON IDS.

    Always use this rather than ``brain.areas[area].winners``, which is a
    different coordinate system (compact engine indices, renumbered as the area
    recruits). Comparing the two returns chance -- silently, looking exactly
    like a negative result. See failure shape 3 in the module docstring.
    """
    from neural_assemblies.assembly_calculus.ops import _snap
    return NeuronIds(np.asarray(_snap(brain, area).winners, dtype=np.int64))


def assembly_overlap(a: NeuronIds, b: NeuronIds) -> float:
    """Overlap between two assemblies of NEURON IDS. Order-insensitive.

    Both operands are asserted into the neuron-ID space, which is what makes
    this the sanctioned pairing for `read_assembly`. Handing it compact engine
    indices is the defect in `core/index_spaces` -- it will return a plausible
    number that reads as chance -- so read through `read_assembly`, never off
    `area.winners` directly.
    """
    from neural_assemblies.assembly_calculus.assembly import overlap
    return float(overlap(NeuronIds(np.asarray(a, dtype=np.int64)),
                         NeuronIds(np.asarray(b, dtype=np.int64))))


def _spread(assemblies: Iterable) -> Measured:
    """Mean pairwise overlap. UNDEFINED with fewer than two assemblies.

    One assembly has no pair to overlap with, so there is no spread -- as
    distinct from a spread of zero, which is what a bare 0.0 here would claim
    and which is the *healthiest* possible reading.
    """
    items = list(assemblies)
    pairs = list(itertools.combinations(items, 2))
    if not pairs:
        return Measured.undefined(
            "fewer than two assemblies, so there is no pair to overlap",
            n_assemblies=len(items))
    return Measured.of(statistics.mean(assembly_overlap(x, y) for x, y in pairs))


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


# --------------------------------------------------------------------------
# Drive decomposition
# --------------------------------------------------------------------------

@dataclass
class DriveBreakdown:
    """Where an area's input actually comes from."""

    target: str
    per_source: Dict[str, float]
    verdicts: List[Verdict] = field(default_factory=list)

    @property
    def total(self) -> float:
        vals = [v for v in self.per_source.values() if v == v]
        return sum(vals) if vals else float("nan")

    def share(self, source: str) -> float:
        t = self.total
        return (self.per_source.get(source, float("nan")) / t) if t else float("nan")


def drive_breakdown(brain, target: str, sources: Sequence[str],
                    target_ids=None,
                    expect_controlling: Optional[str] = None) -> DriveBreakdown:
    """Mean synaptic weight each source delivers onto *target*'s assembly.

    Read straight from the connectome, so no projection is run and neither
    k-WTA nor settling can intervene. This is the measurement that decides
    which input wins the k-WTA, and it is the one that was missing when five
    consecutive interventions on multi-mood word order all failed: the
    conditioning signal controlled 4% of the drive, and nothing anyone changed
    altered that share.

    Args:
        target: area whose incoming drive is decomposed.
        sources: source areas to attribute drive to. Include *target* itself to
            measure self-recurrence, which is usually the largest term and is
            usually the one nobody looked at.
        target_ids: neuron IDs to score against. Defaults to *target*'s current
            assembly.
        expect_controlling: source you believe decides the outcome. When given,
            a verdict fires if it does not hold a plurality of the drive.
    """
    from neural_assemblies.assembly_calculus.ops import _compact_index

    if target_ids is None:
        target_ids = read_assembly(brain, target)

    eng_t = brain.engine_for(target)
    t_inv = _compact_index(eng_t, target) or {}

    per: Dict[str, float] = {}
    for src in sources:
        conn = getattr(eng_t, "_area_conns", {}).get(src, {}).get(target)
        w = getattr(conn, "weights", None)
        if w is None or getattr(w, "shape", (0, 0))[0] == 0:
            per[src] = float("nan")
            continue
        w = np.asarray(w.todense() if hasattr(w, "todense") else w)
        src_win = np.asarray(brain.areas[src].winners)
        rows = [int(x) for x in src_win if int(x) < w.shape[0]]
        cols = [t_inv[int(x)] for x in target_ids
                if int(x) in t_inv and t_inv[int(x)] < w.shape[1]]
        per[src] = (float(w[np.ix_(rows, cols)].mean())
                    if rows and cols else float("nan"))

    d = DriveBreakdown(target=target, per_source=per)

    if expect_controlling is not None:
        share = d.share(expect_controlling)
        others = {s: v for s, v in per.items()
                  if s != expect_controlling and v == v}
        top = max(others, key=lambda s: others[s]) if others else None
        d.verdicts.append(Verdict(
            share == share and share > 0.4, "controlling source",
            f"{expect_controlling} holds {share:.1%} of the drive"
            + (f"; {top} holds {d.share(top):.1%}" if top else "")
            + ("" if (share == share and share > 0.4) else
               " -- it cannot decide the k-WTA, and any fix that leaves this "
               "share unchanged will fail")))
    return d


# --------------------------------------------------------------------------
# Whole-brain sweeps
# --------------------------------------------------------------------------

def recurrence_audit(brain, areas: Optional[Sequence[str]] = None
                     ) -> List[Verdict]:
    """Flag self-recurrent fibers, the collapse channel found at every level.

    Self-recurrence with plasticity is safe for an area holding ONE assembly
    and destroys an area holding many: the first item's self-connections
    potentiate until they beat each later item's input. Measured ceilings under
    norm_init were 32 / 64 / 256 items at n = 1000 / 2000 / 4000; feed-forward
    had no measurable ceiling at all.

    This reports which self-fibers EXIST and how potentiated they are. It cannot
    know how many items an area is meant to hold, so it reports rather than
    judges -- but a heavily potentiated self-fiber on a shared area is the first
    thing to check when retrieval reads chance.

    STATISTIC. The tail-to-median weight ratio, NOT the matrix mean.

    A whole-matrix mean dilutes the potentiated entries into the unpotentiated
    bulk: measured on a self-fiber whose active block averaged 1.95 (39x the
    p=0.05 baseline), the matrix mean did not clear a 2x threshold at all and
    this check silently passed. Comparing to `p` is also wrong under norm_init,
    where the initial weight is normalised per postsynaptic neuron rather than
    set to p.

    p99 / median is free of both problems. A freshly initialised fiber has a
    narrow weight distribution whatever the normalisation, so the ratio is near
    1; Hebbian potentiation concentrates on the assemblies that co-fired and
    produces a heavy tail.
    """  # noqa: D208
    out: List[Verdict] = []
    names = list(areas) if areas is not None else list(brain.areas)
    unknown = sorted(set(names) - set(brain.areas))
    if unknown:
        raise KeyError(f"recurrence_audit areas are unknown: {unknown}")
    for a in names:
        eng = brain.engine_for(a)
        conn = getattr(eng, "_area_conns", {}).get(a, {}).get(a)
        w = getattr(conn, "weights", None)
        if w is None or getattr(w, "shape", (0, 0))[0] == 0:
            continue
        arr = np.asarray(w.todense() if hasattr(w, "todense") else w).ravel()
        arr = arr[arr > 0]
        if arr.size < 8:
            continue
        med = float(np.median(arr))
        tail = float(np.percentile(arr, 99))
        ratio = (tail / med) if med > 0 else float("nan")
        hot = ratio == ratio and ratio > 3.0
        out.append(Verdict(
            not hot, f"self-fiber {a}",
            f"p99/median {ratio:.1f}x (p99 {tail:.4f}, median {med:.4f})"
            + (" -- potentiated. If this area holds MANY items this is the "
               "collapse channel; train it feed-forward" if hot else "")))
    return out


@dataclass
class FiberState:
    """One area->area pathway: does it exist, and does it carry anything?"""

    src: str
    dst: str
    rows: int
    cols: int
    nnz: int
    p99_over_median: float
    dst_w: int
    #: Logical column watermark of a LAZY fiber; None when the fiber is dense
    #: (every column exists) or the engine does not materialize lazily.
    #: None means NOT APPLICABLE and must not be read as zero.
    extent: Optional[int] = None
    #: Neurons the engine has actually materialized in `dst`, or None. NOT the
    #: same as `dst_w`, which is Brain-side and means len(winners) on an
    #: explicit area. See ComputeEngine.fiber_extent.
    dst_materialized: Optional[int] = None

    @property
    def extent_desync(self) -> int:
        """``materialized - extent``: neurons of the target with NO column here.

        Zero whenever the question does not apply (dense fiber, or an engine
        that allocates all ``n`` up front), so a caller can sum this over a
        census without special-casing.

        WHAT A NONZERO VALUE MEANS. The target area grew through some other
        fiber and this one was not expanded with it, so its last columns are
        allocated-but-uninitialised zeros. Any read that slices this fiber by
        the area's neuron count therefore includes columns that deliver
        identically nothing, and any read that slices by the watermark silently
        drops real neurons.

        MEASURED. A `frozen()` probe -- plasticity off, recruitment ON -- adds
        ~26 neurons and desyncs by exactly that many, because the probe drove
        the area from a stimulus while the self fiber was not a source. The same
        probe under `read_only()` desyncs by 0. This is the observable form of
        "the probe changed the thing it was measuring".
        """
        if self.extent is None or self.dst_materialized is None:
            return 0
        return int(self.dst_materialized) - int(self.extent)

    @property
    def dead(self) -> bool:
        """No weights at all -- this pathway delivers exactly zero drive."""
        return self.cols == 0 or self.nnz == 0

    @property
    def potentiated(self) -> bool:
        r = self.p99_over_median
        return r == r and r > 3.0

    @property
    def silently_ignored(self) -> bool:
        """Dead pathway into an area that HAS materialised neurons.

        This is the dangerous combination, and it is not the same as a merely
        unused fiber. The target is live and being driven, so a projection
        naming this source looks like it works: k-WTA still returns k winners
        and the caller gets a plausible assembly. It just contains no
        information from this source.
        """
        return self.dead and self.dst_w > 0


def _safe(fn, *args):
    """Call an OPTIONAL engine accessor, returning None if it is not there.

    `fiber_extent` / `materialized_count` have defaults on the ABC, but engines
    are also constructed directly in research code and third-party subclasses
    predate them. A census that raised on an older engine would be worse than
    one that reports "not applicable", which is already the meaning of None.
    """
    try:
        v = fn(*args)
    except Exception:                                        # noqa: BLE001
        return None
    return None if v is None else int(v)


def _fiber_shape(conn):
    """``(rows, cols, nnz, p99/median)`` for a connectome of EITHER backend.

    THIS EXISTS BECAUSE THE CENSUS WAS BLIND ON TORCH.  It used to read
    ``conn.weights`` directly.  The torch engine's ``CSRConn`` has no such
    attribute, so ``getattr(conn, "weights", None)`` returned ``None``, every
    torch fiber measured as shape ``(0, 0)`` with ``nnz`` 0 -- i.e. *dead*, and
    ``silently_ignored`` whenever the target was live.

    Measured on a two-area torch brain: it flagged **4 of 4** fibers as
    silently ignored, including a healthy one at ``nnz = 1314``.  It could not
    distinguish a genuinely dead fiber from a working one, which on that engine
    makes it worse than useless -- it produces confident false positives, and
    this function's docstring has been cited as grounds to trust a negative.

    Same principle as ``core/_pricing.py``: ask the connectome, do not reach
    into one backend's storage. CSR exposes ``nnz`` / ``_nrows`` / ``_ncols``;
    the dense/1-D path exposes ``weights``.
    """
    w = getattr(conn, "weights", None)
    if w is None:
        # CSR-style (torch). Shape and nnz are first-class; the weight-spread
        # ratio needs the values, which live in `_val`.
        rows = int(getattr(conn, "_nrows", 0) or 0)
        cols = int(getattr(conn, "_ncols", 0) or 0)
        nnz = int(getattr(conn, "nnz", 0) or 0)
        ratio = float("nan")
        val = getattr(conn, "_val", None)
        if val is not None and nnz >= 8:
            try:
                pos = np.asarray(val.detach().float().cpu().numpy())
            except AttributeError:
                pos = np.asarray(val)
            pos = pos[pos > 0]
            if pos.size >= 8:
                med = float(np.median(pos))
                if med > 0:
                    ratio = float(np.percentile(pos, 99)) / med
        return rows, cols, nnz, ratio

    shape = tuple(getattr(w, "shape", (0, 0)) or (0, 0))
    rows, cols = (shape + (0, 0))[:2]
    nnz, ratio = 0, float("nan")
    if cols > 0:
        arr = np.asarray(w.todense() if hasattr(w, "todense") else w)
        pos = arr.ravel()
        pos = pos[pos > 0]
        nnz = int(pos.size)
        if pos.size >= 8:
            med = float(np.median(pos))
            if med > 0:
                ratio = float(np.percentile(pos, 99)) / med
    return rows, cols, nnz, ratio


def fiber_census(brain, driven: Optional[Mapping[str, Sequence[str]]] = None
                 ) -> List[FiberState]:
    """Every area->area pathway in the brain, and whether it can carry drive.

    WHY THIS EXISTS. An area->area weight block is materialised lazily, so a
    pathway that has never successfully carried drive has shape (0, 0) and
    contributes exactly zero -- while the projection that names it still
    returns k winners and looks like it worked. Two bugs of this shape are on
    record: a role area fed by two LEX areas silently used only the first
    (every item through the second "stored" the target's stale assembly), and
    `reset_area_connections` zeroing a connectome so k-WTA fell through to its
    index tie-break.

    The signature in results is "mechanism X turns out to have surprisingly
    little effect", which is indistinguishable from a real negative result by
    inspection of the numbers alone. This function distinguishes them.

    Args:
        driven: optional {src: [dst, ...]} of pathways the caller BELIEVES it
            is using. Any of those found dead is reported as a failed verdict
            rather than merely listed, which is the difference between a census
            and a test.

    Returns FiberState records; pass to `format_report` or filter on
    `.silently_ignored`.
    """
    out: List[FiberState] = []
    for dst_name in brain.areas:
        try:
            eng = brain.engine_for(dst_name)
        except Exception:                                    # noqa: BLE001
            continue
        dst_w = int(getattr(brain.areas[dst_name], "w", 0) or 0)
        seen = set()
        # BOTH connectome maps. The torch engine keeps explicit->sparse edges in
        # a SEPARATE `_dense_area_conns` map, which `pricing_exposure` already
        # reads and this did not -- so a dense explicit fiber was invisible to
        # the census entirely rather than merely mis-measured.
        for attr in ("_area_conns", "_dense_area_conns"):
            for src_name, per_dst in getattr(eng, attr, {}).items():
                conn = per_dst.get(dst_name)
                if conn is None or (src_name, dst_name) in seen:
                    continue
                seen.add((src_name, dst_name))
                rows, cols, nnz, ratio = _fiber_shape(conn)
                # Ask the ENGINE for the logical extent rather than reaching
                # into one backend's storage -- same principle as _fiber_shape.
                # Both are Optional and None means "not applicable", so a dense
                # fiber reports no desync instead of a spurious one.
                extent = _safe(eng.fiber_extent, src_name, dst_name)
                mat = _safe(eng.materialized_count, dst_name)
                out.append(FiberState(src_name, dst_name, int(rows),
                                      int(cols), nnz, ratio, dst_w,
                                      extent=extent, dst_materialized=mat))

    # VERDICTS ONLY WHEN THE CALLER DECLARES `driven`. Appending them
    # unconditionally would change what a plain census RETURNS -- every
    # existing caller does `[f for f in fiber_census(b) if f.dead]` and a
    # Verdict has no `.dead`. A desync is still visible without `driven`: it is
    # `.extent_desync` on the record, which is where a census belongs. The
    # difference between a census and a test is the declaration.
    if driven:
        by_pair = {(f.src, f.dst): f for f in out}
        for src, dsts in driven.items():
            for dst in dsts:
                f = by_pair.get((src, dst))
                if f is None:
                    out.append(Verdict(  # type: ignore[arg-type]
                        False, f"fiber {src}->{dst}",
                        "NO SUCH PATHWAY -- the projection cannot have run"))
                elif f.dead:
                    out.append(Verdict(  # type: ignore[arg-type]
                        False, f"fiber {src}->{dst}",
                        f"DEAD (shape {f.rows}x{f.cols}, nnz {f.nnz}) while "
                        f"{dst} has w={f.dst_w} -- this source delivers ZERO "
                        f"drive and the projection silently returns "
                        f"{dst}'s existing assembly"))
                elif f.extent_desync:
                    out.append(Verdict(  # type: ignore[arg-type]
                        False, f"fiber {src}->{dst}",
                        f"EXTENT DESYNC by {f.extent_desync}: {dst} has "
                        f"{f.dst_materialized} materialized neurons but this "
                        f"fiber has columns for only {f.extent}. The area grew "
                        f"through a different fiber and this one was not "
                        f"expanded with it, so its last {f.extent_desync} "
                        f"columns are uninitialised zeros and deliver nothing. "
                        f"A probe under frozen() does exactly this; "
                        f"read_only() does not"))
    return out


@dataclass
class PricingExposure:
    """Whether one target area sits on a path the k-WTA pricing bugs affected."""

    area: str
    n: int
    src_pops: Dict[str, int]        # source area -> its n, per incoming fiber
    explicit_srcs: List[str]        # sources that are explicit (dense) areas
    norm_init: bool

    @property
    def heterogeneous(self) -> bool:
        """Some source population differs from the target's own n.

        The pre-fix divisor was ``tgt.n * p`` for every fiber, so it was correct
        only here. Anywhere else the incumbent and candidate populations sat a
        factor ``tgt.n / n_pre`` apart, with a sign: candidates over-divided
        (small source) SEAL the area at k, under-divided (large source) never
        let it settle.
        """
        return any(pop != self.n for pop in self.src_pops.values())

    @property
    def exposed(self) -> bool:
        """Results read through this area may have been distorted."""
        return self.norm_init and (self.heterogeneous
                                   or bool(self.explicit_srcs))

    def __str__(self) -> str:
        if not self.norm_init:
            return f"{self.area}: norm_init off -- not exposed"
        if not self.exposed:
            return f"{self.area}: all sources n={self.n} -- not exposed"
        bits = []
        if self.heterogeneous:
            odd = {s: p for s, p in self.src_pops.items() if p != self.n}
            bits.append(f"heterogeneous n (target {self.n}, sources {odd})")
        if self.explicit_srcs:
            bits.append(f"explicit sources {self.explicit_srcs}")
        return f"{self.area}: EXPOSED -- " + "; ".join(bits)


def pricing_exposure(brain) -> List[PricingExposure]:
    """Which areas sit on paths the k-WTA pricing bugs distorted.

    WHY THIS EXISTS. Two engine fixes (`c7ce506`, `54e6c00`, unified in
    `5fa91ed`) changed how ``top-k`` prices materialized incumbents against
    sampled candidates under ``norm_init``. Every result recorded before them is
    potentially affected -- but only if its brain actually reached an affected
    path, and most do not. Re-recording the entire golden corpus to find out is
    expensive and, worse, uninformative about WHY a number moved.

    This answers the cheaper question first: could this brain's topology have
    touched the bug at all? Two conditions, both static:

      * a fiber whose presynaptic area has a different ``n`` from the target,
        which is what the per-fiber divisor fix corrected; and
      * a fiber from an EXPLICIT area, whose drive skipped normalization
        entirely so incumbents ran ~``n*p`` above candidates.

    An area that is not exposed cannot have moved, so its recorded numbers stand
    without a re-run. An exposed one needs re-recording, and the reported detail
    says which of the two mechanisms to expect.

    NOT EXPOSED IS NOT THE SAME AS CORRECT -- this reads topology, not dynamics,
    and says nothing about the other silent-failure classes. Pair it with
    `fiber_census` and `area_health`.

    Deliberately conservative in the other direction too: a connectome the
    engine materialized but nothing ever drove still counts as a fiber here, so
    a reverse edge can flag an area that was only ever a source. Over-reporting
    costs a re-run; under-reporting leaves a wrong number standing.
    """
    out: List[PricingExposure] = []
    norm_init = bool(getattr(brain, "norm_init", False))
    for dst_name, dst in brain.areas.items():
        try:
            eng = brain.engine_for(dst)
        except Exception:                                    # noqa: BLE001
            continue
        eng_areas = getattr(eng, "_areas", {})
        dst_n = int(getattr(dst, "n", 0) or 0)
        pops: Dict[str, int] = {}
        explicit: List[str] = []
        for src_name, per_dst in getattr(eng, "_area_conns", {}).items():
            if per_dst.get(dst_name) is None:
                continue
            src_state = eng_areas.get(src_name)
            if src_state is None:
                continue
            pops[src_name] = int(getattr(src_state, "n", 0) or 0)
            if getattr(src_state, "explicit_source", False):
                explicit.append(src_name)
        # Dense explicit->sparse edges live in their own map on the torch engine.
        for src_name, per_dst in getattr(eng, "_dense_area_conns", {}).items():
            if per_dst.get(dst_name) is None:
                continue
            src_state = eng_areas.get(src_name)
            if src_state is None:
                continue
            pops.setdefault(src_name, int(getattr(src_state, "n", 0) or 0))
            if (getattr(src_state, "explicit_source", False)
                    and src_name not in explicit):
                explicit.append(src_name)
        if pops:
            out.append(PricingExposure(dst_name, dst_n, pops, explicit,
                                       norm_init))
    return out


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


def collapse_scan(brain, stored_by_area: Mapping[str, Mapping]
                  ) -> Dict[str, AreaHealth]:
    """Distinctness check across every area holding stored assemblies.

    The cheapest useful diagnostic there is, and the one whose absence cost the
    most: run it on the LEXICON and the PARENT areas before concluding anything
    about composition or retrieval downstream. Collapse upstream reads as chance
    downstream and is indistinguishable from a genuine negative by inspection.
    """
    return {area: area_health(brain, area, stored)
            for area, stored in stored_by_area.items()}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_report(items) -> str:
    """Render health objects, breakdowns or verdicts as aligned text."""
    lines: List[str] = []
    seq = items.values() if isinstance(items, Mapping) else items
    for it in (seq if isinstance(seq, (list, tuple, type({}.values()))) else [seq]):
        if isinstance(it, AreaHealth):
            # `_num` prints "n/a" for an undefined quantity rather than a
            # number. It must never silently substitute one: the whole reason
            # these are `Measured` is that a missing spread used to print as
            # NaN and a missing margin as nothing at all.
            def _num(m: Measured, spec: str, suffix: str = "") -> str:
                return f"{m:{spec}}{suffix}" if m.defined else "n/a"

            lines.append(
                f"  {it.area:<16} items {it.n_items:<5} spread "
                f"{_num(it.spread, '.4f')} (floor {_num(it.floor, '.4f')})"
                # Printed next to spread ON PURPOSE: these two disagree exactly
                # when partial collapse is happening, and seeing the pair is
                # what makes that visible at a glance.
                + f"  distinct {_num(it.distinct_frac, '.3f')}"
                + (f"  acc {_num(it.accuracy, '.4f')}"
                   f"  margin {_num(it.margin, '.2f', 'x')}"
                   if it.accuracy.defined or it.margin.defined else "")
                + (f"  [{it.unbounded_margins} unbounded]"
                   if it.unbounded_margins else "")
                + (f"  [{it.unmatched_reads} matched nothing]"
                   if it.unmatched_reads else ""))
            lines += [f"      {v}" for v in it.verdicts if not v.ok]
        elif isinstance(it, Regime):
            lines.append("  " + str(it))
        elif isinstance(it, DriveBreakdown):
            lines.append(f"  drive into {it.target}:")
            for s, v in sorted(it.per_source.items(),
                               key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
                lines.append(f"      {s:<16}{v:>10.4f}  {it.share(s):>7.1%}")
            lines += [f"      {v}" for v in it.verdicts if not v.ok]
        elif isinstance(it, FiberState):
            tag = ("DEAD*" if it.silently_ignored else
                   "dead " if it.dead else
                   "hot  " if it.potentiated else "ok   ")
            lines.append(
                f"  [{tag}] {it.src:>14} -> {it.dst:<14} "
                f"{it.rows:>6}x{it.cols:<6} nnz {it.nnz:>8}  "
                f"dst_w {it.dst_w:>6}"
                + (f"  p99/med {it.p99_over_median:.1f}x"
                   if it.p99_over_median == it.p99_over_median else "")
                + ("   <-- delivers ZERO drive into a live area"
                   if it.silently_ignored else ""))
        elif isinstance(it, PricingExposure):
            lines.append(f"  [{'EXPOSED' if it.exposed else 'clean  '}] {it}")
        elif isinstance(it, Verdict):
            if not it.ok:
                lines.append(f"  {it}")
    return "\n".join(lines) if lines else "  (nothing flagged)"


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


# --------------------------------------------------------------------------
# Rank statistics (numpy-only; scipy.stats stays off the import path --
# the perf A2 lesson). Promoted from research/experiments/overlap_ceiling.py
# after six experiments imported them from there (#149 literate pass).
# --------------------------------------------------------------------------

def rankdata(a) -> np.ndarray:
    """Average-rank transform with ties (scipy.stats.rankdata semantics)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation; NaN when either input is constant.

    A NaN here is an OUTCOME, not noise (the undefinedness lesson):
    report the degenerate seeds explicitly, never filter them.
    """
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman(x, y) with z partialled out (least squares on ranks)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([np.ones_like(rz), rz]).T

    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    ex, ey = resid(rx), resid(ry)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


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
