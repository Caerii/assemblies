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
import statistics
import warnings
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

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

def read_assembly(brain, area: str) -> np.ndarray:
    """Current assembly in *area* as STABLE NEURON IDS.

    Always use this rather than ``brain.areas[area].winners``, which is a
    different coordinate system (compact engine indices, renumbered as the area
    recruits). Comparing the two returns chance -- silently, looking exactly
    like a negative result. See failure shape 3 in the module docstring.
    """
    from neural_assemblies.assembly_calculus.ops import _snap
    return np.asarray(_snap(brain, area).winners, dtype=np.int64)


def assembly_overlap(a, b) -> float:
    """Overlap between two assemblies of neuron IDs. Order-insensitive."""
    from neural_assemblies.assembly_calculus.assembly import overlap
    return float(overlap(np.asarray(a, dtype=np.int64),
                         np.asarray(b, dtype=np.int64)))


def _spread(assemblies: Iterable) -> float:
    pairs = list(itertools.combinations(list(assemblies), 2))
    if not pairs:
        return float("nan")
    return statistics.mean(assembly_overlap(x, y) for x, y in pairs)


# --------------------------------------------------------------------------
# Area health
# --------------------------------------------------------------------------

@dataclass
class AreaHealth:
    """Whether an area can still tell its occupants apart."""

    area: str
    n_items: int
    floor: float                    #: chance pairwise overlap, k/n
    spread: float                   #: measured mean pairwise overlap
    distinct_frac: float = float("nan")  #: unique assemblies / items stored
    accuracy: float = float("nan")  #: rank-1 identity, if cues were supplied
    margin: float = float("nan")    #: best match / second best
    identity: float = float("nan")  #: re-cue overlap with what was stored
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
    floor = (k / n) if n else float("nan")
    spread = _spread(stored.values())

    uniq = len({tuple(sorted(int(x) for x in a)) for a in stored.values()})
    frac = (uniq / len(items)) if items else float("nan")

    h = AreaHealth(area=area, n_items=len(items), floor=floor, spread=spread,
                   distinct_frac=frac)

    # DISTINCTNESS. Two-thirds of the way to the floor is a generous bar; the
    # collapses measured were 0.5-1.0 against floors of 0.01-0.05, so this
    # separates cleanly rather than splitting hairs.
    if floor == floor:
        h.verdicts.append(Verdict(
            spread < max(3 * floor, floor + 0.05), "distinct",
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
    if frac == frac:
        h.verdicts.append(Verdict(
            frac >= 0.9, "no duplicates",
            f"{uniq}/{len(items)} assemblies unique ({frac:.3f})"))

    if cues is None:
        return h

    hits, margins, idents = 0, [], []
    for key in items:
        cues[key]()
        live = read_assembly(brain, area)
        sims = sorted(((assembly_overlap(live, a), j)
                       for j, a in stored.items()), reverse=True)
        hits += sims[0][1] == key
        idents.append(assembly_overlap(live, stored[key]))
        if len(sims) > 1 and sims[1][0] > 0:
            margins.append(sims[0][0] / sims[1][0])

    h.accuracy = hits / len(items) if items else float("nan")
    h.margin = statistics.mean(margins) if margins else float("nan")
    h.identity = statistics.mean(idents) if idents else float("nan")
    ch = chance if chance is not None else (1.0 / len(items) if items else 0.0)

    # DEAD PROBE. Exactly chance with a unit margin means every read returned
    # the same thing -- usually recruitment blocked under read_only() in an
    # area that was never materialised. This is a claim about the MEASUREMENT,
    # so it is checked before anything is concluded about the area.
    dead = (abs(h.margin - 1.0) < 1e-9) or (
        h.margin == h.margin and h.margin < 1.001 and abs(h.accuracy - ch) < 1e-9)
    h.verdicts.append(Verdict(
        not dead, "live probe",
        "margin is exactly 1.00 and accuracy is exactly chance -- every read "
        "returned the same assembly. Materialise the area OUTSIDE the probe "
        "before measuring" if dead else f"margin {h.margin:.2f}x"))

    h.verdicts.append(Verdict(
        h.accuracy > ch + 0.2, "discriminable",
        f"rank-1 {h.accuracy:.4f} vs chance {ch:.4f}"))

    # STABILITY IS NOT DISCRIMINABILITY. High identity with chance accuracy is
    # the signature of a collapsed area: every item stably returns THE one
    # assembly. Reported as its own verdict because reading identity alone is
    # exactly how this was missed.
    if h.identity > 0.5 and h.accuracy < ch + 0.2:
        h.verdicts.append(Verdict(
            False, "stability != identity",
            f"re-cue overlap {h.identity:.4f} looks healthy but rank-1 is "
            f"{h.accuracy:.4f} -- items are STABLE and INDISTINGUISHABLE"))

    # A margin this thin is one step from failing even while accuracy is high.
    if h.margin == h.margin and h.margin < 1.2 and h.accuracy > 0.9:
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

    eng_t = brain._engine_for(brain.areas[target])
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
    for a in names:
        try:
            eng = brain._engine_for(brain.areas[a])
        except Exception:
            continue
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
            eng = brain._engine_for(brain.areas[dst_name])
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
                out.append(FiberState(src_name, dst_name, int(rows),
                                      int(cols), nnz, ratio, dst_w))

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
            eng = brain._engine_for(dst)
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
            lines.append(
                f"  {it.area:<16} items {it.n_items:<5} spread "
                f"{it.spread:.4f} (floor {it.floor:.4f})"
                # Printed next to spread ON PURPOSE: these two disagree exactly
                # when partial collapse is happening, and seeing the pair is
                # what makes that visible at a glance.
                + (f"  distinct {it.distinct_frac:.3f}"
                   if it.distinct_frac == it.distinct_frac else "")
                + (f"  acc {it.accuracy:.4f}  margin {it.margin:.2f}x"
                   if it.accuracy == it.accuracy else ""))
            lines += [f"      {v}" for v in it.verdicts if not v.ok]
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
    by_arm: Dict[str, object]

    def ratio(self, arm: str = "sampled", truth: str = "explicit"):
        """`arm / truth`, elementwise for sequences, else scalar."""
        a, t = self.by_arm.get(arm), self.by_arm.get(truth)
        if a is None or t is None:
            return None
        try:
            return [float(x) / float(y) if y else float("nan")
                    for x, y in zip(a, t)]
        except TypeError:
            return float(a) / float(t) if t else float("nan")

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
                eng = brain._engine_for(area)
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
    vals = [float(run(s)) for s in seeds]
    mean = statistics.mean(vals)
    # t critical value, two-sided 95%, for the small n used in practice.
    tcrit = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
             9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}.get(len(vals), 1.96)
    ci = tcrit * statistics.stdev(vals) / len(vals) ** 0.5
    return Ensemble(label, tuple(vals), mean, ci)


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
    diffs = [x - y for x, y in zip(a.values, b.values)]
    mean = statistics.mean(diffs)
    tcrit = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
             9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}.get(len(diffs), 1.96)
    ci = tcrit * statistics.stdev(diffs) / len(diffs) ** 0.5
    return Ensemble(label, tuple(diffs), mean, ci)
