"""What each fiber holds and what the sampler charges for it: the fiber census and
the pricing exposure.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Dict, List, Mapping, Optional, Sequence)

import numpy as np


from .verdict import Verdict


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
