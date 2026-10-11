"""Whole-brain sweeps: self-recurrent fibers (the collapse channel) and every
area's health at once.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from typing import (Dict, List, Mapping, Optional, Sequence)

import numpy as np


from .health import AreaHealth, area_health
from .verdict import Verdict


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
