"""Drive decomposition: where a target area's drive comes from, fiber by fiber.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Dict, List, Optional, Sequence)

import numpy as np


from .readout import read_assembly
from .verdict import Verdict


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
