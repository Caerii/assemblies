"""norm_init on materialized and per-fiber connectomes: the REAL invariants.

CORRECTION HISTORY, kept because the wrong version was committed (b21d353).
This file first claimed norm_init was "a silent no-op on materialize_area",
from a diagnostic that compared STORED column masses. That tested the wrong
invariant: `_norm_scale` is deliberately a READ-TIME scale -- storage stays on
the unit scale so `w_max` semantics and sampled-candidate commensurability
survive, and the division by d_j happens at drive time (see `_norm_scale`'s
docstring, which says exactly this and was read too late). Measured with the
right instrument, norm_init is LIVE on materialized fibers: winners differ
with it on vs off (overlap 0.75 at k=40).

THE ACTUAL DEFECT found by re-diagnosing the S5 intervention flood
(27 -> 5861 soft pairs): `_norm_scale` prices UNMATERIALIZED rows at the
GLOBAL brain p while a per-fiber connectome's rows arrive at the FIBER p::

    inverse_indegree(deg, unknown, 0, self.p)      # line ~1049

Measured on a p=0.4 fiber inside a p=0.05 brain, source 86/2000 materialized:
engine d_j = 130.0 (exactly the global-p formula), correct d_j = 799.9 --
drive over-scaled 6.15x, and the factor DRIFTS toward 1x as the source
materializes, so a training run potentiates under a moving mis-scale. This is
the same defect class as the w_max clamp fixed the same morning ("clamp
scaled by GLOBAL p while weights drawn at fiber p").
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import read_assembly


def _hetero_brain(norm_init: bool):
    random.seed(7)
    np.random.seed(7)
    b = Brain(p=0.05, save_winners=True, seed=7, engine="numpy_sparse",
              norm_init=norm_init)
    b.add_area("SRC", 2000, 40, 0.1)
    b.add_area("TGT", 1200, 40, 0.1)
    b.add_connectivity("SRC", "TGT", 0.4)
    b.add_stimulus("s", 40)
    return b


def test_norm_init_is_live_on_materialized_fibers():
    """The corrected claim: read-time normalization DOES reach a
    materialize_area fiber. This is what falsifies the retracted "silent
    no-op" -- stored weights are identical by design; the DRIVE is not."""
    winners = {}
    for flag in (False, True):
        b = _hetero_brain(flag)
        b.materialize_area("TGT")
        b.materialize_area("SRC")
        for _ in range(4):
            b.project({"s": ["SRC"]}, {})
        with b.frozen():
            b.project({}, {"SRC": ["TGT"]})
        winners[flag] = set(read_assembly(b, "TGT").tolist())
    assert winners[False] != winners[True], (
        "norm_init changed nothing at read time -- if this regresses, the "
        "scale application in project_into is dead, which is the failure the "
        "retracted version of this file wrongly diagnosed at storage level")


@pytest.mark.xfail(
    reason="_norm_scale prices unmaterialized rows at the GLOBAL p, not the "
           "fiber's own p: measured d_j 130.0 vs correct 799.9 (6.15x "
           "over-scaling) on a p=0.4 fiber in a p=0.05 brain. Same class as "
           "the per-fiber w_max clamp bug. Fix: pass the fiber p at the "
           "three _norm_scale call sites.",
    strict=False,
)
def test_norm_scale_unknown_rows_use_the_fiber_p():
    b = _hetero_brain(True)
    b.materialize_area("TGT")
    for _ in range(3):
        b.project({"s": ["SRC"]}, {})
    b.project({}, {"SRC": ["TGT"]})
    eng = b._engine
    conn = eng._area_conns["SRC"]["TGT"]
    src = eng._areas["SRC"]
    rows_known = min(src.w, conn.weights.shape[0])
    nscale = eng._norm_scale(conn, 2000, rows_known, 1200)
    d_used = 1.0 / np.asarray(nscale)
    w = conn.weights
    dense = np.asarray(w.todense() if hasattr(w, "todense") else w)
    counts = (dense[:rows_known, :1200] != 0).sum(axis=0)
    unknown = 2000 - rows_known
    correct = counts + 0.40 * unknown
    ratio = float(correct.mean() / d_used.mean())
    assert 0.8 < ratio < 1.25, (
        f"d_j off by {ratio:.2f}x -- unknown rows priced at the wrong p")
