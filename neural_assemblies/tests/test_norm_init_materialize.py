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


def test_norm_scale_unknown_rows_use_the_fiber_p():
    """REGRESSION for the global-p defect: d_j 130.0 vs correct 799.9 (6.15x)
    before the fix. Unmaterialized rows must be priced at the FIBER p.

    Observed THROUGH THE REAL PATH: the first draft called `_norm_scale`
    directly, which bypasses the call sites the fix changed and silently
    tests the fallback. The projection is what must pass the fiber p.
    """
    b = _hetero_brain(True)
    b.materialize_area("TGT")
    for _ in range(3):
        b.project({"s": ["SRC"]}, {})
    # Prime the fiber: the FIRST SRC->TGT projection takes the bootstrap
    # path, which does not consult _norm_scale at all. The steady-state
    # drive path is the one under test.
    b.project({}, {"SRC": ["TGT"]})
    eng = b._engine
    seen = {}
    orig = eng._norm_scale

    def recording(conn, n_pre, rows_known, needed, p=None):
        seen[(n_pre, needed)] = p
        return orig(conn, n_pre, rows_known, needed, p=p)

    eng._norm_scale = recording
    try:
        b.project({}, {"SRC": ["TGT"]})
    finally:
        eng._norm_scale = orig
    area_calls = [p for (n_pre, _), p in seen.items() if n_pre == 2000]
    assert area_calls, "the SRC->TGT projection never consulted _norm_scale"
    assert all(p == 0.4 for p in area_calls), (
        f"projection passed p={area_calls} for a 0.4 fiber -- unmaterialized "
        f"rows are being priced at the wrong density again")

    # And the arithmetic itself, given the right p:
    conn = eng._area_conns["SRC"]["TGT"]
    src = eng._areas["SRC"]
    rows_known = min(src.w, conn.weights.shape[0])
    nscale = orig(conn, 2000, rows_known, 1200, p=0.4)
    d_used = 1.0 / np.asarray(nscale)
    w = conn.weights
    dense = np.asarray(w.todense() if hasattr(w, "todense") else w)
    counts = (dense[:rows_known, :1200] != 0).sum(axis=0)
    correct = counts + 0.40 * (2000 - rows_known)
    ratio = float(correct.mean() / d_used.mean())
    assert 0.8 < ratio < 1.25, f"d_j off by {ratio:.2f}x with the fiber p given"
