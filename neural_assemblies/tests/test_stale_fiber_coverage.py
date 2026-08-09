"""A consulted fiber covers its source's growth, or one quiet episode kills it.

#151 dead-fiber hunt (dead_fiber_hunt.py, Brown at n=1e5): area->area row/col
growth lived ONLY in `_expand_connectomes`, which returns early when the
target recruits no first-time winner. So the sequence

    fiber created (rows = source's size THEN)
    -> source grows through other projections
    -> one projection through the fiber recruits nobody

froze the fiber forever: out-of-range rows are silently dropped from the
drive slice, zero drive recruits nobody, and no recruitment means no
expansion. Seed 45's NOUN_CORE->NUMBER_PL froze at 180 rows against a source
of 18641 and read own-drive EXACTLY 0.0 on 45/46 trained words -- while
k-WTA kept returning k winners, so nothing raised.

The repair rides the deferred-init path (`_init_deferred_area_srcs`), the
same next-round semantics the empty-block case has always had: after any
projection that names the fiber, its initialised region covers
(src.w, tgt.w). Content-addressed init makes the late fill value-identical
to the fill the recruit path would have done at the same coordinates.

These tests pin the frozen state DIRECTLY (truncate a healthy fiber's rows,
exactly the state the defect left behind) rather than reproducing the
stochastic no-recruitment episode: the invariant is deterministic, the
death spiral that establishes it is not.
"""

from __future__ import annotations

import numpy as np

from neural_assemblies.core.brain import Brain

N, K, P, BETA = 400, 20, 0.05, 0.05


def _brain():
    b = Brain(p=P, seed=3, engine="numpy_sparse", norm_init=True)
    b.add_stimulus("S1", K)
    b.add_stimulus("S2", K)
    b.add_area("A", N, K, BETA)
    b.add_area("B", N, K, BETA)
    return b


def _eng(b):
    return b._engine_for(b.areas["A"])


def _grown_then_truncated():
    """A healthy A->B fiber, then A grows, then the fiber is pinned to the
    frozen state the recruitment-gated growth defect produced."""
    b = _brain()
    for _ in range(3):
        b.project({"S1": ["A"]}, {})
    b.project({}, {"A": ["B"]})

    # Grow A well past the fiber's rows through a DIFFERENT stimulus.
    for _ in range(6):
        b.project({"S2": ["A"]}, {})

    eng = _eng(b)
    conn = eng._area_conns["A"]["B"]
    keep = min(10, conn.weights.shape[0])
    conn.weights = np.array(conn.weights[:keep], copy=True)
    conn._log_rows = keep
    conn._log_cols = int(conn.weights.shape[1])
    # The degree counters tallied the full block; drop them so they rebuild
    # against the truncated state rather than asserting about the old one.
    for attr in ("_deg_rows", "_deg_counts_arr", "_deg_dirty"):
        if hasattr(conn, attr):
            delattr(conn, attr)
    assert eng._areas["A"].w > keep, "test setup: source must outgrow fiber"
    return b, eng, conn


class TestStaleFiberRepair:

    def test_projection_restores_row_coverage(self):
        """One projection through the stale fiber must leave its initialised
        rows covering the source -- recruitment or not. This is the exact
        state seed 45 could never leave."""
        b, eng, conn = _grown_then_truncated()
        b.project({}, {"A": ["B"]})
        src_w = int(eng._areas["A"].w)
        assert int(getattr(conn, "_log_rows", conn.weights.shape[0])) >= src_w
        assert conn.weights.shape[0] >= src_w

    def test_repaired_fiber_delivers_drive_again(self):
        """After the repair round, the source's CURRENT winners must reach
        the target with nonzero mass -- the quantity that read exactly 0.0
        on 45/46 words. Two projections: one to repair, one to deliver."""
        b, eng, conn = _grown_then_truncated()
        b.project({}, {"A": ["B"]})
        b.project({}, {"A": ["B"]})
        conn = eng._area_conns["A"]["B"]
        rows = np.asarray(eng._areas["A"].winners)
        rows = rows[rows < conn.weights.shape[0]]
        assert rows.size == K, "every current source winner has a row"
        assert float(np.asarray(conn.weights)[rows, :].sum()) > 0.0

    def test_healthy_fiber_is_untouched(self):
        """The marking must not fire on a covered fiber: same protocol, no
        truncation, and the block's logical extent is what normal growth
        left -- the repair path adds nothing."""
        b = _brain()
        for _ in range(3):
            b.project({"S1": ["A"]}, {})
        b.project({}, {"A": ["B"]})
        eng = _eng(b)
        conn = eng._area_conns["A"]["B"]
        before = np.array(conn.weights, copy=True)
        with b.read_only():
            b.project({}, {"A": ["B"]})
        conn = eng._area_conns["A"]["B"]
        assert conn.weights.shape == before.shape
        assert np.array_equal(np.asarray(conn.weights), before)
