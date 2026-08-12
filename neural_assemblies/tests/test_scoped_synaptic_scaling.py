"""Scoped synaptic scaling: the flag's set form touches ONLY listed targets.

From task #130 (E1, `research/experiments/scaled_feature_recall.py`).
`synaptic_scaling` has existed since the norm_init work with a documented
attractor-cancellation flaw when applied globally; the scoped form (a
collection of TARGET area names) exists so stimulus-anchored feature areas
can be column-normalized while every attractor-bearing area stays untouched.

WHY AGGREGATE SIGNATURES AND NOT BIT-EQUALITY. Two earlier drafts of this
test failed for reasons worth keeping:
  * exact-setpoint assertions on the listed target fight the ENGINE'S
    ORDERING -- connectome expansion runs after the scaling step, so a
    freshly-expanded column legitimately sits above the setpoint;
  * bit-identical unlisted-target weights across paired brains fail through
    the SHARED RNG CURSOR, not the mechanism: scaling changes the listed
    area's winner set, which changes how many first-winner draws it
    consumes, which shifts every later sample any OTHER area takes.
    Bit-identity is a realization claim; on the sampler engine the honest
    contract is the aggregate one (ensemble, not realization).
Measured totals behind the thresholds (seed 0, 6 rounds, beta=0.2):
OFF B=76.5/C=75.4; True B=42.4/C=23.0; {"B"} B=36.0/C=70.5.

The recall effect (+0.160 +/- 0.111 paired on number, every seed positive,
at a -0.064 cost to tense) lives in the experiment and its note; the
default remains OFF, so no accuracy is pinned here.
"""
from __future__ import annotations

import numpy as np

from neural_assemblies.core.brain import Brain


def _fiber_totals(synaptic_scaling):
    b = Brain(p=0.05, seed=0, engine="numpy_sparse",
              synaptic_scaling=synaptic_scaling, norm_init=False)
    b.add_stimulus("s", 10)
    for name in ("A", "B", "C"):
        b.add_area(name, 500, 10, beta=0.2)
    b.project({"s": ["A"]}, {})
    for _ in range(6):
        b.project({"s": ["A"]}, {"A": ["B", "C"]})
    eng = b._engine
    if synaptic_scaling:
        expected = (synaptic_scaling if isinstance(synaptic_scaling, bool)
                    else frozenset(synaptic_scaling))
        assert eng.synaptic_scaling == expected
    return {t: float(np.asarray(eng._area_conns["A"][t].weights).sum())
            for t in ("B", "C")}


def test_scoped_scaling_suppresses_only_listed_targets():
    off = _fiber_totals(False)
    scoped = _fiber_totals(frozenset({"B"}))

    # The protocol is symmetric, so unscaled fibers grow alike.
    assert 0.8 < off["B"] / off["C"] < 1.25, off
    # Listed target: homeostatically suppressed (measured 0.47x).
    assert scoped["B"] < 0.6 * off["B"], (scoped, off)
    # Unlisted target: keeps growing like the unscaled brain (measured
    # 0.93x; the residual gap is RNG-cursor realization noise, see module
    # docstring).
    assert scoped["C"] > 0.8 * off["C"], (scoped, off)


def test_bool_true_still_scales_every_target():
    off = _fiber_totals(False)
    full = _fiber_totals(True)
    assert full["B"] < 0.6 * off["B"], (full, off)
    assert full["C"] < 0.6 * off["C"], (full, off)


def test_setpoint_uses_fiber_p_not_brain_p():
    """Third member of the fiber-p defect class (see _scale_columns_now).

    A p=0.40 fiber inside a p=0.05 brain: the homeostatic setpoint must be
    rows * 0.40. Priced at the global p it renormalized every TRAINED column
    to 1/8 of its natural mass while untouched columns kept full mass --
    inverting learning. Aggregate-signature assertion (module docstring):
    touched columns must sit at the fiber's own scale, nowhere near the
    brain-p scale.
    """
    b = Brain(p=0.05, seed=0, engine="numpy_sparse",
              synaptic_scaling=True, norm_init=False)
    b.add_stimulus("s", 10)
    b.add_area("A", 500, 10, beta=0.2)
    b.add_area("B", 500, 10, beta=0.2)
    b.add_connectivity("A", "B", 0.4)
    b.project({"s": ["A"]}, {})
    for _ in range(6):
        b.project({"s": ["A"]}, {"A": ["B"]})
    eng = b._engine
    conn = eng._area_conns["A"]["B"]
    w = np.asarray(conn.weights)
    rows = min(int(eng._areas["A"].w), w.shape[0])
    touched = np.asarray(eng._areas["B"].winners, dtype=int)
    touched = touched[touched < w.shape[1]]
    sums = w[:rows, touched].sum(axis=0)
    fiber_setpoint = rows * 0.4
    brain_setpoint = rows * 0.05
    # Touched columns live at the fiber's scale (recruitment transients sit
    # ABOVE the setpoint, never an 8x below it).
    assert float(sums.min()) > 2.0 * brain_setpoint, (
        sums, fiber_setpoint, brain_setpoint)
    assert float(np.median(sums)) > 0.5 * fiber_setpoint, (
        sums, fiber_setpoint)


def test_scaled_fiber_is_stored_column_major():
    """Perf guard: a scaled area->area fiber is column-major (F-order).

    `_scale_columns_now`'s whole cost is a per-column gather/scatter of `k`
    winner columns of a large dense block; in C (row-major) order a column's
    rows are strided by the row width, so it is cache-miss-bound (measured
    34 ms/call on a 16000x4200 organ_p=0.5 fiber, ~84% of the homeostatic
    training arm under load). Storing the fiber F-order makes each column
    contiguous, byte-identically (asfortranarray preserves logical [i,j], so
    drive-reads and plasticity are unchanged) -- proven end-to-end: two builds
    that differ ONLY in the fiber's memory order produced identical census and
    connectome crc.

    This asserts the MECHANISM stays active. Like the CSR cache-the-NO
    scan-count guard, it is behavioural, not a correctness claim: if the fiber
    silently reverted to C-order the speedup would vanish with no other signal.
    Growth reallocates C-order and `_scale_columns_now` reconverts on the next
    touch, so the guard reads the fiber AFTER a post-growth scaling step.
    """
    b = Brain(p=0.05, seed=0, engine="numpy_sparse",
              synaptic_scaling=True, norm_init=False)
    b.add_stimulus("s", 10)
    b.add_area("A", 500, 10, beta=0.2)
    b.add_area("B", 500, 10, beta=0.2)
    b.add_connectivity("A", "B", 0.4)
    b.project({"s": ["A"]}, {})
    for _ in range(6):
        b.project({"s": ["A"]}, {"A": ["B"]})
    w = np.asarray(b._engine._area_conns["A"]["B"].weights)
    assert w.ndim == 2 and w.flags["F_CONTIGUOUS"], (w.shape, str(w.flags))
