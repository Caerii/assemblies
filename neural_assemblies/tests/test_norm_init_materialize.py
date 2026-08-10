"""norm_init must apply to MATERIALIZED fibers. It silently does not.

FOUND 2026-08-10 via the S5 norm_init intervention, which BACKFIRED: turning
norm_init on took the soft-transition count from 27 to 5861 across 40 organs
(~97% of all transitions), while the arithmetic says correct per-column
normalization moves relative drives by ~1% at these degrees. The diagnostic
below explains the flood:

  * On the `materialize_area` path, norm_init is a SILENT NO-OP: column-mass
    CV is bit-identical between norm_init=True and False (ratio 1.000) --
    per-fiber and global-p fibers alike, both materialization orders.
  * On the RECRUITMENT path (`_expand_connectomes`), `_norm_scale` IS applied
    (it shows in the training profile).

So a brain that mixes materialized and recruited rows in one fiber under
norm_init=True gets interleaved scaled/unscaled rows -- O(1) column-drive
distortions, which is the measured flood. Inconsistent application is worse
than absent: see [[self-fibers-excluded-from-deferred-init]] ("fixing any one
alone is worse than none") and [[explicit-source-skips-norm-init]], the same
defect class on the explicit-dense path, already fixed once.

BLAST RADIUS. `norm_init` defaults to TRUE, so every production brain that
calls `materialize_area` and then trains sits on the mixed regime.

The fingerprint harness (`test_connectome_representation_fingerprint`) did
not catch this because its materialized configuration runs norm_init=False
only; extend it when fixing.

The invariant test is marked xfail (not strict): it documents the defect while
the suite stays green, and flips to XPASS the moment the fix lands -- at which
point remove the marker and extend the fingerprint.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain


def _column_mass_cv(norm_init: bool, per_fiber: bool) -> float:
    random.seed(7)
    np.random.seed(7)
    b = Brain(p=0.05 if per_fiber else 0.4, save_winners=True, seed=7,
              engine="numpy_sparse", norm_init=norm_init)
    b.add_area("SRC", 4000, 70, 0.1)
    b.add_area("TGT", 1400, 70, 0.1)
    if per_fiber:
        b.add_connectivity("SRC", "TGT", 0.4)
    b.materialize_area("TGT")
    b.materialize_area("SRC")
    conn = b._engine._area_conns["SRC"]["TGT"]
    w = conn.weights
    dense = np.asarray(w.todense() if hasattr(w, "todense") else w,
                       dtype=np.float64)
    mass = dense[:4000, :1400].sum(axis=0)
    return float(mass.std() / mass.mean())


@pytest.mark.parametrize("per_fiber", [True, False])
@pytest.mark.xfail(
    reason="norm_init is a silent no-op on the materialize_area path: "
           "column-mass CV is bit-identical with it on or off (ratio 1.000). "
           "Measured consequence: 27 -> 5861 soft transitions when a trained "
           "organ mixes materialized (unscaled) and recruited (scaled) rows. "
           "seq_s5_norm_init_intervention.py, commit 0b660b6.",
    strict=False,
)
def test_norm_init_equalizes_materialized_columns(per_fiber):
    """The invariant normalization exists to enforce: near-equal column mass."""
    cv_off = _column_mass_cv(False, per_fiber)
    cv_on = _column_mass_cv(True, per_fiber)
    assert cv_on < 0.5 * cv_off, (
        f"norm_init left materialized column dispersion unchanged "
        f"(CV {cv_on:.4f} vs {cv_off:.4f} without)")


def test_the_no_op_is_exact_not_approximate():
    """Pins the DIAGNOSIS while the defect stands: bit-identical, not merely
    similar. If this starts failing while the xfail above still fails, the
    behaviour changed without becoming correct -- flag, do not celebrate.
    When the fix lands this test is DELETED along with the xfail marker."""
    assert _column_mass_cv(True, True) == _column_mass_cv(False, True)
