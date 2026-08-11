"""VirtualWeights must be BIT-IDENTICAL to the dense operations it replaces.

The reference here is not a formula -- it is a float32 ndarray driven through
the same operation sequence `_sparse.py` performs: content-addressed block
fills, explicit first-winner override writes (`w[chosen, col] = 1.0`), and
Hebbian events as an in-place ``*= (1 + beta)`` on the co-firing submatrix
followed by a clip. Ulps count: the fingerprint digests at round(9), finer
than float32 resolution, so a float64 closed-form power would differ. The
chain replay must reproduce the dense result EXACTLY, not approximately.
"""

from __future__ import annotations

import unittest

import numpy as np

from neural_assemblies.core.numpy_engine._seeding import hash_area_weights
from neural_assemblies.core.numpy_engine._virtual_weights import VirtualWeights

ROWS, COLS, SEED, P, BETA = 300, 200, 12345, 0.4, 0.1
LO, HI = 0.0, 20.0


def _dense_reference(ops):
    """Drive a real ndarray through the engine's exact operations."""
    w = hash_area_weights(0, ROWS, 0, COLS, SEED, P)
    for kind, a, b in ops:
        if kind == "override":
            w[a, b] = 1.0
        else:                                    # hebbian event
            ix = np.ix_(a, b)
            w[ix] *= (1 + BETA)                  # python float, as the engine
            sub = w[ix]
            np.clip(sub, LO, HI, out=sub)
            w[ix] = sub
    return w


def _virtual(ops):
    vw = VirtualWeights(ROWS, COLS, SEED, P, BETA, LO, HI)
    for kind, a, b in ops:
        if kind == "override":
            vw.override(a, b)
        else:
            vw.bump(a, b, BETA)
    return vw


def _script(rng, n_events=40):
    """A plausible training history: overrides on fresh columns interleaved
    with repeated Hebbian events on recurring co-firing blocks."""
    ops, used_cols = [], set()
    blocks = [(rng.choice(ROWS, 30, replace=False),
               rng.choice(COLS, 30, replace=False)) for _ in range(6)]
    fresh = [c for c in range(COLS) if not any(c in b[1] for b in blocks)]
    for i in range(n_events):
        if i % 7 == 3 and fresh:
            col = fresh.pop()
            used_cols.add(col)
            ops.append(("override", rng.choice(ROWS, 12, replace=False), col))
        elif i % 7 == 5:
            # the engine's real recruitment order: potentiate a column, then
            # override cells in it -- assignment must clobber the history
            r, c = blocks[i % len(blocks)]
            ops.append(("hebb", r, c))
            ops.append(("override", r[:8], int(c[0])))
        else:
            r, c = blocks[i % len(blocks)]
            ops.append(("hebb", r, c))
    return ops


class TestBitIdentity(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.ops = _script(self.rng)
        self.dense = _dense_reference(self.ops)
        self.vw = _virtual(self.ops)

    def test_todense_is_bit_identical(self):
        got = self.vw.todense()
        self.assertEqual(got.dtype, self.dense.dtype)
        exact = np.array_equal(got, self.dense)
        if not exact:
            bad = np.argwhere(got != self.dense)
            r, c = bad[0]
            self.fail(f"{len(bad)} cells differ; first ({r},{c}): "
                      f"virtual {got[r, c]!r} vs dense {self.dense[r, c]!r}")

    def test_row_sum_matches_dense_row_sum(self):
        rows = self.rng.choice(ROWS, 40, replace=False)
        want = self.dense[rows, :150].sum(axis=0, dtype=np.float64)
        got = self.vw.row_sum(rows, 150).astype(np.float64)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)

    def test_cells_match(self):
        for r, c in self.rng.integers(0, [ROWS, COLS], size=(200, 2)):
            self.assertEqual(self.vw.cell(int(r), int(c)),
                             self.dense[r, c], f"cell ({r},{c})")

    def test_column_nnz_matches(self):
        want = (self.dense != 0).sum(axis=0)
        got = self.vw.column_nnz()
        np.testing.assert_array_equal(got, want)

    def test_storage_is_deviations_not_cells(self):
        """Storage must scale with DEVIATIONS, not with the block.

        Not asserted as "smaller than dense" -- this toy script bumps 9% of
        all cells, far denser co-firing than any organ, and the first version
        of this test failed on exactly that toy geometry. The real-scale
        claim (2.69 GB -> ~20 MB on the S5 organ) is a measurement in the
        integration work, not a toy assertion.
        """
        self.assertEqual(self.vw.nbytes,
                         24 * (self.vw.deviations + self.vw.overrides))
        empty = VirtualWeights(10**6, 10**6, 1, 0.4, 0.1, LO, HI)
        self.assertEqual(empty.nbytes, 0,
                         "an unpotentiated fiber must cost nothing")


class TestGuards(unittest.TestCase):

    def test_beta_mismatch_refused(self):
        vw = VirtualWeights(50, 50, 1, 0.4, 0.1, LO, HI)
        vw.bump([1, 2], [3, 4], 0.1)
        with self.assertRaises(ValueError):
            vw.bump([1, 2], [3, 4], 0.2)

    def test_override_clobbers_potentiation_like_dense_assignment(self):
        """The engine's real order: plasticity precedes expansion within a
        recruitment round, so an override lands on a potentiated cell and
        dense assignment ERASES the history. The first version refused this
        order and the fingerprint run refuted it immediately."""
        vw = VirtualWeights(50, 50, 1, 0.9, 0.1, LO, HI)   # p high: base != 0
        vw.bump([1], [3], 0.1)
        vw.override([1], 3)
        self.assertEqual(vw.cell(1, 3), 1.0)
        vw.bump([1], [3], 0.1)
        self.assertEqual(vw.cell(1, 3), np.float32(np.float32(1.0)
                                                   * np.float32(1.1)))

    def test_shrink_refused(self):
        vw = VirtualWeights(50, 50, 1, 0.4, 0.1, LO, HI)
        with self.assertRaises(ValueError):
            vw.resize(40, 50)

    def test_supports_gate(self):
        self.assertTrue(VirtualWeights.supports(
            synaptic_scaling=False, content_init=True))
        self.assertFalse(VirtualWeights.supports(
            synaptic_scaling=True, content_init=True))
        self.assertFalse(VirtualWeights.supports(
            synaptic_scaling=False, content_init=False))


if __name__ == "__main__":
    unittest.main()
