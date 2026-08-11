"""The virtual representation must be INVISIBLE: same winners, same weights.

The fingerprint golden digests the dense path's PHYSICAL buffer, padding
included, so a padding-free representation can never match it field-for-field
even when every computed value is identical. This test makes the comparison
the golden cannot: run each configuration twice IN-PROCESS -- gate off, gate
on (the gate reads the environment per call) -- and demand bit-identical
winners every round and bit-identical weights over the LOGICAL region.

This is the acceptance test the design note promised. It found its first bug
before it existed as a file: `VirtualWeights` without a `size` property made
`_norm_scale`'s ``getattr(w, 'size', 0) == 0`` guard treat every virtual
fiber as empty -- normalization silently OFF, winners diverging at round 1 on
exactly the *_norm configs. A missing attribute was a kill-switch two layers
away; `getattr`-with-default is how it stayed silent.
"""

from __future__ import annotations

import os
import random
import unittest

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine._virtual_weights import VirtualWeights
from neural_assemblies.diagnostics import read_assembly

CONFIGS = [
    ("p05", 0.05, False, False),
    ("p05_norm", 0.05, True, False),
    ("p40", 0.40, False, False),
    ("p40_norm", 0.40, True, False),
    ("p40_materialized", 0.40, False, True),
]


def _run(gate: str, norm: bool, p_fiber: float, materialize: bool):
    os.environ["ASSEMBLIES_VIRTUAL_WEIGHTS"] = gate
    try:
        random.seed(7)
        np.random.seed(7)
        b = Brain(p=0.05, save_winners=True, seed=7, engine="numpy_sparse",
                  norm_init=norm)
        b.add_area("SRC", 900, 30, 0.1)
        b.add_area("TGT", 700, 30, 0.1)
        b.add_connectivity("SRC", "TGT", p_fiber)
        b.add_stimulus("s0", 30)
        b.add_stimulus("s1", 30)
        if materialize:
            b.materialize_area("TGT")
        winners = []
        for i in range(6):
            b.project({"s0" if i % 2 == 0 else "s1": ["SRC"]}, {})
            b.project({}, {"SRC": ["TGT"]})
            winners.append(read_assembly(b, "TGT").tolist())
        conn = b._engine._area_conns["SRC"]["TGT"]
        w = conn.weights
        rows = int(getattr(conn, "_log_rows", 0) or w.shape[0])
        cols = int(getattr(conn, "_log_cols", 0) or w.shape[1])
        dense = np.asarray(w.todense() if hasattr(w, "todense") else w,
                           dtype=np.float32)[:rows, :cols]
        return winners, dense, isinstance(w, VirtualWeights)
    finally:
        os.environ.pop("ASSEMBLIES_VIRTUAL_WEIGHTS", None)


class TestGateEquivalence(unittest.TestCase):

    def test_all_configs_bit_identical(self):
        for name, p_fiber, norm, mat in CONFIGS:
            with self.subTest(config=name):
                w_off, d_off, virt_off = _run("0", norm, p_fiber, mat)
                w_on, d_on, virt_on = _run("1", norm, p_fiber, mat)
                self.assertFalse(virt_off, f"{name}: gate off built virtual")
                self.assertTrue(virt_on,
                                f"{name}: gate on did NOT build a virtual "
                                f"fiber -- the test is comparing dense to "
                                f"dense and proving nothing")
                self.assertEqual(w_off, w_on, f"{name}: winners diverged")
                self.assertEqual(d_off.shape[0], d_on.shape[0], name)
                min_c = min(d_off.shape[1], d_on.shape[1])
                self.assertTrue(
                    np.array_equal(d_off[:, :min_c], d_on[:, :min_c]),
                    f"{name}: logical-region weights differ")


if __name__ == "__main__":
    unittest.main()
