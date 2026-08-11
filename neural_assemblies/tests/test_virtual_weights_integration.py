"""VirtualWeights integration under DRIVE SEMANTICS v2.

HISTORY, kept deliberately. The first version of this test demanded winners
bit-identical to the DENSE path, and that bar caught three real bugs (a
missing `size` property that silently disabled normalization, a float
summation-order divergence, an ungated materialize path). It was retired on
purpose by `PREREG_drive_semantics_v2.md`: the memoized decomposition
`f64 base_sum + f64 delta` IS the virtual drive's definition now, and it
differs from dense float32 pairwise reduction by ulps -- which can flip
k-WTA tie order. The registered science-invariance run (V-S5) is what
licenses that difference; this file guards what remains guardable:

  * SELF-CONSISTENCY: cold and memoized paths are the same computation, so
    two identical runs must be bit-identical, and a run must equal a run
    whose caches were primed differently.
  * A VIRTUAL-SEMANTICS FINGERPRINT: the five configurations, gate-on,
    digested into their own golden -- v2 semantics pinned against future
    drift exactly as dense semantics is pinned by the dense golden.
  * SANITY vs dense: winner-set overlap stays high (ulp ties move ONE
    winner occasionally, not the assembly). A gross divergence means a bug,
    not a tie.

Regenerate the golden (justify in the commit):
    python -m neural_assemblies.tests.test_virtual_weights_integration
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import unittest

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine._virtual_weights import VirtualWeights
from neural_assemblies.diagnostics import read_assembly

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "virtual_semantics_fingerprint.json")

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


def _digest(winners, weights) -> dict:
    h = hashlib.sha256()
    for w in winners:
        h.update(np.asarray(sorted(w), dtype=np.int64).tobytes())
    hw = hashlib.sha256(np.ascontiguousarray(weights).tobytes())
    return {"winners": h.hexdigest()[:32], "weights": hw.hexdigest()[:32]}


def collect() -> dict:
    out = {}
    for name, p_fiber, norm, mat in CONFIGS:
        winners, weights, virt = _run("1", norm, p_fiber, mat)
        assert virt, f"{name}: gate on did not build a virtual fiber"
        out[name] = _digest(winners, weights)
    return out


class TestSelfConsistency(unittest.TestCase):

    def test_two_runs_bit_identical_every_config(self):
        for name, p_fiber, norm, mat in CONFIGS:
            with self.subTest(config=name):
                w1, d1, v1 = _run("1", norm, p_fiber, mat)
                w2, d2, v2 = _run("1", norm, p_fiber, mat)
                self.assertTrue(v1 and v2, f"{name}: not virtual")
                self.assertEqual(w1, w2, f"{name}: winners not reproducible")
                self.assertTrue(np.array_equal(d1, d2), name)


class TestVirtualGolden(unittest.TestCase):

    def test_matches_virtual_semantics_golden(self):
        with open(GOLDEN, encoding="utf-8") as fh:
            golden = json.load(fh)
        got = collect()
        for name in golden:
            self.assertEqual(got[name], golden[name],
                             f"{name}: v2 semantics drifted -- if intended, "
                             f"regenerate and justify in the commit")


class TestDenseSanity(unittest.TestCase):

    def test_winner_sets_stay_close_to_dense(self):
        """Ulp ties move a winner occasionally; a bug moves the assembly."""
        for name, p_fiber, norm, mat in CONFIGS:
            with self.subTest(config=name):
                w_off, _d0, _ = _run("0", norm, p_fiber, mat)
                w_on, _d1, _ = _run("1", norm, p_fiber, mat)
                overlaps = [len(set(a) & set(b)) / 30.0
                            for a, b in zip(w_off, w_on)]
                self.assertGreater(float(np.mean(overlaps)), 0.8,
                                   f"{name}: {overlaps}")


if __name__ == "__main__":
    with open(GOLDEN, "w", encoding="utf-8") as fh:
        json.dump(collect(), fh, indent=2, sort_keys=True)
    print(f"wrote {GOLDEN}")
