"""A byte-identity fingerprint of the connectome, for changing how it is STORED.

WHY THIS EXISTS, BEFORE THE CHANGE IT GUARDS. The sparse engine stores a
materialised fiber DENSE at `n_src x n_tgt`. That is the wrong representation:
`hash_area_weights` is a pure function of (row, col), so BASE weights need no
storage at all; multiplicative Hebbian never creates a nonzero, so the sparsity
pattern is a pure function of the hash too; and only the pairs that actually
co-fired deviate from base. Storing base + sparse deviations is ~100x smaller
than dense at the densities the sequence organs use, where CSR saves almost
nothing (p=0.4 -> 134M nonzeros against 336M cells).

That change must be INVISIBLE. A representation is not a model: if swapping it
moves a single winner, the new one is wrong, or the old one was. This file
pins enough of the current engine to prove it either way, and it is written
FIRST so the golden is captured from the implementation being replaced rather
than from its replacement.

WHAT IS FINGERPRINTED, AND WHY EACH. A change of representation could break any
of these independently, and only the first is visible in an experiment's
output:

  winners       the observable. Recorded per projection, in NEURON IDS, so a
                compact-index renumbering is not mistaken for a behaviour
                change ([[two-index-spaces-compact-vs-neuron-id]]).
  weights       sampled cell VALUES. Winners can coincide while the underlying
                weights differ -- k-WTA discards magnitude, so a wrong
                connectome hides until the ordering happens to flip.
  degree        per-column nonzero counts. `norm_init` divides by these, and a
                virtual representation must derive them analytically instead of
                scanning; if it derives them wrongly, the scale is wrong and
                the winners follow much later.
  scale         the per-column normalisation vector itself.

CONFIGURATIONS. Density matters (0.05 is where CSR pays, 0.4 is where the
organs run), `norm_init` matters (it is the only consumer of `degree`), and
whether the target was pre-materialised matters most of all -- that is the axis
the change is about, and it is also a MEASURED behaviour difference, not a
neutral one ([[sampler-is-the-whole-discrepancy]]).

REGENERATING. `python -m neural_assemblies.tests.test_connectome_representation_fingerprint`
rewrites the golden. Doing that is a claim that the engine's arithmetic SHOULD
have changed, and the diff belongs in the commit message.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import unittest

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import read_assembly

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "connectome_fingerprint.json")

CONFIGS = [
    # (label, p_fiber, norm_init, materialize_target)
    ("sparse_p05", 0.05, False, False),
    ("sparse_p05_norm", 0.05, True, False),
    ("organ_p40", 0.40, False, False),
    ("organ_p40_norm", 0.40, True, False),
    ("organ_p40_materialized", 0.40, False, True),
]

N_SRC, N_TGT, K, BETA, ROUNDS = 900, 700, 30, 0.10, 6


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a))
        h.update(str(arr.shape).encode())
        h.update(arr.astype(np.float64).round(9).tobytes())
    return h.hexdigest()[:32]


def fingerprint(p_fiber: float, norm_init: bool, materialize: bool) -> dict:
    """Drive a fixed protocol and digest winners, weights, degree and scale."""
    random.seed(7)
    np.random.seed(7)
    b = Brain(p=0.05, save_winners=True, seed=7, engine="numpy_sparse",
              norm_init=norm_init)
    b.add_area("SRC", N_SRC, K, BETA)
    b.add_area("TGT", N_TGT, K, BETA)
    b.add_connectivity("SRC", "TGT", p_fiber)
    b.add_stimulus("s0", K)
    b.add_stimulus("s1", K)
    if materialize:
        b.materialize_area("TGT")

    winners = []
    for i in range(ROUNDS):
        stim = "s0" if i % 2 == 0 else "s1"
        b.project({stim: ["SRC"]}, {})
        b.project({}, {"SRC": ["TGT"]})
        winners.append(read_assembly(b, "TGT"))

    conn = b._engine._area_conns["SRC"]["TGT"]
    w = conn.weights
    dense = np.asarray(w.todense() if hasattr(w, "todense") else w,
                       dtype=np.float64)
    # THE WHOLE BLOCK, not a lattice. The first draft digested a 23x29 sample
    # to keep the golden readable -- but the golden stores a HASH, so sampling
    # costs nothing to store and buys nothing except blind spots. Measured:
    # against a single perturbed cell the sampled version was detected only by
    # `nnz` and `degree`; `weights` missed it because the cell was not on the
    # lattice. A representation swap fails systematically rather than one cell
    # at a time, but a guard should not depend on that being true.
    sampled = dense
    degree = (dense != 0).sum(axis=0)
    scale = np.asarray(getattr(conn, "_norm_scale_vec", np.zeros(1)),
                       dtype=np.float64)

    return {
        "winners": _digest(*winners),
        "weights": _digest(sampled),
        "degree": _digest(degree),
        "scale": _digest(scale),
        "nnz": int((dense != 0).sum()),
        "shape": list(dense.shape),
    }


def collect() -> dict:
    return {label: fingerprint(p, norm, mat)
            for label, p, norm, mat in CONFIGS}


class TestConnectomeFingerprint(unittest.TestCase):

    def test_fingerprint_matches_golden(self):
        with open(GOLDEN, encoding="utf-8") as fh:
            golden = json.load(fh)
        got = collect()
        for label in golden:
            self.assertIn(label, got, f"config {label} disappeared")
            for field, want in golden[label].items():
                self.assertEqual(
                    got[label][field], want,
                    f"{label}.{field} changed: the connectome's ARITHMETIC "
                    f"moved, not just its storage. If that is intended, "
                    f"regenerate the golden and justify the diff.")

    def test_materialization_is_not_neutral(self):
        """Guards the guard. If pre-materialising ever stopped changing the
        answer, this fingerprint would be blind to the axis it exists for --
        and the difference is a MEASURED one, not a theoretical worry."""
        plain = fingerprint(0.40, False, False)
        mat = fingerprint(0.40, False, True)
        self.assertNotEqual(plain["winners"], mat["winners"])

    def test_density_reaches_the_regime_the_organs_use(self):
        """A fingerprint taken only at p=0.05 would not cover the fibers that
        motivated the change."""
        f = fingerprint(0.40, False, False)
        occupancy = f["nnz"] / (f["shape"][0] * f["shape"][1])
        self.assertGreater(occupancy, 0.15)


if __name__ == "__main__":
    with open(GOLDEN, "w", encoding="utf-8") as fh:
        json.dump(collect(), fh, indent=2, sort_keys=True)
    print(f"wrote {GOLDEN}")
