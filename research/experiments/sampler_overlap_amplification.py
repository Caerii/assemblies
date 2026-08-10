"""Does the sparse candidate sampler AMPLIFY input overlap? [[SEQ-EXACT-RECOVERY]]

WHY THIS EXISTS. A3's H4 asks whether an induced state collapses. At toy scale
it collapses hard -- two arcs overlapping 0.35 write states overlapping 0.97 --
with plasticity off on the fiber and no training of any kind, so nothing
Hebbian is involved. That points at selection rather than learning, and the
sparse engine's candidate sampler is the one part of selection that INVENTS a
number: for neurons that have never fired it draws drive from a distribution
instead of computing it. If the sampler amplifies overlap, H4 measures the
engine and not the organ.

[[kwta-amplifies-input-overlap]] already records that k-WTA amplifies rather
than contracts, against the COLT22 bound. That is a property of the RULE and
would appear on any engine. The question here is narrower and is about the
IMPLEMENTATION: does the sampler amplify MORE than the exact drive does?

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE SEEDED RUN. An unseeded probe
at one seed suggested P2; that probe is what motivated this script and its
numbers are void.

  P1  materialised sparse == exact.  `materialize_area` bypasses the sampler
      ([[sampler-is-the-whole-discrepancy]]), so these must be
      indistinguishable at every input overlap. If they differ, the difference
      is NOT the sampler and this whole framing is wrong.
  P2  unmaterialised sparse > exact at input overlap >= 0.5, judged by the
      lower bound of the PAIRED per-seed difference.
  P3  all three arms at chance when the inputs are disjoint. A sampler that
      merged disjoint inputs would be a much larger defect than amplification,
      and the earlier probe says it does not.

INTERPRETATION, STATED NOW. P2 holding means every sequence result measured on
numpy_sparse carries an overlap inflation that grows with the overlap already
present -- i.e. exactly the regime a collapsing state enters. It would NOT
invalidate a comparison between two arms on the same engine (#14's CONTEXT vs
A3), because both sides inflate; it WOULD make any absolute overlap, including
#14's 0.7566 and A3's H4, an upper bound rather than a measurement.

P2 failing sends the collapse back to the organ, where it is a real result.

No plasticity anywhere: assemblies are CONSTRUCTED at an exact overlap and
projected once under `frozen()`. Construction rather than training is what
makes the input overlap an independent variable instead of an outcome.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import (
    assembly_overlap, ensemble, paired_delta, read_assembly,
)

N_SRC, N_TGT, K, P_FIBER, P_AMBIENT = 2000, 2000, 40, 0.4, 0.05
SEEDS = list(range(42, 52))
INPUT_OVERLAPS = [0.0, 0.25, 0.5, 0.75, 1.0]
ARMS = ("sparse", "sparse_materialized", "exact")


def _pair_at_overlap(rng, target_ov):
    """Two k-assemblies sharing exactly ``round(target_ov * k)`` neurons."""
    base = rng.choice(N_SRC, K, replace=False)
    shared = int(round(target_ov * K))
    rest = np.setdiff1d(np.arange(N_SRC), base)
    other = np.concatenate([base[:shared],
                            rng.choice(rest, K - shared, replace=False)])
    return base.astype(np.uint32), other.astype(np.uint32)


def run_cell(seed, arm, target_ov):
    random.seed(seed)
    np.random.seed(seed)
    engine = "numpy_exact" if arm == "exact" else "numpy_sparse"
    b = Brain(p=P_AMBIENT, save_winners=True, seed=seed, engine=engine)
    b.add_area("SRC", N_SRC, K, 0.0)
    b.add_area("TGT", N_TGT, K, 0.0)
    if engine == "numpy_sparse":
        b.add_connectivity("SRC", "TGT", P_FIBER)
    b.materialize_area("SRC")
    if arm == "sparse_materialized":
        b.materialize_area("TGT")

    rng = np.random.default_rng(seed)
    winners = _pair_at_overlap(rng, target_ov)
    outs = []
    for w in winners:
        b.inhibit_areas(["SRC", "TGT"])
        b.areas["SRC"].winners = w
        b._engine.set_winners("SRC", w)
        with b.frozen():
            b.project({}, {"SRC": ["TGT"]})
        outs.append(read_assembly(b, "TGT"))
    return assembly_overlap(outs[0], outs[1])


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== does the sparse sampler amplify input overlap? ===")
    print(f"    n_src={N_SRC} n_tgt={N_TGT} k={K} p_fiber={P_FIBER} "
          f"chance={K / N_TGT:.4f}")
    print(f"    seeds {seeds[0]}..{seeds[-1]}, no plasticity anywhere\n")

    cells = {}
    for ov in INPUT_OVERLAPS:
        row = {}
        for arm in ARMS:
            row[arm] = ensemble(lambda s, a=arm, o=ov: run_cell(s, a, o),
                                seeds, label=f"{arm}@{ov}")
        cells[ov] = row
        print(f"  input overlap {ov:.2f}")
        for arm in ARMS:
            print(f"      {arm:22s} {row[arm].mean:.4f} +/- {row[arm].ci:.4f}")
        d_mat = paired_delta(row["sparse_materialized"], row["exact"])
        d_spa = paired_delta(row["sparse"], row["exact"])
        print(f"      materialized - exact  {d_mat.mean:+.4f} "
              f"[{d_mat.low:+.4f}, {d_mat.high:+.4f}]")
        print(f"      sparse      - exact   {d_spa.mean:+.4f} "
              f"[{d_spa.low:+.4f}, {d_spa.high:+.4f}]", flush=True)
        row["d_materialized"] = d_mat
        row["d_sparse"] = d_spa

    print("\n=== BARS ===")
    p1 = all(cells[ov]["d_materialized"].indistinguishable_from(0.0)
             for ov in INPUT_OVERLAPS)
    p2 = all(cells[ov]["d_sparse"].low > 0.0
             for ov in INPUT_OVERLAPS if ov >= 0.5)
    chance = K / N_TGT
    p3 = all(cells[0.0][arm].high < 10 * chance for arm in ARMS)
    for name, ok in (("P1 materialized == exact", p1),
                     ("P2 sparse > exact at overlap >= 0.5", p2),
                     ("P3 disjoint inputs stay near chance", p3)):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    out = {
        "seeds": seeds, "k": K, "n_tgt": N_TGT, "p_fiber": P_FIBER,
        "chance": chance,
        "cells": {str(ov): {arm: {"mean": cells[ov][arm].mean,
                                  "ci": cells[ov][arm].ci,
                                  "values": list(cells[ov][arm].values)}
                            for arm in ARMS}
                  for ov in INPUT_OVERLAPS},
        "deltas": {str(ov): {"materialized_minus_exact":
                             [cells[ov]["d_materialized"].mean,
                              cells[ov]["d_materialized"].low,
                              cells[ov]["d_materialized"].high],
                             "sparse_minus_exact":
                             [cells[ov]["d_sparse"].mean,
                              cells[ov]["d_sparse"].low,
                              cells[ov]["d_sparse"].high]}
                   for ov in INPUT_OVERLAPS},
        "verdicts": {"P1": p1, "P2": p2, "P3": p3},
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sampler_overlap_amplification_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
