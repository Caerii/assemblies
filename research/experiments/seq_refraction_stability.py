"""Refraction as the stability/conjunctivity tradeoff.

Implements `research/notes/memory/PREREG_refraction_stability.md` (+ Amendment 1).

`_homeostasis.refraction_increment` returns `(net_drive + current_bias) *
strength` and the bias is never cleared, so a persistently winning arc neuron
accumulates penalty GEOMETRICALLY (ratio 1+strength) and is burned out of its
own assembly. That one mechanism predicts both measured facts: assemblies
drift, and the arc saturates.

Bars R1-R4 are in the note. BOTH readouts every cell, because either alone
misleads: a machine that is perfectly stable because it stopped
discriminating would pass a stability bar and be worthless.

Amendment 1: stability is read from `saved_winners`, which training already
records (the arc is the target of exactly one of the two projections per
transition), so no probe pass is needed. One cell computes the probed form
too, for validation.

Amendment 3 (pre-data, aggregation only): every bar is judged on a CONFIDENCE
BOUND rather than a point estimate. R1 tests `Ensemble.beats`; the four
ordering bars R2/R3/R4/R5 test `paired_delta`, the per-seed difference, which
is what an A/B actually asks -- comparing two independent intervals is a
different and weaker test, and comparing a difference against a single arm's sd
understates the spread by ~sqrt(2). WHAT IS MEASURED IS UNCHANGED: the same
cells, the same statistic per cell, the same thresholds, the same directions.
Only the aggregation across seeds changed, and it changed BEFORE any data
existed. See the note's Amendment 3.

The one surviving `np.mean` is a mean over TRANSITIONS inside a single
presentation of a single seed -- it forms that seed's value, and putting a
confidence interval over transitions within one brain would be a different and
wrong claim.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.diagnostics import ensemble_from_values, paired_delta

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SEEDS = [42, 43, 44]
T = 16
LONGEST = 100
#: (refracted_strength, constant_mode). strength 0 is mode-independent.
CONFIGS = [(0.0, False), (0.05, False), (0.05, True),
           (0.10, False), (0.10, True)]
#: Amendment 2: synaptic_scaling is a FACTOR -- the decay this study explains
#: was measured with it ON, and the original design registered it OFF.
SCALINGS = [False, True]


def _stability_from_saved(fsm, n_trans, presentations):
    """Identical-assembly fraction per presentation, from recorded winners.

    `saved_winners[t*n_trans + i]` is transition i's arc assembly at
    presentation t, in NEURON IDs (the stable space).
    """
    sw = fsm.brain.areas[fsm.arc_area].saved_winners
    if len(sw) < n_trans * presentations:
        raise RuntimeError(f"expected {n_trans*presentations} saved winner "
                           f"sets, got {len(sw)}")
    sets = [frozenset(int(x) for x in np.asarray(w)) for w in sw]
    out = []
    for t in range(1, presentations):
        cur = sets[t * n_trans:(t + 1) * n_trans]
        prev = sets[(t - 1) * n_trans:t * n_trans]
        same = sum(1 for a, b in zip(cur, prev) if a == b)
        ov = float(np.mean([len(a & b) / max(len(a), 1)
                            for a, b in zip(cur, prev)]))
        out.append({"pres": t + 1, "identical": same / n_trans, "overlap": ov})
    return out


def worker(strength, constant, scaling, seed):
    # Read at call time by `constant_refraction_enabled()`, so setting it here
    # is enough and must be set on EVERY call (workers are reused).
    os.environ["ASSEMBLIES_CONSTANT_REFRACTION"] = "1" if constant else "0"
    random.seed(seed)
    np.random.seed(seed)
    from seq_s5_word_problem import build
    from neural_assemblies.programs.word_problems import (
        true_trajectory, word_problem_fsm,
    )
    group, fsm, symbols = build(
        "Z60", seed, "trained", norm_init=False, synaptic_scaling=scaling,
        organ_p=0.5, presentations=T, w_max=None,
        refracted_strength=strength)
    _s, _y, transitions = word_problem_fsm(group)
    curve = _stability_from_saved(fsm, len(transitions), T)

    rng = random.Random(seed + 4242)
    word = [rng.choice(symbols) for _ in range(LONGEST)]
    truth = true_trajectory(group, word)
    labels = fsm.run(word, group.label(group.identity))
    acc = sum(a == t for a, t in zip(labels, truth)) / len(truth)
    return {"strength": strength, "constant": constant,
            "scaling": scaling, "seed": seed,
            "curve": curve, "terminal_identical": curve[-1]["identical"],
            "terminal_overlap": curve[-1]["overlap"], "acc": acc}


def main():
    from _parallel import run_cells
    cells = [(s, c, sc, sd) for (s, c) in CONFIGS
             for sc in SCALINGS for sd in SEEDS]
    print(f"=== refraction stability/conjunctivity tradeoff ===")
    print(f"    Z60 organ_p=0.5 T={T} w_max=None, seeds {SEEDS}\n")
    res = run_cells(worker, cells, max_workers=min(len(cells), 14))

    print(f"\n  {'strength':>8} {'mode':>9}  {'terminal identical':>22} "
          f"{'terminal overlap':>18} {'acc(L=100)':>18}")
    summary = {}
    for (s, c) in CONFIGS:
        ids = [res[(s, c, True, sd)]["terminal_identical"] for sd in SEEDS]
        ovs = [res[(s, c, True, sd)]["terminal_overlap"] for sd in SEEDS]
        accs = [res[(s, c, True, sd)]["acc"] for sd in SEEDS]
        tag = f"{s}/{'const' if c else 'geom'}"
        e_id = ensemble_from_values(ids, f"{tag}/identical", keys=SEEDS)
        e_ov = ensemble_from_values(ovs, f"{tag}/overlap", keys=SEEDS)
        e_ac = ensemble_from_values(accs, f"{tag}/acc", keys=SEEDS)
        summary[tag] = {"identical": ids, "overlap": ovs, "acc": accs,
                        "identical_ci": e_id.ci, "overlap_ci": e_ov.ci,
                        "acc_ci": e_ac.ci}
        print(f"  {s:8.2f} {'constant' if c else 'geometric':>9}  "
              f"{e_id.mean:.3f}+-{e_id.ci:.3f} {str([round(x,2) for x in ids]):>9} "
              f"{e_ov.mean:.3f}+-{e_ov.ci:.3f}  "
              f"{e_ac.mean:.3f}+-{e_ac.ci:.3f} "
              f"{str([round(a,2) for a in accs])}")

    def term(s, c, sc=True):
        """Terminal identical-assembly fraction as an ENSEMBLE over seeds.

        Amendment 3: this returned a bare mean until the methodology ratchet
        flagged it. A mean over seeds with no interval cannot support a
        comparison, and every bar below is a comparison.
        """
        return ensemble_from_values(
            [res[(s, c, sc, sd)]["terminal_identical"] for sd in SEEDS],
            f"term/{s}/{'const' if c else 'geom'}/scal={sc}", keys=SEEDS)

    def acc(s, c, sc=True):
        return ensemble_from_values(
            [res[(s, c, sc, sd)]["acc"] for sd in SEEDS],
            f"acc/{s}/{'const' if c else 'geom'}/scal={sc}", keys=SEEDS)

    def gt(a, b, label):
        """Is `a` above `b`? Judged on the PAIRED per-seed difference.

        Returns (verdict, text). A delta whose interval straddles zero is
        INCONCLUSIVE, not a pass -- with 3 seeds the t multiplier is 4.303 and
        the interval is wide, which is a fact about the sampling and belongs in
        the output rather than hidden behind a point estimate.
        """
        d = paired_delta(a, b, label=label)
        ok = d.beats(0.0)
        tag = "PASS" if ok else (
            "INCONCLUSIVE" if d.indistinguishable_from(0.0) else "FAIL")
        return ok, (f"{a.mean:.3f} vs {b.mean:.3f}, paired delta "
                    f"{d.mean:+.3f}+-{d.ci:.3f} [{tag}]")

    print("\n=== BARS ===")
    print("    judged on CONFIDENCE BOUNDS (Amendment 3); orderings are "
          "PAIRED per-seed deltas")
    e0 = term(0.0, False)
    r1 = e0.beats(0.90)
    print(f"  {'PASS' if r1 else 'FAIL'}  R1 strength=0 terminal identical "
          f"{e0.mean:.3f}+-{e0.ci:.3f}, CI-low {e0.low:.3f} > 0.90")

    r2a, t2a = gt(e0, term(0.05, False), "R2 0>0.05")
    r2b, t2b = gt(term(0.05, False), term(0.10, False), "R2 0.05>0.10")
    r2 = r2a and r2b
    print(f"  {'PASS' if r2 else 'FAIL'}  R2 monotone in strength (geometric)")
    print(f"        0 > 0.05:     {t2a}")
    print(f"        0.05 > 0.10:  {t2b}")

    r3a, t3a = gt(term(0.05, True), term(0.05, False), "R3 const>geom @.05")
    r3b, t3b = gt(term(0.10, True), term(0.10, False), "R3 const>geom @.10")
    r3 = r3a and r3b
    print(f"  {'PASS' if r3 else 'FAIL'}  R3 constant > geometric at equal "
          f"strength")
    print(f"        strength 0.05: {t3a}")
    print(f"        strength 0.10: {t3b}")

    r4, t4 = gt(acc(0.10, False), acc(0.0, False), "R4 acc(.1)>acc(0)")
    print(f"  {'PASS' if r4 else 'FAIL'}  R4 refraction earns its keep: {t4}")
    if not r4:
        print("        -> R4 FAILING is the bigger result: refraction costs "
              "stability and buys nothing measurable here.")

    r5_each = [gt(term(s, c, False), term(s, c, True), f"R5 {s}/{c}")
               for (s, c) in CONFIGS]
    r5 = all(ok for ok, _ in r5_each)
    print(f"  {'PASS' if r5 else 'FAIL'}  R5 scaling ON is LESS stable than "
          f"OFF at equal refraction (Amendment 2)")
    for (s, c), (_, txt) in zip(CONFIGS, r5_each):
        print(f"        strength={s} {'const' if c else 'geom':>5}: "
              f"OFF {txt}")

    path = os.path.join(_HERE, "seq_refraction_stability_results.json")
    with open(path, "w") as fh:
        json.dump({"configs": [list(c) for c in CONFIGS], "seeds": SEEDS,
                   "T": T, "summary": summary,
                   "cells": {f"{s}/{c}/{sc}/{sd}": res[(s, c, sc, sd)]
                             for (s, c) in CONFIGS for sc in SCALINGS
                             for sd in SEEDS},
                   "bars": {"R1": bool(r1), "R2": bool(r2),
                            "R3": bool(r3), "R4": bool(r4),
                            "R5": bool(r5)}}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
