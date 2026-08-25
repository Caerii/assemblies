"""Refraction as the stability/conjunctivity tradeoff.

Implements `research/notes/PREREG_refraction_stability.md` (+ Amendment 1).

`_refraction.refraction_increment` returns `(net_drive + current_bias) *
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
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

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

    print(f"\n  {'strength':>8} {'mode':>9}  {'terminal identical':>18} "
          f"{'terminal overlap':>16} {'acc(L=100)':>11}")
    summary = {}
    for (s, c) in CONFIGS:
        ids = [res[(s, c, True, sd)]["terminal_identical"] for sd in SEEDS]
        ovs = [res[(s, c, True, sd)]["terminal_overlap"] for sd in SEEDS]
        accs = [res[(s, c, True, sd)]["acc"] for sd in SEEDS]
        summary[f"{s}/{'const' if c else 'geom'}"] = {
            "identical": ids, "overlap": ovs, "acc": accs}
        print(f"  {s:8.2f} {'constant' if c else 'geometric':>9}  "
              f"{np.mean(ids):8.3f} {str([round(x,2) for x in ids]):>9} "
              f"{np.mean(ovs):8.3f}  {np.mean(accs):8.3f} "
              f"{str([round(a,2) for a in accs])}")

    def term(s, c, sc=True):
        return float(np.mean([res[(s, c, sc, sd)]["terminal_identical"]
                              for sd in SEEDS]))

    def acc(s, c, sc=True):
        return float(np.mean([res[(s, c, sc, sd)]["acc"] for sd in SEEDS]))

    print("\n=== BARS ===")
    r1 = term(0.0, False) > 0.90
    print(f"  {'PASS' if r1 else 'FAIL'}  R1 strength=0 terminal identical "
          f"{term(0.0, False):.3f} > 0.90")
    r2 = term(0.0, False) > term(0.05, False) > term(0.10, False)
    print(f"  {'PASS' if r2 else 'FAIL'}  R2 monotone in strength (geometric): "
          f"{term(0.0, False):.3f} > {term(0.05, False):.3f} > "
          f"{term(0.10, False):.3f}")
    r3 = (term(0.05, True) > term(0.05, False)
          and term(0.10, True) > term(0.10, False))
    print(f"  {'PASS' if r3 else 'FAIL'}  R3 constant > geometric at equal "
          f"strength: {term(0.05, True):.3f}>{term(0.05, False):.3f}, "
          f"{term(0.10, True):.3f}>{term(0.10, False):.3f}")
    r4 = acc(0.0, False) < acc(0.10, False)
    print(f"  {'PASS' if r4 else 'FAIL'}  R4 refraction earns its keep: "
          f"acc(0)={acc(0.0, False):.3f} < acc(0.1)={acc(0.10, False):.3f}")
    if not r4:
        print("        -> R4 FAILING is the bigger result: refraction costs "
              "stability and buys nothing measurable here.")

    r5 = all(term(s, c, True) < term(s, c, False) for (s, c) in CONFIGS)
    print(f"  {'PASS' if r5 else 'FAIL'}  R5 scaling ON is LESS stable than "
          f"OFF at equal refraction (Amendment 2)")
    for (s, c) in CONFIGS:
        print(f"        strength={s} {'const' if c else 'geom':>5}: "
              f"scaling ON {term(s, c, True):.3f}  OFF {term(s, c, False):.3f}")

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
