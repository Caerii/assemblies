"""Does word-learning CAPACITY obey the anchor law?  PREREG_word_capacity.md.

The learner is `unaligned_scenes.Aligner` exactly as it passed U1-U3; only
n, k, the phon stimulus size and the vocabulary size V change between cells.
V* is where per-type alignment (against the whole inventory, chance 1/V)
crosses 0.90, read by the shared `ceiling_from_curve` standard -- a curve,
not a grid point, and CENSORED when it never crosses.

    python research/experiments/word_capacity.py [--seeds 42,1,2,3,4] [--cells A,B,C,D,E] [--smoke]

`--smoke` checks the API on a tiny grid. Its numbers are VOID.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from neural_assemblies.diagnostics import ensemble_from_values           # noqa: E402
from _substrate import ceiling_from_curve                               # noqa: E402
import unaligned_scenes as U                                            # noqa: E402

VS = (16, 32, 64, 128, 256, 512)
EXPOSURES = 6          # scenes per referent, on average
PER_SCENE = 3
CATS = 4
THRESHOLD = 0.90
#: (n, k, stim_size) per registered cell. D pairs with B on n/k; E pairs with
#: B on the anchor.
CELLS = {
    "A": (1000, 50, 50),
    "B": (2000, 50, 50),
    "C": (4000, 50, 50),
    "D": (4000, 100, 100),
    "E": (2000, 50, 100),
}


# ---------------------------------------------------------------------------
# synthetic grounded experience
# ---------------------------------------------------------------------------

def corpus(V, seed):
    """V referents (IDENT_i, CAT_{i mod CATS}); scenes of PER_SCENE referents;
    the sentence names them. Returns (experience, targets, words, features)
    in the learner's own shapes."""
    rng = random.Random(seed + 9001)
    bundles = [tuple(sorted((f"IDENT_{i}", f"CAT_{i % CATS}")))
               for i in range(V)]
    words = [f"w{i}" for i in range(V)]
    n_scenes = max(V * EXPOSURES // PER_SCENE, PER_SCENE)
    exp = []
    for _ in range(n_scenes):
        idx = rng.sample(range(V), PER_SCENE)
        exp.append(([words[i] for i in idx], [bundles[i] for i in idx]))
    targets = {words[i]: bundles[i] for i in range(V)}
    features = sorted({f for b in bundles for f in b})
    return exp, targets, words, features


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def type_accuracy(seed, V, n, k, stim_size):
    exp, targets, words, features = corpus(V, seed)
    exposures = Counter(w for ws, _b in exp for w in ws)
    al = U.Aligner(seed, words, features, scaling=True,
                   n=n, k=k, stim_size=stim_size)
    al.train(exp, random.Random(seed + 11))
    inventory = sorted({b for _w, bs in exp for b in bs})
    inv_asm = {b: al.bundle_assembly(b) for b in inventory}
    hits = 0
    scored = 0
    for w in words:
        if exposures[w] < U.MIN_EXPOSURES:
            continue
        rec = al.reconstruct(w)
        best = max(inventory, key=lambda b: rec.overlap(inv_asm[b]))
        hits += int(best == targets[w])
        scored += 1
    return hits / max(scored, 1), scored


def run_cell(name, seeds, vs):
    n, k, s = CELLS[name]
    curve = {}                                      # V -> [acc per seed]
    for V in vs:
        accs = []
        for seed in seeds:
            t0 = time.perf_counter()
            acc, scored = type_accuracy(seed, V, n, k, s)
            accs.append(acc)
            print(f"    {name} n={n} k={k} s={s} V={V:4d} seed {seed:2d}: "
                  f"type-acc {acc:.3f} (chance {1 / V:.3f}, n={scored})  "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        curve[V] = accs
        # stop early once the curve is clearly below threshold on every seed
        if max(accs) < THRESHOLD - 0.3:
            break
    return curve


def ceilings(curve, seeds):
    """Per-seed V* by the shared standard; ensemble across seeds."""
    stars, censored = [], 0
    for i, _seed in enumerate(seeds):
        pts = [(V, accs[i]) for V, accs in curve.items()]
        c = ceiling_from_curve(pts, threshold=THRESHOLD)
        if c.censored:
            censored += 1
        stars.append(float(c.m_star))
    return stars, censored


# ---------------------------------------------------------------------------

def judge(results, seeds):
    print("\n=== BARS (PREREG_word_capacity.md) ===")
    ens, cens = {}, {}
    for name in results:
        stars, c = ceilings(results[name], seeds)
        n, k, s = CELLS[name]
        e = ensemble_from_values(stars, label=f"{name} n={n} k={k} s={s} V*")
        ens[name], cens[name] = e, c
        print(f"  {e}   n/k {n / k:.0f}   censored seeds {c}/{len(seeds)}")

    def ok(*names):
        return all(nm in ens and cens[nm] == 0 for nm in names)

    if ok("A", "B", "C"):
        a, b, c = ens["A"], ens["B"], ens["C"]
        step1 = (b.mean - a.mean) > (a.high - a.mean) + (b.mean - b.low)
        step2 = (c.mean - b.mean) > (b.high - b.mean) + (c.mean - c.low)
        print(f"  {'PASS' if step1 and step2 else 'FAIL'}  W1 V* rises with "
              f"n/k: A<B {step1}, B<C {step2} (beyond pooled CI)")
    else:
        print("  VOID  W1 -- a cell is censored or missing")
    if ok("B", "D"):
        r = ens["D"].mean / ens["B"].mean
        v = ("PASS" if abs(r - 1) <= 0.25 else
             ("FAIL" if abs(r - 1) > 0.40 else "INCONCLUSIVE"))
        print(f"  {v}  W2 ratio law: V*(D)/V*(B) = {r:.2f} at equal n/k")
    else:
        print("  VOID  W2 -- a cell is censored or missing")
    if ok("B", "E"):
        r = ens["E"].mean / ens["B"].mean
        v = "PASS" if r >= 1.3 else ("FAIL" if r <= 1.1 else "INCONCLUSIVE")
        print(f"  {v}  W3 anchor: V*(E)/V*(B) = {r:.2f} (s doubled)")
    else:
        print("  VOID  W3 -- a cell is censored or missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--cells", default="A,B,C,D,E")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = [int(x) for x in args.seeds.split(",")]
    vs = (8, 16) if args.smoke else VS
    if args.smoke:
        print("SMOKE: API check only; numbers VOID")
    print(f"WORD CAPACITY  cells {args.cells}  V grid {vs}  seeds {seeds}  "
          f"threshold {THRESHOLD}")
    results = {}
    for name in args.cells.split(","):
        results[name] = run_cell(name, seeds, vs)
    judge(results, seeds)
    path = os.path.join(_HERE, "word_capacity_results.json")
    with open(path, "w") as fh:
        json.dump({"seeds": seeds, "cells": {nm: {str(V): a for V, a in c.items()}
                                              for nm, c in results.items()}},
                  fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
