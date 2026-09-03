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
#: Hashed path, Amendment 2: two cross rounds per (word, bundle) step. The
#: registered five were measured not load-bearing -- U1 on the hashed learner
#: reads 1.000 on all five brains at rounds 2, 3 and 5 (0.97-1.00 at 1) -- and
#: the study's cost is linear in them.
ROUNDS_HASHED = 2
EXPOSURES = 12         # scenes per referent (Amendment 1: 6 sat below threshold at V=16)
FEAT_N, FEAT_K = 1000, 50
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
    # FEAT is FIXED across cells (registration: n=1000, k=50); only LEX and
    # the phon anchor vary. Letting FEAT follow n made the n=4000 cell read
    # WORSE than n=1000 at every exposure level -- more FEAT columns
    # competing at readout -- which is a readout floor, not capacity.
    al = U.Aligner(seed, words, features, scaling=True,
                   n=n, k=k, stim_size=stim_size,
                   feat_n=FEAT_N, feat_k=FEAT_K)
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


def type_accuracy_hashed(seeds, V, n, k, stim_size, track_pinned=False):
    """All seeds as ONE batch of brains on the hashed substrate.

    The corpus is shared across the brains (seeded by the first seed): the
    brains differ by connectome, which is what a seed varies in every other
    hashed study. The numpy path draws a corpus per seed; that is the one
    protocol difference between the engines here, and it is a nuisance
    variable, not a treatment. Returns per-brain accuracies, and the
    pinned-winner trace when asked (DESIGN_hashed_aligner.md).
    """
    import torch
    from neural_assemblies.core.torch_engine._hashed_aligner import HashedAligner
    exp, targets, words, features = corpus(V, seeds[0])
    exposures = Counter(w for ws, _b in exp for w in ws)
    # Unclipped, max-relative pricing, anchors at gain 1/p -- the exact
    # regime the parity gate verified (DESIGN_hashed_aligner.md).
    al = HashedAligner(seeds, words, features, n=n, k=k, feat_n=FEAT_N,
                       feat_k=FEAT_K, stim_size=stim_size, p=U.P, beta=U.BETA,
                       rounds_word=ROUNDS_HASHED, track_pinned=track_pinned)
    al.train(exp, random.Random(seeds[0] + 11))
    inventory = sorted({b for _w, bs in exp for b in bs})
    scored = [w for w in words if exposures[w] >= U.MIN_EXPOSURES]
    table = al.overlap_table(scored, inventory)            # [V, I, B]
    best = table.argmax(dim=1)                             # [V, B]
    tgt = torch.tensor([inventory.index(targets[w]) for w in scored],
                       device=best.device).view(-1, 1)
    acc = (best == tgt).float().mean(dim=0).cpu().numpy()  # per brain
    return acc, len(scored), al.pinned


def run_cell_scheduled(name, seeds, vs):
    """Every (V, seed) task of a cell in ONE launch (layer 1). Each brain has
    its own corpus (seeded by its seed), vocabulary, bundle inventory and
    schedule; only the area shape is shared. Returns the same curve record
    as `run_cell` so `judge` cannot tell the difference."""
    import torch
    from neural_assemblies.core.torch_engine._scheduled_aligner import (
        ScheduledAligner, pad_schedules, schedule_of)
    n, k, stim = CELLS[name]
    tasks = [(V, seed) for V in vs for seed in seeds]
    per = []
    for V, seed in tasks:
        exp, targets, words, features = corpus(V, seed)
        exposures = Counter(w for ws, _b in exp for w in ws)
        inventory = sorted({b for _w, bs in exp for b in bs})
        wi = {w: i for i, w in enumerate(words)}
        bi = {b: j for j, b in enumerate(inventory)}
        order = list(range(len(exp)))
        random.Random(seed + 11).shuffle(order)
        per.append(dict(V=V, seed=seed, words=words, features=features,
                        inventory=inventory, targets=targets,
                        exposures=exposures, sched=schedule_of(exp, wi, bi, order)))
    Vmax = max(len(t["words"]) for t in per)
    Fmax = max(len(t["features"]) for t in per)
    Imax = max(len(t["inventory"]) for t in per)
    Fper = max(len(b) for t in per for b in t["inventory"])
    B = len(per)
    feats = torch.full((B, Imax, Fper), -1, dtype=torch.int64)
    tgt = torch.full((B, Vmax), -1, dtype=torch.int64)
    expo = torch.zeros(B, Vmax, dtype=torch.int64)
    nb = torch.zeros(B, dtype=torch.int64)
    for b, t in enumerate(per):
        fi = {f: i for i, f in enumerate(t["features"])}
        bi = {bb: j for j, bb in enumerate(t["inventory"])}
        for j, bb in enumerate(t["inventory"]):
            for sl, f in enumerate(bb):
                feats[b, j, sl] = fi[f]
        for i, w in enumerate(t["words"]):
            tgt[b, i] = bi[t["targets"][w]]
            expo[b, i] = t["exposures"][w]
        nb[b] = len(t["inventory"])
    W, Bd = pad_schedules([t["sched"] for t in per])
    t0 = time.perf_counter()
    # word/feature INDEX i means brain b's own word i: every brain seeds its
    # phon fibers by (seed_b, "phon_i"), so brains share nothing but shape
    al = ScheduledAligner([t["seed"] * 1000 + t["V"] for t in per], n=n, k=k,
                          feat_n=FEAT_N, feat_k=FEAT_K, n_words=Vmax,
                          n_features=Fmax, stim_size=stim, p=U.P, beta=U.BETA,
                          rounds_word=ROUNDS_HASHED)
    al.prepare(feats)
    al.train(W, Bd, device_loop=True)          # layer 3: one launch per cell
    acc, scored = al.type_accuracy(tgt, nb, expo, U.MIN_EXPOSURES)
    acc = acc.cpu().numpy()
    print(f"    {name} n={n} k={k} s={stim}: {B} brains (V x seed) in one "
          f"launch, {W.shape[1]} steps  [{time.perf_counter() - t0:.0f}s]",
          flush=True)
    curve = {V: [] for V in vs}
    for b, t in enumerate(per):
        curve[t["V"]].append(float(acc[b]))
    for V in vs:
        print(f"      V={V:4d}: type-acc {' '.join(f'{a:.3f}' for a in curve[V])}"
              f"  (chance {1 / V:.3f})", flush=True)
    return curve


def run_cell(name, seeds, vs, engine="numpy", track_pinned=False):
    if engine == "scheduled":
        return run_cell_scheduled(name, seeds, vs)
    n, k, s = CELLS[name]
    curve = {}                                      # V -> [acc per seed]
    pinned = {}
    for V in vs:
        accs = []
        if engine == "hashed":
            t0 = time.perf_counter()
            acc, scored, pin = type_accuracy_hashed(seeds, V, n, k, s,
                                                     track_pinned)
            accs = [float(a) for a in acc]
            if pin:
                pinned[V] = pin
            print(f"    {name} n={n} k={k} s={s} V={V:4d} hashed B={len(seeds)}: "
                  f"type-acc {' '.join(f'{a:.3f}' for a in accs)} "
                  f"(chance {1 / V:.3f}, n={scored})"
                  + (f"  pinned min {min(pin):.3f} mean "
                     f"{sum(pin) / len(pin):.3f}" if pin else "")
                  + f"  [{time.perf_counter() - t0:.0f}s]", flush=True)
        else:
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
        # Censored in EITHER direction: never crossed (high) or never above
        # the threshold at all (low -- the standard returns the smallest V
        # uncensored there, which would read as a value).
        if c.censored or max(a for _v, a in pts) <= THRESHOLD:
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
    ap.add_argument("--engine", choices=("numpy", "hashed", "scheduled"),
                    default="numpy",
                    help="hashed = all seeds batched on the generated-connectome "
                         "substrate (DESIGN_hashed_aligner.md)")
    ap.add_argument("--track-pinned", action="store_true",
                    help="hashed only: measure the GEMM-shortcut precondition")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = [int(x) for x in args.seeds.split(",")]
    vs = (8, 16) if args.smoke else VS
    if args.smoke:
        print("SMOKE: API check only; numbers VOID")
    print(f"WORD CAPACITY  engine {args.engine}  cells {args.cells}  "
          f"V grid {vs}  seeds {seeds}  threshold {THRESHOLD}")
    results = {}
    for name in args.cells.split(","):
        results[name] = run_cell(name, seeds, vs, engine=args.engine,
                                 track_pinned=args.track_pinned)
    judge(results, seeds)
    path = os.path.join(_HERE, f"word_capacity_results_{args.engine}.json")
    with open(path, "w") as fh:
        json.dump({"seeds": seeds, "cells": {nm: {str(V): a for V, a in c.items()}
                                              for nm, c in results.items()}},
                  fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
