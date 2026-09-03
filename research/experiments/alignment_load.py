"""Does word learning obey the anchor law, and does Zipf attack the corrector?

Implements `research/notes/PREREG_alignment_load.md`. Reuses the learner from
`unaligned_scenes.py` unchanged -- only the EXPERIENCE it is given changes, so
a difference between cells cannot come from the model.

    LOAD       distractor bundles from the corpus's own inventory: objects
               perceived but not named. A referent is then 1/P of the
               grounding drive on any one exposure.
    FREQUENCY  Zipfian word-type frequencies by resampling sentences, total
               presentations held fixed.

    python research/experiments/alignment_load.py [--seeds 42,1,2,3,4]
"""
from __future__ import annotations

import argparse
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
from unaligned_scenes import (                                          # noqa: E402
    MIN_EXPOSURES, align, experience_of, targets_of)

LOADS = (2, 3, 5, 8)
#: Amendment 1: P=2 TRIMS a participant while 5 and 8 ADD distractors, so only
#: these three differ by load alone and only these are judged for monotonicity.
ADD_ONLY = (3, 5, 8)
ZIPF_S = 1.0
#: L3's gap is measured at these loads only -- the scaling-OFF arm doubles the
#: cell count and the mechanism claim does not need every load.
GAP_LOADS = (3, 8)


def set_load(exp, P, seed):
    """Force every scene to exactly P bundles: trim, or add DISTRACTORS.

    Distractors are drawn from the corpus's own bundle inventory, excluding
    the scene's own -- things present in the perceived situation that nobody
    named. Trimming (P=2) keeps the ACTION bundle plus the first participant,
    so a scene never loses the referent of a word it contains... which cannot
    be guaranteed, so trimmed-away referents are handled by the scorer: a word
    whose target is absent from its scene is excluded from that occurrence.
    """
    rng = random.Random(seed + 313)
    inventory = sorted({b for _w, bs in exp for b in bs})
    out = []
    for words, bundles in exp:
        bs = list(bundles)
        if len(bs) > P:
            bs = bs[:P]
        else:
            pool = [b for b in inventory if b not in bs]
            rng.shuffle(pool)
            bs = bs + pool[:P - len(bs)]
        out.append((words, bs))
    return out


def zipfify(exp, seed, s=ZIPF_S):
    """Resample sentences so word-type frequencies follow Zipf, N fixed."""
    rng = random.Random(seed + 4242)
    types = sorted({w for ws, _b in exp for w in ws})
    ranks = types[:]
    rng.shuffle(ranks)
    weight = {w: (i + 1) ** (-s) for i, w in enumerate(ranks)}
    sw = [max(sum(weight[w] for w in ws), 1e-9) for ws, _b in exp]
    tot = sum(sw)
    probs = [x / tot for x in sw]
    idx = rng.choices(range(len(exp)), weights=probs, k=len(exp))
    return [exp[i] for i in idx]


def per_occurrence(scores, exp, targets, exposures):
    """Registered metric: argmax over the scene's OWN bundles."""
    occ = hit = 0
    chance = 0.0
    by_word = Counter()
    seen = Counter()
    for words, bundles in exp:
        for w in words:
            if w not in targets or exposures[w] < MIN_EXPOSURES:
                continue
            if targets[w] not in bundles:      # trimmed away; not scoreable
                continue
            best = max(bundles, key=lambda b: scores[w][b])
            occ += 1
            seen[w] += 1
            ok = int(best == targets[w])
            hit += ok
            by_word[w] += ok
            chance += 1.0 / len(bundles)
    return (hit / max(occ, 1), chance / max(occ, 1), occ,
            {w: by_word[w] / seen[w] for w in seen})


def cell(seed, exp, targets, words, features, scaling):
    exposures = Counter(w for ws, _b in exp for w in ws)
    scores, inventory = align(seed, exp, words, features, scaling=scaling)
    acc, ch, occ, by_word = per_occurrence(scores, exp, targets, exposures)
    return acc, ch, occ, by_word, exposures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from grounded_corpus import build
    corpus = build()
    base = experience_of(corpus)
    targets = targets_of(corpus)
    words = sorted({w for ws, _b in base for w in ws})
    features = sorted({f for _w, bs in base for b in bs for f in b})

    print("ALIGNMENT UNDER LOAD AND ZIPF  (PREREG_alignment_load.md)")
    print(f"  base scenes {len(base)}  word types {len(words)}  "
          f"loads {LOADS}  zipf s={ZIPF_S}  seeds {seeds}")

    results = {}          # (corpus, P, scaling) -> [acc per seed]
    tails = {}
    for corpus_kind in ("flat", "zipf"):
        for P in LOADS:
            arms = [True] + ([False] if P in GAP_LOADS else [])
            for scaling in arms:
                accs, chs = [], []
                for seed in seeds:
                    t0 = time.perf_counter()
                    e = set_load(base, P, seed)
                    if corpus_kind == "zipf":
                        e = zipfify(e, seed)
                    acc, ch, occ, by_word, exposures = cell(
                        seed, e, targets, words, features, scaling)
                    accs.append(acc)
                    chs.append(ch)
                    if corpus_kind == "zipf" and P == 3:
                        ranked = sorted(by_word, key=lambda w: -exposures[w])
                        half = max(len(ranked) // 2, 1)
                        head = float(np.mean([by_word[w] for w in ranked[:half]]))
                        tail = float(np.mean([by_word[w] for w in ranked[half:]]
                                             or [float("nan")]))
                        tails.setdefault(scaling, []).append((head, tail))
                    print(f"    {corpus_kind:4s} P={P} scaling={str(scaling):5s} "
                          f"seed {seed:2d}: acc {acc:.3f} (chance {ch:.3f}, "
                          f"n={occ})  [{time.perf_counter() - t0:.0f}s]",
                          flush=True)
                results[(corpus_kind, P, scaling)] = (accs, float(np.mean(chs)))

    print("\n=== BARS ===")
    # L1 is a THREE-WAY as registered (pass / fail / neither), and one clause
    # of it is ill-posed: at P=2, "lower bound above 2x chance" demands an
    # accuracy above 1.000, which nothing can reach. Those cells are reported
    # and excluded from the verdict rather than counted as failures. The
    # earlier binary print called the whole bar FAIL on that clause alone.
    means, l1_fail, l1_pass = {}, False, True
    for corpus_kind in ("flat", "zipf"):
        print(f"  -- {corpus_kind} corpus, scaling ON")
        for P in LOADS:
            accs, ch = results[(corpus_kind, P, True)]
            e = ensemble_from_values(accs, label=f"{corpus_kind} P={P}")
            means[(corpus_kind, P)] = e.mean
            unreachable = 2 * ch >= 1.0
            beats = e.beats(2 * ch)
            note = ("bar 2x chance is UNREACHABLE here (>= 1.0), excluded"
                    if unreachable else
                    f"lower bound > 2x chance: {beats}")
            print(f"    {e}   chance {ch:.3f}  x chance {e.mean / ch:.2f}  "
                  f"{note}")
            if unreachable:
                continue
            if e.mean <= 1.5 * ch:
                l1_fail = True
            if not beats:
                l1_pass = False
        heaviest = means[(corpus_kind, LOADS[-1])]
        if heaviest < 0.35:
            l1_fail = True
        if heaviest < 0.60:
            l1_pass = False
    verdict = "FAIL" if l1_fail else ("PASS" if l1_pass else "INCONCLUSIVE")
    print(f"  {verdict}  L1 graceful: reachable loads above 2x chance on the "
          f"bound, heaviest load >= 0.60; FAIL zone is <= 1.5x chance or "
          f"heaviest < 0.35")

    ok_l2 = True
    for corpus_kind in ("flat", "zipf"):
        seq = [means[(corpus_kind, P)] for P in ADD_ONLY]
        mono = all(a >= b - 1e-9 for a, b in zip(seq, seq[1:]))
        gap = seq[0] - seq[-1]
        e2 = ensemble_from_values(results[(corpus_kind, ADD_ONLY[0], True)][0],
                                  label=f"P={ADD_ONLY[0]}")
        e8 = ensemble_from_values(results[(corpus_kind, 8, True)][0],
                                  label="P=8")
        sep = gap > (e2.mean - e2.low) + (e8.high - e8.mean)
        print(f"  {corpus_kind} (add-only {ADD_ONLY}): "
              f"{[f'{x:.3f}' for x in seq]}  monotone {mono}  "
              f"P{ADD_ONLY[0]}-P{ADD_ONLY[-1]} gap {gap:+.3f}  "
              f"exceeds pooled CI {sep}")
        ok_l2 = ok_l2 and mono and sep
    print(f"  {'PASS' if ok_l2 else 'FAIL'}  L2 accuracy falls monotonically "
          f"over the ADD-ONLY loads, lightest beating heaviest beyond seed "
          f"noise (Amendment 1; P=2 is a trimmed cell, reported not judged)")

    gaps = {}
    print("  -- L3 scaling ON minus OFF")
    for corpus_kind in ("flat", "zipf"):
        for P in GAP_LOADS:
            on = float(np.mean(results[(corpus_kind, P, True)][0]))
            off = float(np.mean(results[(corpus_kind, P, False)][0]))
            gaps[(corpus_kind, P)] = on - off
            print(f"    {corpus_kind} P={P}: ON {on:.3f}  OFF {off:.3f}  "
                  f"gap {on - off:+.3f}")
    ok_l3 = (all(g > 0 for g in gaps.values())
             and all(gaps[("zipf", P)] > gaps[("flat", P)] for P in GAP_LOADS))
    print(f"  {'PASS' if ok_l3 else 'FAIL'}  L3 gap positive everywhere and "
          f"LARGER under Zipf at both loads")

    if tails:
        print("  -- Z1 head vs tail under Zipf at P=3 (reported, no bar)")
        for scaling, pairs in sorted(tails.items(), key=lambda kv: not kv[0]):
            h = float(np.mean([a for a, _b in pairs]))
            t = float(np.nanmean([b for _a, b in pairs]))
            print(f"    scaling={str(scaling):5s}  head {h:.3f}  tail {t:.3f}"
                  f"  head-tail {h - t:+.3f}")


if __name__ == "__main__":
    main()
