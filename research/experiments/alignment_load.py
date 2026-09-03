"""Does word learning obey the anchor law, and does Zipf attack the corrector?

Implements `research/notes/PREREG_alignment_load.md`. Reuses the learner from
`unaligned_scenes.py` unchanged -- only the EXPERIENCE it is given changes, so
a difference between cells cannot come from the model.

    LOAD       distractor bundles from the corpus's own inventory: objects
               perceived but not named. A referent is then 1/P of the
               grounding drive on any one exposure.
    FREQUENCY  Zipfian word-type frequencies by resampling sentences, total
               presentations held fixed.

The bars are judged by `judge()`, which is fed either fresh cells or a
committed log (``--replay alignment_load.log``): the verdict is a function of
the per-seed numbers and nothing else, so it can be re-derived without
re-running an hour of cells. Every seed-level comparison goes through
`diagnostics.ensemble_from_values` and is judged on the confidence BOUND.

    python research/experiments/alignment_load.py [--seeds 42,1,2,3,4]
    python research/experiments/alignment_load.py --replay research/experiments/alignment_load.log
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

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
CORPORA = ("flat", "zipf")


# ---------------------------------------------------------------------------
# the two manipulations
# ---------------------------------------------------------------------------

def set_load(exp, P, seed):
    """Force every scene to exactly P bundles: trim, or add DISTRACTORS.

    Distractors are drawn from the corpus's own bundle inventory, excluding
    the scene's own -- things present in the perceived situation that nobody
    named. Trimming (P=2) keeps the first P bundles; a word whose referent was
    trimmed away is excluded from that occurrence by the scorer.
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


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

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


def head_tail(by_word, exposures):
    """Mean per-word accuracy over the more- and less-exposed halves.

    Averages over WORDS within one seed (a condition mean, not a seed mean);
    the seed-level statistic is formed by `judge`.
    """
    ranked = sorted(by_word, key=lambda w: -exposures[w])
    half = max(len(ranked) // 2, 1)
    head = float(np.mean([by_word[w] for w in ranked[:half]]))
    tail = float(np.mean([by_word[w] for w in ranked[half:]]
                         or [float("nan")]))
    return head, tail


def run_cells(seeds, base, targets, words, features):
    """Every registered cell. Returns the per-seed record `judge` consumes."""
    acc = defaultdict(dict)        # (corpus, P, scaling) -> {seed: acc}
    chance = defaultdict(dict)     # (corpus, P, scaling) -> {seed: chance}
    tails = defaultdict(dict)      # scaling -> {seed: (head, tail)}
    for corpus_kind in CORPORA:
        for P in LOADS:
            arms = [True] + ([False] if P in GAP_LOADS else [])
            for scaling in arms:
                for seed in seeds:
                    t0 = time.perf_counter()
                    e = set_load(base, P, seed)
                    if corpus_kind == "zipf":
                        e = zipfify(e, seed)
                    exposures = Counter(w for ws, _b in e for w in ws)
                    scores, _inv = align(seed, e, words, features,
                                         scaling=scaling)
                    a, ch, occ, by_word = per_occurrence(
                        scores, e, targets, exposures)
                    acc[(corpus_kind, P, scaling)][seed] = a
                    chance[(corpus_kind, P, scaling)][seed] = ch
                    if corpus_kind == "zipf" and P == 3:
                        tails[scaling][seed] = head_tail(by_word, exposures)
                    print(f"    {corpus_kind:4s} P={P} scaling={str(scaling):5s} "
                          f"seed {seed:2d}: acc {a:.3f} (chance {ch:.3f}, "
                          f"n={occ})  [{time.perf_counter() - t0:.0f}s]",
                          flush=True)
    return acc, chance, tails


# ---------------------------------------------------------------------------
# the bars
# ---------------------------------------------------------------------------

def judge(acc, chance, tails, seeds):
    """The registered bars, from per-seed records only.

    L1 is a THREE-WAY as registered (pass / fail / neither). One clause of it
    is ill-posed: at P=2, "lower bound above 2x chance" demands an accuracy
    above 1.000, which nothing can reach. Those cells are reported and
    excluded from the verdict rather than counted as failures. L3's gap is
    PAIRED (same seed, ON minus OFF) and judged on the bound; the first draft
    compared two bare seed means, which the methodology ratchet caught.
    """
    def ens(key, label):
        return ensemble_from_values([acc[key][s] for s in seeds], label=label)

    def chance_of(key):
        return ensemble_from_values([chance[key][s] for s in seeds],
                                    label="chance").mean

    print("\n=== BARS ===")
    means, l1_fail, l1_pass = {}, False, True
    for corpus_kind in CORPORA:
        print(f"  -- {corpus_kind} corpus, scaling ON")
        for P in LOADS:
            key = (corpus_kind, P, True)
            e, ch = ens(key, f"{corpus_kind} P={P}"), chance_of(key)
            means[(corpus_kind, P)] = e.mean
            unreachable = 2 * ch >= 1.0
            beats = e.beats(2 * ch)
            note = ("bar 2x chance is UNREACHABLE here (>= 1.0), excluded"
                    if unreachable else f"lower bound > 2x chance: {beats}")
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
    for corpus_kind in CORPORA:
        seq = [means[(corpus_kind, P)] for P in ADD_ONLY]
        mono = all(a >= b - 1e-9 for a, b in zip(seq, seq[1:]))
        gap = seq[0] - seq[-1]
        lo = ens((corpus_kind, ADD_ONLY[0], True), "lightest")
        hi = ens((corpus_kind, ADD_ONLY[-1], True), "heaviest")
        sep = gap > (lo.mean - lo.low) + (hi.high - hi.mean)
        print(f"  {corpus_kind} (add-only {ADD_ONLY}): "
              f"{[f'{x:.3f}' for x in seq]}  monotone {mono}  "
              f"P{ADD_ONLY[0]}-P{ADD_ONLY[-1]} gap {gap:+.3f}  "
              f"exceeds pooled CI {sep}")
        ok_l2 = ok_l2 and mono and sep
    print(f"  {'PASS' if ok_l2 else 'FAIL'}  L2 accuracy falls monotonically "
          f"over the ADD-ONLY loads, lightest beating heaviest beyond seed "
          f"noise (Amendment 1; P=2 is a trimmed cell, reported not judged)")

    print("  -- L3 scaling ON minus OFF, PAIRED per seed, judged on the bound")
    gaps, positive = {}, True
    for corpus_kind in CORPORA:
        for P in GAP_LOADS:
            g = ensemble_from_values(
                [acc[(corpus_kind, P, True)][s] - acc[(corpus_kind, P, False)][s]
                 for s in seeds], label=f"{corpus_kind} P={P} ON-OFF")
            gaps[(corpus_kind, P)] = g.mean
            positive = positive and g.beats(0.0)
            print(f"    {g}   beats 0: {g.beats(0.0)}")
    grows = all(gaps[("zipf", P)] > gaps[("flat", P)] for P in GAP_LOADS)
    print(f"  {'PASS' if positive else 'FAIL'}  L3a gap positive on the bound "
          f"in every cell")
    print(f"  {'PASS' if grows else 'FAIL'}  L3b gap LARGER under Zipf at "
          f"both loads (registered; see the Result for why this clause "
          f"cannot test the claim)")

    if tails:
        print("  -- Z1 head vs tail under Zipf at P=3 (reported, no bar)")
        for scaling in (True, False):
            if scaling not in tails:
                continue
            h = ensemble_from_values([tails[scaling][s][0] for s in seeds],
                                     label=f"scaling={scaling} head")
            t = ensemble_from_values([tails[scaling][s][1] for s in seeds],
                                     label=f"scaling={scaling} tail")
            print(f"    {h}\n    {t}    head - tail {h.mean - t.mean:+.3f}")


# ---------------------------------------------------------------------------
# replay: the verdict from a committed log
# ---------------------------------------------------------------------------

_CELL = re.compile(r"\s+(flat|zipf) P=(\d) scaling=(True|False)\s+seed\s+(\d+):"
                   r" acc ([\d.]+) \(chance ([\d.]+)")


def replay(path):
    """Per-seed records from a committed log. Z1's head/tail lines were not
    logged per seed in the first run, so a replay judges L1-L3 only."""
    acc, chance = defaultdict(dict), defaultdict(dict)
    seeds = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _CELL.match(line)
        if not m:
            continue
        key = (m[1], int(m[2]), m[3] == "True")
        seed = int(m[4])
        acc[key][seed] = float(m[5])
        chance[key][seed] = float(m[6])
        if seed not in seeds:
            seeds.append(seed)
    return acc, chance, {}, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--replay", default=None,
                    help="judge the bars from a committed log instead of running")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.replay:
        acc, chance, tails, seeds = replay(args.replay)
        print(f"REPLAY of {args.replay}: seeds {seeds}, "
              f"{sum(len(v) for v in acc.values())} cells")
        judge(acc, chance, tails, seeds)
        return

    seeds = [int(x) for x in args.seeds.split(",")]
    from grounded_corpus import build
    corpus = build()
    base = experience_of(corpus)
    targets = targets_of(corpus)
    words = sorted({w for ws, _b in base for w in ws})
    features = sorted({f for _w, bs in base for b in bs for f in b})
    print("ALIGNMENT UNDER LOAD AND ZIPF  (PREREG_alignment_load.md)")
    print(f"  base scenes {len(base)}  word types {len(words)}  "
          f"loads {LOADS}  zipf s={ZIPF_S}  seeds {seeds}")
    acc, chance, tails = run_cells(seeds, base, targets, words, features)
    judge(acc, chance, tails, seeds)


if __name__ == "__main__":
    main()
