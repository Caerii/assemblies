"""A Zipfian corpus, because uniform item frequency is the least plausible
thing about this whole campaign. (task #46)

WHAT IS WRONG WITH WHAT WE HAVE. Every measurement so far presents each item the
same number of times. Real vocabularies are Zipfian -- frequency falls roughly
as 1/rank -- so a handful of items are seen thousands of times and the majority
a handful of times. That is not a cosmetic difference here, because plasticity
is CUMULATIVE: a fiber is potentiated once per co-firing, so

    effective gain of an item ~ (1 + beta)^(T * presentations)

and under Zipf the number of presentations spans orders of magnitude WITHIN ONE
AREA. The gain, which every result so far has treated as a single control
parameter, becomes a DISTRIBUTION over items.

THE PREDICTION THAT FOLLOWS, registered before measuring. The wedge has two
walls, and frequency slides items along the gain axis, so:

  * HEAD items (frequent, over-potentiated) should hit the CROWDING wall first,
    and should be the ones that merge.
  * TAIL items (rare, under-potentiated) should sit against the STARVATION
    wall, and should be the ones that never form.
  * Therefore the band of beta where BOTH work is NARROWER than the uniform
    wedge, and at strong enough skew it may not exist at all -- head and tail
    demanding opposite corrections in the same area at the same time.

If that holds, it is a much stronger version of the per-area-gain argument: not
merely that different AREAS want different gain, but that a single area cannot
be served by one gain when its vocabulary is Zipfian. Which in turn predicts
that something frequency-compensating is needed -- weight normalisation,
homeostatic scaling, or a per-item learning rate.

WHAT WOULD REFUTE IT. Head and tail failing together, or the Zipfian wedge
matching the uniform one. That would mean cumulative potentiation saturates
(w_max is 20.0, so this is entirely possible) and frequency stops mattering
above some count -- itself worth knowing, and the reason presentations are
recorded per item rather than assumed.

MEASUREMENT. Items are drawn with Zipf(s) weights for a fixed total number of
presentations, so the comparison against uniform holds TOTAL WORK constant
rather than per-item work -- otherwise a Zipfian run would simply have done more
training. Retrieval is then scored separately by frequency tercile.

FIRST VERSION MEASURED THE WRONG THING, and the correction is itself a result.
It applied the Zipfian schedule to a single feed-forward stimulus->area lexicon,
and frequency had NO effect whatever: at skew 1.0 the head item was presented 81
times and the tail once, and both retrieved at 1.000.

That is not a null result, it is the known mechanism. Feed-forward selection
depends on the ORDERING of a stimulus's weights, and potentiation multiplies the
winning assembly's weights uniformly, which cannot reorder them. It is the same
reason beta has no measurable effect on lexicon capacity. So repeated
presentation of a word cannot change which neurons represent it.

PREDICTION, sharpened: the lexicon is frequency-ROBUST -- a word seen once is
represented as well as a word seen a thousand times -- while COMPOSITION is
frequency-sensitive, because there the potentiated fiber competes against other
potentiated fibers and relative magnitude decides the winner. The experiment
therefore measures a depth-D composition chain, with each item's MERGE repeated
according to its frequency.
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from _substrate import read, similarity, probe  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.environ.get("ZIPF_OUT", "zipf_gain.csv"))

N = int(os.environ.get("ZIPF_N", "4000"))
K = int(os.environ.get("ZIPF_K", "50"))
P = float(os.environ.get("ZIPF_P", "0.05"))
M = int(os.environ.get("ZIPF_M", "64"))
T = 2
#: 0.0 is uniform (the control); 1.0 is classic Zipf.
SKEWS = [float(x) for x in os.environ.get("ZIPF_S", "0.0,0.7,1.0,1.3").split(",")]
GAINS = [float(x) for x in os.environ.get(
    "ZIPF_GAINS", "1.08,1.22,1.40,1.60,1.78,1.96,2.20").split(",")]
SEEDS = [int(x) for x in os.environ.get("ZIPF_SEEDS", "42,43,44,45").split(",")]
# TOTAL = M, so the uniform arm gives EXACTLY ONE merge per item and therefore
# reproduces the standard protocol used everywhere else in this campaign. That
# matters: a first version used 6 presentations per item, which makes the
# effective gain g^6 and put even the UNIFORM control far past the crowding wall
# (accuracy 0.219 with distinctness 0.719, i.e. the baseline was broken, not the
# Zipfian arm). The swept `gain` is then per-merge, and an item presented c
# times experiences g^c -- which is precisely the mechanism under test.
TOTAL_PRESENTATIONS = int(os.environ.get("ZIPF_PRES", "0")) or M
DEPTH = int(os.environ.get("ZIPF_DEPTH", "3"))
BUILD_ROUNDS = 6
#: Same gated merge channels as every other measurement in this campaign, so
#: the Zipfian result is comparable to the uniform phase diagram.
GATED = dict(parent_self=False, target_self=False, back_project=False)
#: Homeostatic weight scaling, the candidate frequency compensator.
SCALING = os.environ.get("ZIPF_SCALING", "0") == "1"


def zipf_counts(m_items, s, total, rng):
    """Presentation counts per item: Zipf(s) weights, fixed total, >= 1 each."""
    w = [1.0 / ((i + 1) ** s) for i in range(m_items)]
    z = sum(w)
    counts = [max(1, int(round(total * x / z))) for x in w]
    return counts


def trial(gain, skew, seed):
    """Depth-D composition with Zipfian MERGE repetition; score by tercile."""
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain
    from _substrate import pinned

    beta = gain ** (1.0 / T) - 1.0
    rng = random.Random(seed)
    # SYNAPTIC SCALING as a candidate frequency compensator. Zipfian skew was
    # measured to close the composition wedge outright (skew 1.0 peaks at 0.152
    # against 1.000 uniform), and the mechanism is cumulative potentiation
    # making an item's effective gain g^count. Any fix has to remove that
    # dependence on count. Homeostatic scaling is the standing candidate and is
    # already a Brain option, so it costs one flag to test rather than a new
    # mechanism to write.
    brain = Brain(p=P, seed=seed, synaptic_scaling=SCALING)
    brain.add_area("A", N, K, beta=beta)
    for L in range(1, DEPTH + 1):
        brain.add_area(f"P{L}", N, K, beta=beta)
        brain.add_area(f"C{L}", N, K, beta=beta)
    for i in range(M):
        brain.add_stimulus(f"a{i}", K)
        for L in range(1, DEPTH + 1):
            brain.add_stimulus(f"p{L}_{i}", K)

    # Leaf and partner assemblies are built UNIFORMLY. Frequency was measured
    # to have no effect on a feed-forward lexicon (see the module docstring),
    # so varying it here would add cost without adding signal, and would
    # confound the composition effect under test.
    for i in range(M):
        for _ in range(BUILD_ROUNDS):
            brain.project({f"a{i}": ["A"]}, {})
        for L in range(1, DEPTH + 1):
            for _ in range(BUILD_ROUNDS):
                brain.project({f"p{L}_{i}": [f"P{L}"]}, {})

    counts = zipf_counts(M, skew, TOTAL_PRESENTATIONS, rng)

    # Frequency enters HERE: an item's composition is rehearsed as often as the
    # item occurs, so its C(L-1)->C(L) fibers accumulate that many rounds of
    # potentiation. Effective gain per item ~ (1+beta)^(T * count).
    stored = {L: {} for L in range(1, DEPTH + 1)}
    order = list(range(M))
    rng.shuffle(order)          # interleave, so frequency is not recency
    for i in order:
        for _ in range(counts[i]):
            merge(brain, "A", "P1", "C1", stim_a=f"a{i}", stim_b=f"p1_{i}",
                  rounds=T, **GATED)
        stored[1][i] = read(brain, "C1")
    for L in range(2, DEPTH + 1):
        for i in order:
            with pinned(brain, f"C{L - 1}", stored[L - 1][i]):
                for _ in range(counts[i]):
                    merge(brain, f"C{L - 1}", f"P{L}", f"C{L}",
                          stim_b=f"p{L}_{i}", rounds=T,
                          unstimulated_source_mode="require-fixed", **GATED)
                stored[L][i] = read(brain, f"C{L}")

    hits = {}
    for i in range(M):
        with probe(brain):
            for _ in range(BUILD_ROUNDS):
                brain.project({f"a{i}": ["A"]}, {})
            src = "A"
            for L in range(1, DEPTH + 1):
                brain.project({}, {src: [f"C{L}"]})
                src = f"C{L}"
            live = read(brain, f"C{DEPTH}")
        best = max(stored[DEPTH], key=lambda j: similarity(live, stored[DEPTH][j]))
        hits[i] = int(best == i)

    third = max(1, M // 3)
    head = sum(hits[i] for i in range(third)) / third
    tail = sum(hits[i] for i in range(M - third, M)) / third
    overall = sum(hits.values()) / M
    uniq = len({tuple(sorted(int(x) for x in a))
                for a in stored[DEPTH].values()})
    return overall, head, tail, uniq / M, counts[0], counts[-1]


if __name__ == "__main__":
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["cut", "n", "k", "p", "M", "T", "gain", "zipf_s", "seed",
                    "depth", "level", "acc", "acc_head", "acc_tail",
                    "distinct_frac", "pres_head", "pres_tail"])
        fh.flush()

    print(f"\n  ZIPFIAN CORPUS   n={N} k={K} M={M} p={P} "
          f"total presentations={TOTAL_PRESENTATIONS}")
    print(f"  {len(SKEWS)} skews x {len(GAINS)} gains x {len(SEEDS)} seeds\n")
    print(f"  {'s':>5} {'gain':>6} {'acc':>6} {'head':>6} {'tail':>6} "
          f"{'h-t':>6} {'distinct':>9} {'pres h/t':>10}")

    for skew in SKEWS:
        for gain in GAINS:
            t0 = time.time()
            accs, heads, tails, ds = [], [], [], []
            ph = pt = 0
            for seed in SEEDS:
                a, hd, tl, dfrac, ph, pt = trial(gain, skew, seed)
                accs.append(a); heads.append(hd); tails.append(tl); ds.append(dfrac)
                w.writerow([f"zipf{'_scaled' if SCALING else ''}", N, K, P, M, T, f"{gain:.4f}",
                            f"{skew:.2f}", seed, 1, 1, f"{a:.6f}",
                            f"{hd:.6f}", f"{tl:.6f}", f"{dfrac:.6f}", ph, pt])
            fh.flush()
            mean = lambda v: sum(v) / len(v)  # noqa: E731
            print(f"  {skew:>5.2f} {gain:>6.2f} {mean(accs):>6.3f} "
                  f"{mean(heads):>6.3f} {mean(tails):>6.3f} "
                  f"{mean(heads) - mean(tails):>+6.3f} {mean(ds):>9.3f} "
                  f"{ph:>5}/{pt:<4} [{time.time() - t0:.0f}s]")
        print()
    fh.close()
    print(f"  wrote {OUT}")
