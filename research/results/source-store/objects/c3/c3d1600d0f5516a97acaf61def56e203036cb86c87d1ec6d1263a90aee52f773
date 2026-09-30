"""Settling and collapse are the same knob. Split them.

WHAT `round_resolved_read.py` FOUND, AND WHAT IT REFUTED
--------------------------------------------------------
It refuted its own R2. On a SHARED target area, identity accuracy is FLAT across
eight settling rounds (0.9688 at every round, mergeR=1). Settling does not
destroy a correct selection, so "the attractor overwrites the answer" -- the
mechanism the two-phase read was built on -- is not what happened.

What it did find was in a control column added on a hunch:

    mergeR   cue_spread  stored_spread     acc     fid
         1       0.0612         0.0715  0.9688  0.3640
         2       0.0829         0.0731  0.9271  0.3825
         5       0.9431         0.5379  0.0625  0.7254
        20       0.9992         1.0000  0.0625  1.0000

`cue_spread` is the mean pairwise overlap of the sixteen RE-CUED PARENT
assemblies. At mergeR>=5 it is ~1.0: the sixteen parents have become ONE
assembly. At mergeR=20 so have the sixteen constituents. Accuracy is exactly
chance because there is nothing left to distinguish -- not because retrieval
failed.

So the shared area is fine. 0.9688 identity from a partial cue, fully neural:
no slots, no cross-area argmax, no stored label in the mechanism. The cap is
`fid = 0.3640` -- right item, thin content -- and the reason mergeR cannot
simply be raised to fix it is the subject of this file.

THE MECHANISM, NAMED
--------------------
`ops.merge` runs the reference's map:

    project({a: [A], b: [B]}, {A: [A, C], B: [B, C], C: [C, A, B]})
                                                        ^^^^^^^^^
The back-projection `C -> A, B` is not incidental -- `ops.merge` documents it as
what [PNAS20] means by "strong two-way synaptic connectivity", and measures the
parent potentiation it creates (3.47x driven, 0.000 with fixed parents). It is
what makes merge a BINDING operation rather than a downstream readout.

It is also the channel that destroys the parents. Sixteen merges into one shared
target, each running T rounds of `C -> A`, drag A toward whatever C currently
holds; and because the C's are themselves converging, they drag A toward a
COMMON assembly. The operation erodes its own inputs.

One knob currently controls both. `rounds` sets how long C settles (via C -> C,
which raises fidelity) AND how long C rewrites its parents (via C -> A,B, which
destroys distinctness). Every result in this line has been read off a diagonal
through a two-dimensional space.

THE DESIGN
----------
Two independent parameters:

    T   total rounds. Controls C -> C. Expected to raise fidelity.
    B   how many of those rounds keep C -> A,B OPEN. Controls the back-
        projection, and therefore the collapse. B <= T.

B is implemented by gating a fiber, which is what `core/inhibition.py` says
gating is FOR: it decides which projections happen, rather than suppressing
their results afterwards. B = T reproduces `ops.merge` exactly and is the
control arm; B = 0 is a pure feed-forward merge with no two-way connectivity.

The projection loop is written out here rather than added to `ops.merge` as a
parameter. The library op is under test, and a result that required editing it
would be hard to trust; if this wins, promoting B to a real argument is a
separate, reviewable change.

WHAT THE FOUR OUTCOMES MEAN
---------------------------
Two spreads separate two collapse channels that have never been told apart:

    cue_spread     parent collapse. If it tracks B and not T, `C -> A,B` is the
                   source-collapse channel, and gating it is the fix.
    stored_spread  target collapse. If it tracks T and not B, that is the
                   shared-target training-window law (`(1+beta)^T` above the
                   upper wall collapses onto whatever was stored first) acting
                   through `C -> C`, which is a SEPARATE problem needing a
                   separate fix.

PRE-REGISTERED
--------------
M1 cue_spread rises with B and is near chance (~k/n = 0.05) at B = 0, at every
   T. This is the mechanism claim: the back-projection is the source-collapse
   channel.
M2 fid rises with T at fixed B = 0. Settling is what buys content, and it is
   affordable once it is no longer coupled to collapse.
M3 There exists a cell with acc > 0.90 AND fid > 0.50 -- strictly better than
   anything on the diagonal, where 0.9688/0.3640 and 0.0625/1.0000 were the
   only options. This is the point of the file.
M4 If stored_spread rises with T even at B = 0, target collapse is real and
   independent, M3 will fail at large T, and the next fix is the upper wall of
   the training window on a shared target -- not the back-projection.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, probe, rank1, read, similarity, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS = 6
READ_SETTLE = 4
LEAF, PARTNER, TARGET = "A", "B", "C"

#: (T, B) cells. B = T reproduces `ops.merge`; B = 0 gates the fiber shut.
GRID = [(1, 0), (2, 0), (2, 2), (5, 0), (5, 1), (5, 5),
        (10, 0), (10, 1), (20, 0), (20, 1), (20, 20)]


def merge_gated(brain, stim_a, stim_b, total, back):
    """`ops.merge`'s map with `C -> A,B` open for only the first *back* rounds.

    Round 1 is feed-forward into the target in the reference protocol, so the
    back-projection can only run on rounds 2..total; `back` counts those.
    """
    stims = {stim_a: [LEAF], stim_b: [PARTNER]}
    brain.project(stims, {LEAF: [LEAF, TARGET], PARTNER: [PARTNER, TARGET]})
    for r in range(1, total):
        tgt = [TARGET] + ([LEAF, PARTNER] if r <= back else [])
        brain.project(stims, {LEAF: [LEAF, TARGET],
                              PARTNER: [PARTNER, TARGET],
                              TARGET: tgt})
    return read(brain, TARGET)


def trial(total, back, seed):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (LEAF, PARTNER, TARGET):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)
    for m in range(M_ITEMS):
        build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
        build(brain, f"b{m}", PARTNER, PARENT_ROUNDS)

    stored = {m: merge_gated(brain, f"a{m}", f"b{m}", total, back)
              for m in range(M_ITEMS)}

    hits_ff = hits_st = 0
    fid_ff, fid_st, cues = [], [], []
    for m in range(M_ITEMS):
        with probe(brain):
            cues.append(build(brain, f"a{m}", LEAF, PARENT_ROUNDS))
            brain.project({}, {LEAF: [TARGET]})
            live = read(brain, TARGET)
            hits_ff += rank1(live, stored) == m
            fid_ff.append(similarity(live, stored[m]))
            for _ in range(READ_SETTLE):
                brain.project({}, {LEAF: [TARGET], TARGET: [TARGET]})
            live = read(brain, TARGET)
            hits_st += rank1(live, stored) == m
            fid_st.append(similarity(live, stored[m]))

    return (hits_ff, hits_st, statistics.mean(fid_ff), statistics.mean(fid_st),
            spread(cues), spread(stored.values()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS}, ONE SHARED "
          f"target, {M_ITEMS} constituents")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  T = total merge rounds (C->C).  B = rounds with C->A,B OPEN.")
    print(f"  B = T is `ops.merge` exactly.  Random-pair overlap ~ k/n = "
          f"{K / N:.4f}\n")
    print(f"  {'T':>4}{'B':>4}{'cue_spr':>10}{'stor_spr':>10}"
          f"{'acc_ff':>9}{'fid_ff':>9}{'acc_set':>9}{'fid_set':>9}")

    rows = {}
    for total, back in GRID:
        res = [trial(total, back, s) for s in SEEDS]
        r = (sum(x[0] for x in res) / trials, sum(x[1] for x in res) / trials,
             statistics.mean(x[2] for x in res),
             statistics.mean(x[3] for x in res),
             statistics.mean(x[4] for x in res),
             statistics.mean(x[5] for x in res))
        rows[(total, back)] = r
        mark = "   <- ops.merge" if back == total and total > 1 else ""
        print(f"  {total:>4}{back:>4}{r[4]:>10.4f}{r[5]:>10.4f}"
              f"{r[0]:>9.4f}{r[2]:>9.4f}{r[1]:>9.4f}{r[3]:>9.4f}{mark}")

    print("\n  READING")
    b0 = sorted(k for k in rows if k[1] == 0)
    print(f"    M1 back-projection is the source-collapse channel")
    for total, back in b0:
        diag = rows.get((total, total))
        d = f"{diag[4]:.4f}" if diag else "   --"
        print(f"       T={total:<3} cue_spread  B=0 {rows[(total, 0)][4]:.4f}"
              f"   vs B=T {d}")
    print(f"    M2 settling buys content at B=0")
    for total, back in b0:
        print(f"       T={total:<3} fid_set {rows[(total, 0)][3]:.4f}"
              f"   acc_set {rows[(total, 0)][1]:.4f}"
              f"   stored_spread {rows[(total, 0)][5]:.4f}")

    good = {k: v for k, v in rows.items() if v[1] > 0.90 and v[3] > 0.50}
    print(f"\n    M3 a cell with acc > 0.90 AND fid > 0.50: "
          f"{'YES' if good else 'NO'}")
    if good:
        best = max(good, key=lambda k: good[k][3])
        v = good[best]
        print(f"       best T={best[0]} B={best[1]}: acc {v[1]:.4f} "
              f"fid {v[3]:.4f}, cue_spread {v[4]:.4f}")
        print(f"\n    THE KNOBS SEPARATE. Gating `C -> A,B` lets the target")
        print(f"    settle without eroding its own parents, so fidelity and")
        print(f"    distinctness stop trading off. The diagonal offered only")
        print(f"    0.9688/0.3640 or 0.0625/1.0000; off it, both hold at once.")
    else:
        big = [k for k in b0 if k[0] >= 10]
        rising = big and rows[big[-1]][5] > 0.30
        if rising:
            print(f"\n    M4. Parents survive (M1) but the TARGET still")
            print(f"    collapses at large T with B=0, so `C -> C` is a second,")
            print(f"    independent channel and the back-projection was only")
            print(f"    half the story. The next fix is the upper wall of the")
            print(f"    training window on a shared target.")
        else:
            print(f"\n    Neither channel explains the cap. Read the columns")
            print(f"    directly -- fidelity is limited by something other than")
            print(f"    collapse, and that is the finding.")


if __name__ == "__main__":
    main()
