"""WHEN do you read? Identity and content peak at different rounds.

WHY THE SLOT ARCHITECTURE HAS TO GO
------------------------------------
`two_phase_read.py` worked (T3 = 0.9896 wrong-cue routing, against 0.0000 for a
single-phase read), but phase 1 was computed in numpy, and it is worth being
precise about WHICH parts were not neural, because the answer names the fix:

    mean W[cue_rows, stored_cols] over 16 slot areas, argmax

  * summing W over cue rows IS neural -- that is exactly the input drive the
    engine computes before k-WTA.
  * restricting to `stored_cols` is NOT: it needs the stored assembly's
    identity as an external label.
  * argmax over 16 AREAS is NOT: it is a cross-area comparison, and
    `core/inhibition.py` documents that this substrate does not support one
    (a 7x pre-k-WTA separation collapses to a ~7% margin).

Two of the three are artifacts of ONE choice: one area per constituent. Sixteen
slots FORCE a cross-area argmax. Collapse them into a single shared area and
that argmax becomes a k-WTA over neurons -- the only comparison primitive the
substrate has, and one that a projection performs for free.

So the shared target was never the mistake. What was?

WHAT THE OLD SHARED-TARGET FILES ACTUALLY DID
----------------------------------------------
Every one of them read through this:

    settle(src, tgt):  project({}, {src: [tgt]})                   # round 1
                       project({}, {src: [tgt], tgt: [tgt]}) x T-1 # rounds 2..T

Round 1 is already feed-forward-only. The selection those files needed was
therefore COMPUTED -- and then they read after round T, letting T-1 rounds of
attractor dynamics overwrite it. The failure was not that the drive never
selected. It is that nobody read while the answer was still there.

That reframes the whole line. Gating is not adding a missing computation; it is
choosing WHEN to look, and the fiber is the switch that makes the choice
expressible. This file measures the thing that claim rests on and has never
been measured: accuracy and fidelity AS A FUNCTION OF ROUND.

WHY A SHARED AREA MAKES THE CONTROL FREE
-----------------------------------------
With slots, "recall" and "addressing" were separate numbers and every recall
figure had to be paired with a wrong-cue control, because a slot returns its own
occupant whatever you present. In ONE area there is nothing to tell the readout
and nowhere for it to hide: if the read were input-independent it would return
the SAME item for all 16 cues, which scores exactly 1/16. Accuracy here IS
addressing IS input-dependence. `distinct` (how many different items the 16 cues
elicit) reports the collapse directly rather than by inference.

`cue_spread` is the control this line should have had from the start: the mean
pairwise overlap of the 16 RE-CUED leaf assemblies. If the cues are not
themselves distinct, everything downstream is at chance for a reason that has
nothing to do with the target -- which is exactly what parent drift did at
mergeR=20 (`slot_addressing_drift.py`).

MEASUREMENT vs MECHANISM. `rank1(live, stored)` compares against all 16 stored
assemblies. That is the EXPERIMENTER asking "which item did the area land on",
and it is legitimate -- the system never consults it. The line that matters is
that nothing the SYSTEM does uses a stored label, and after this file nothing
does.

PRE-REGISTERED
--------------
R1 acc(round 1) clears chance by a wide margin. The feed-forward drive selects.
R2 acc FALLS with round. This is the mechanism claim: settling destroys a
   correct answer. If acc is flat, the attractor is not the culprit and the
   two-phase read has no basis on a shared area.
R3 fid RISES with round. Completion improves content while degrading identity.
R1+R2+R3 together are the tradeoff, and they make "read early, then complete"
   a measured prescription rather than an analogy.
R4 distinct(round 1) is near 16 and falls toward 1. Collapse is what R2 is
   made of.
R5 The crossover is a real operating point: there exists r where acc is still
   high and fid has already risen above fid(1).
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
MERGE_ROUNDS = (1, 2, 5, 20)
READ_ROUNDS = 8
LEAF, PARTNER, TARGET = "A", "B", "C"


def trial(merge_rounds, seed):
    from neural_assemblies.assembly_calculus.ops import merge
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

    stored = {}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER, TARGET, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=merge_rounds)
        stored[m] = read(brain, TARGET)

    hits = {r: 0 for r in range(1, READ_ROUNDS + 1)}
    fid = {r: [] for r in range(1, READ_ROUNDS + 1)}
    picks = {r: set() for r in range(1, READ_ROUNDS + 1)}
    cues = []
    for m in range(M_ITEMS):
        with probe(brain):
            cues.append(build(brain, f"a{m}", LEAF, PARENT_ROUNDS))
            for r in range(1, READ_ROUNDS + 1):
                # Round 1 is feed-forward only; the self-fiber opens after it.
                if r == 1:
                    brain.project({}, {LEAF: [TARGET]})
                else:
                    brain.project({}, {LEAF: [TARGET], TARGET: [TARGET]})
                live = read(brain, TARGET)
                pick = rank1(live, stored)
                picks[r].add(pick)
                hits[r] += pick == m
                fid[r].append(similarity(live, stored[m]))

    return (hits, {r: statistics.mean(v) for r, v in fid.items()},
            {r: len(picks[r]) for r in picks}, spread(cues),
            spread(stored.values()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS}, ONE SHARED "
          f"target area, {M_ITEMS} constituents")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  round 1 = feed-forward only; rounds 2+ add {TARGET}->{TARGET}")
    print(f"  distinct = how many of {M_ITEMS} items the 16 cues elicit "
          f"(1 = fully collapsed)\n")

    summary = {}
    for mr in MERGE_ROUNDS:
        res = [trial(mr, s) for s in SEEDS]
        acc = {r: sum(x[0][r] for x in res) / trials
               for r in range(1, READ_ROUNDS + 1)}
        fid = {r: statistics.mean(x[1][r] for x in res)
               for r in range(1, READ_ROUNDS + 1)}
        dis = {r: statistics.mean(x[2][r] for x in res)
               for r in range(1, READ_ROUNDS + 1)}
        cue_spread = statistics.mean(x[3] for x in res)
        tgt_spread = statistics.mean(x[4] for x in res)
        summary[mr] = (acc, fid, dis, cue_spread, tgt_spread)

        print(f"  mergeR={mr}   cue_spread {cue_spread:.4f}   "
              f"stored_spread {tgt_spread:.4f}")
        print(f"  {'round':>7}{'acc':>9}{'fid':>9}{'distinct':>11}")
        for r in range(1, READ_ROUNDS + 1):
            print(f"  {r:>7}{acc[r]:>9.4f}{fid[r]:>9.4f}{dis[r]:>11.2f}")
        print()

    print("  READING")
    for mr in MERGE_ROUNDS:
        acc, fid, dis, cs, ts = summary[mr]
        a1, aT = acc[1], acc[READ_ROUNDS]
        f1, fT = fid[1], fid[READ_ROUNDS]
        r1 = a1 > chance + 0.20
        r2 = aT < a1 - 0.10
        r3 = fT > f1 + 0.05
        cross = [r for r in range(2, READ_ROUNDS + 1)
                 if acc[r] > chance + 0.20 and fid[r] > f1]
        print(f"    mergeR={mr:<3} R1 select {str(r1):<5} ({a1:.4f})   "
              f"R2 decays {str(r2):<5} ({a1:.4f}->{aT:.4f})   "
              f"R3 fid rises {str(r3):<5} ({f1:.4f}->{fT:.4f})   "
              f"R5 crossover {cross[0] if cross else 'none'}")

    best = max(MERGE_ROUNDS,
               key=lambda mr: summary[mr][0][1])
    acc, fid, dis, cs, ts = summary[best]
    print()
    if acc[1] > chance + 0.20 and acc[READ_ROUNDS] < acc[1] - 0.10:
        print(f"    THE TRADEOFF IS REAL, best at mergeR={best}. A SHARED area")
        print(f"    selects the right constituent from the feed-forward drive")
        print(f"    ({acc[1]:.4f}) and then loses it to its own recurrence")
        print(f"    ({acc[READ_ROUNDS]:.4f}). No slots, no cross-area argmax, no")
        print(f"    stored label anywhere in the mechanism -- so the two-phase")
        print(f"    read reduces to gating ONE fiber at the right moment, which")
        print(f"    is a primitive this repo already has.")
    elif acc[1] > chance + 0.20:
        print(f"    Selection works on a shared area ({acc[1]:.4f}) but does NOT")
        print(f"    decay with settling. The attractor is then not what destroyed")
        print(f"    the earlier shared-target reads, and R2's mechanism claim is")
        print(f"    refuted -- find what else differs before trusting the gate.")
    else:
        print(f"    Feed-forward selection FAILS on a shared area ({acc[1]:.4f}).")
        print(f"    Check cue_spread ({cs:.4f}) and stored_spread ({ts:.4f})")
        print(f"    first: if either is high the constituents or the cues are")
        print(f"    not distinct and this says nothing about the read.")


if __name__ == "__main__":
    main()
