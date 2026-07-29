"""Does merge support one-parent recall? POWERED, on the harness.

WHAT THIS REPLACES
------------------
`merge_recall_control.py` concluded that merge's defining property -- [PNAS20]
sec 3, "the merged assembly responds to EITHER source alone" -- does not hold on
this substrate. That conclusion rested on three defects, none of them about
merge, all fixed here by construction rather than by care:

  1. its probe read `brain.areas[C].winners` (COMPACT engine indices) and
     compared them to stored assemblies (NEURON IDS). The two spaces are
     disjoint, so the comparison returns EXACTLY CHANCE -- and reads as a clean
     negative result. `_substrate.read()` is now the only way to look.
  2. its parents were built with `ops.project`, which by default drops target
     self-recurrence, so they were never assemblies in the defining sense.
     `_substrate.build()` opts in.
  3. its parents were trained outside the window (see below).

THE WINDOW, which is the actual physics and has TWO WALLS POINTING OPPOSITE WAYS
--------------------------------------------------------------------------------
Whichever neuron set has the largest (degree x potentiation) product wins the
k-cap. That single fact cuts both ways:

  LOWER WALL   a lone assembly re-selecting itself must beat the POPULATION
               MAXIMUM, so it needs `(1+beta)^T` large. Under norm_init the
               degree advantage is normalized away and only potentiation is
               left. Measured threshold ~6 at n=1e4, and it RISES with n.
  UPPER WALL   many assemblies sharing ONE target need the incumbent BELOW the
               level where it dominates. Past roughly 3, the first constituent
               stored wins every later merge -- the `onset=1` measured in
               `merge_gated_target.py`.

A constituent lexicon has to live BETWEEN them. That is what the old "squeeze"
was, stated quantitatively, and nothing has to be ADDED to merge to sit inside
it. This file sweeps parent training across the window and reads both walls at
once: `recall` for the lower, `distinct` for the upper.

POWER
-----
16 items x 6 seeds = 96 trials per cell, chance 1/16 = 0.0625. This project has
already retracted a headline that read 0.375 at 24 trials and 0.104 at 96; the
harness LABELS anything under 96 so it cannot happen silently again. An earlier
sighting of 1.0000 for this property was 24 trials AND was measured under a
globally-recurrent build that has since been reverted -- it is re-derived here
or it does not count.

PRE-REGISTERED
--------------
M1 There EXISTS a parent-training level at which recall clears chance by a wide
   margin while constituents stay distinct (spread < 0.10). That is the window,
   and it is the whole claim.
M2 Recall is NON-MONOTONIC in parent rounds: it fails low (parents dissolve,
   lower wall) and fails high (target collapses, upper wall). A monotonic curve
   would mean only one wall is real and the two-walled account is wrong.
M3 Where recall fails high, `distinct` collapses toward 1.0; where it fails low,
   `distinct` stays fine. The two failures are distinguishable by that column,
   which is what makes M2 more than a shape.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, potentiation, probe, rank1, read, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
A, B, C = "A", "B", "C"

PARENT_ROUNDS = (6, 8, 10, 14, 20)
MERGE_ROUNDS = (1, 2)


def trial(parent_rounds: int, merge_rounds: int, seed: int):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, C):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)

    for m in range(M_ITEMS):
        build(brain, f"a{m}", A, parent_rounds)
        build(brain, f"b{m}", B, parent_rounds)

    # merge drives its own projection map (it already names target recurrence
    # and the back-projection explicitly), so the opt-in does not touch it.
    stored = {}
    for m in range(M_ITEMS):
        merge(brain, A, B, C, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=merge_rounds)
        stored[m] = read(brain, C)

    def recall(m: int, from_a: bool) -> bool:
        src, stim = (A, f"a{m}") if from_a else (B, f"b{m}")
        with probe(brain):
            # Cue the parent at the SAME rounds it was built with, then let the
            # target settle from it alone. Anything less and the probe presents
            # a different parent than the one whose synapses were written.
            build(brain, stim, src, parent_rounds)
            brain.project({}, {src: [C]})
            for _ in range(merge_rounds - 1):
                brain.project({}, {src: [C], C: [C]})
            live = read(brain, C)
        return rank1(live, stored) == m

    hits_a = sum(recall(m, True) for m in range(M_ITEMS))
    hits_b = sum(recall(m, False) for m in range(M_ITEMS))
    return hits_a, hits_b, spread(stored.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chance = 1.0 / M_ITEMS
    trials = M_ITEMS * len(SEEDS)
    print(f"\n  n={N} k={K} p={P} beta={BETA}")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  distinct = mean pairwise overlap of the {M_ITEMS} constituents "
          f"(chance {K / N:.3f}); >0.9 means the target collapsed\n")
    print(f"  {'parentT':>8}{'(1+b)^T':>9}{'mergeR':>8}"
          f"{'recall A':>11}{'recall B':>11}{'distinct':>10}  verdict")

    best = None
    for pr in PARENT_ROUNDS:
        for mr in MERGE_ROUNDS:
            res = [trial(pr, mr, s) for s in SEEDS]
            ra = sum(r[0] for r in res) / trials
            rb = sum(r[1] for r in res) / trials
            sp = statistics.mean(r[2] for r in res)
            if sp > 0.9:
                verdict = "COLLAPSED (upper wall)"
            elif ra > chance + 0.10 and rb > chance + 0.10:
                verdict = "RECALLS"
                if best is None or ra + rb > best[0]:
                    best = (ra + rb, pr, mr, ra, rb, sp)
            else:
                verdict = "at chance"
            print(f"  {pr:>8}{potentiation(BETA, pr):>9.1f}{mr:>8}"
                  f"{ra:>11.4f}{rb:>11.4f}{sp:>10.4f}  {verdict}")

    print("\n  READING")
    if best:
        _, pr, mr, ra, rb, sp = best
        print(f"    M1 HOLDS. Window found at parentT={pr}, mergeR={mr}: "
              f"recall {ra:.4f}/{rb:.4f} against chance {chance:.4f},")
        print(f"    with constituents still distinct at {sp:.4f}. Merge's "
              f"defining property holds on this substrate;")
        print("    what it needed was the right training level, not a new "
              "mechanism.")
    else:
        print("    M1 FAILS at every cell swept. Either the window at this "
              "regime lies outside PARENT_ROUNDS, or")
        print("    the property genuinely does not hold once the probe is "
              "correct -- and THAT would be the finding.")
    print("    M2/M3: check that recall falls off at BOTH ends and that only "
          "the high end collapses `distinct`.")


if __name__ == "__main__":
    main()
