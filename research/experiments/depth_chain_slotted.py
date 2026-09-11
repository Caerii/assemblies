"""Depth with PER-CONSTITUENT areas: is the chain near-lossless?

THE PREDICTION BEING TESTED
---------------------------
On a SHARED target the chain decayed 0.9167 -> 0.6250 -> 0.3646
(`compositional_depth_chain.py`), and every bit of that loss was erosion of the
re-derived intermediate, not failure to compose: composition given a clean
parent was EXACT at both levels (step_2 = step_3 = 1.0000).

`depth_role_slots.py` then removed the erosion. One area per constituent lifts
re-derivation quality 0.351 -> 0.741, and spending the freedom that buys --
mergeR=20, which a shared area could never afford without collapsing -- lifts it
to 1.0000 with recall 1.0000. Both changes are needed: slots alone stall at
0.741, rounds alone in a shared area collapse everything.

So the prediction is sharp and falsifiable: WITH A CLEAN PARENT AT EVERY LEVEL,
AND COMPOSITION ALREADY EXACT GIVEN A CLEAN PARENT, THE CHAIN SHOULD BE NEARLY
FLAT. 0.9167 -> 0.6250 -> 0.3646 should become roughly 1.0 -> 1.0 -> 1.0. If it
does not, something about DEPTH itself -- not erosion, not crowding -- is being
missed, and that would be the finding.

A NOTE ON THE RETIRED METRIC. `repair` is not reported here. Once quality
saturates at 1.0 settling can only move a state AWAY from perfect, so negative
repair measures perturbation rather than failure to correct; it was only
meaningful while erosion existed. Quality is the quantity that mattered
throughout.

SLOT ADDRESSING, the acknowledged gap, measured here for the first time
------------------------------------------------------------------------
Every earlier per-slot number TOLD the readout which area to look in. A parser
cannot be told -- it must find the constituent. The reference solves this with
fiber gating (only one slot's fiber is open), which is a control mechanism, not
a search. Measured here without any gating at all, as the honest lower bound:
cue the leaf, settle into EVERY slot at that level, and take the global best
match. `addressed` is that rate; `told` is the rate when handed the slot. The
gap between them is exactly what a gating mechanism would have to supply.

PRE-REGISTERED
--------------
C1 step_L = 1.0 at every level (reproduces the shared-target finding, which was
   never about the composition step).
C2 full_L is FLAT within noise -- no worse than 0.9 at level 3. This is the
   prediction; a decay here refutes the erosion account of depth.
C3 quality stays ~1.0 at every level. If quality falls at level 3 while level 1
   holds, erosion returns with depth for a reason slots do not address.
C4 `addressed` is BELOW `told`. It cannot be above; the question is how far
   below, because that gap is the size of the addressing problem and it has
   never been quantified here.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, pinned, probe, read, similarity,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS, MERGE_ROUNDS = 6, 20
DEPTH = 3

LEAF = "A"
PARTNER = {1: "B", 2: "D", 3: "E"}
STIM_OF = {"A": "a", "B": "b", "D": "d", "E": "e"}


def slot(level, m):
    return f"C{level}_{m}"


def trial(seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in STIM_OF:
        brain.add_area(area, N, K, beta=BETA)
    for L in range(1, DEPTH + 1):
        for m in range(M_ITEMS):
            brain.add_area(slot(L, m), N, K, beta=BETA)
    for m in range(M_ITEMS):
        for pre in STIM_OF.values():
            brain.add_stimulus(f"{pre}{m}", K)
    for m in range(M_ITEMS):
        for area, pre in STIM_OF.items():
            build(brain, f"{pre}{m}", area, PARENT_ROUNDS)

    stored = {L: {} for L in range(1, DEPTH + 1)}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER[1], slot(1, m),
              stim_a=f"a{m}", stim_b=f"b{m}", rounds=MERGE_ROUNDS)
        stored[1][m] = read(brain, slot(1, m))
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            prev = slot(L - 1, m)
            with pinned(brain, prev, stored[L - 1][m]):
                merge(brain, prev, PARTNER[L], slot(L, m),
                      stim_b=f"{STIM_OF[PARTNER[L]]}{m}", rounds=MERGE_ROUNDS,
                      unstimulated_source_mode="require-fixed")
                stored[L][m] = read(brain, slot(L, m))

    def settle(src, tgt):
        brain.project({}, {src: [tgt]})
        for _ in range(MERGE_ROUNDS - 1):
            brain.project({}, {src: [tgt], tgt: [tgt]})

    step = {L: 0 for L in range(1, DEPTH + 1)}
    full = {L: 0 for L in range(1, DEPTH + 1)}
    qual = {L: [] for L in range(1, DEPTH + 1)}
    addressed = 0
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, slot(1, m))
            live = read(brain, slot(1, m))
            step[1] += max((similarity(live, a), j)
                           for j, a in stored[1].items())[1] == m
        for L in range(2, DEPTH + 1):
            with probe(brain):
                with pinned(brain, slot(L - 1, m), stored[L - 1][m]):
                    settle(slot(L - 1, m), slot(L, m))
                    live = read(brain, slot(L, m))
                    step[L] += max((similarity(live, a), j)
                                   for j, a in stored[L].items())[1] == m
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, slot(1, m))
            for L in range(1, DEPTH + 1):
                if L > 1:
                    settle(slot(L - 1, m), slot(L, m))
                live = read(brain, slot(L, m))
                qual[L].append(similarity(live, stored[L][m]))
                full[L] += max((similarity(live, a), j)
                               for j, a in stored[L].items())[1] == m
        # C4: NOT told the slot. Settle the leaf into every level-1 area and
        # take the global best. No gating -- the honest lower bound.
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            best, arg = -1.0, None
            for j in range(M_ITEMS):
                settle(LEAF, slot(1, j))
                s = similarity(read(brain, slot(1, j)), stored[1][j])
                if s > best:
                    best, arg = s, j
            addressed += arg == m

    return (step, full, {L: statistics.mean(qual[L]) for L in qual},
            addressed / M_ITEMS)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    step = {L: sum(r[0][L] for r in res) / trials for L in range(1, DEPTH + 1)}
    full = {L: sum(r[1][L] for r in res) / trials for L in range(1, DEPTH + 1)}
    qual = {L: statistics.mean(r[2][L] for r in res) for L in range(1, DEPTH + 1)}
    addr = statistics.mean(r[3] for r in res)

    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS}, ONE AREA PER CONSTITUENT")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/probe "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}\n")
    print(f"  {'level':>6}{'step':>10}{'full':>10}{'quality':>10}")
    for L in range(1, DEPTH + 1):
        print(f"  {L:>6}{step[L]:>10.4f}{full[L]:>10.4f}{qual[L]:>10.4f}")

    print(f"\n  SHARED-TARGET BASELINE (compositional_depth_chain.py):")
    print(f"    full  0.9167 -> 0.6250 -> 0.3646")
    print(f"  THIS RUN:")
    print(f"    full  {full[1]:.4f} -> {full[2]:.4f} -> {full[3]:.4f}")

    print(f"\n  C4 SLOT ADDRESSING (level 1, no gating)")
    print(f"    told the slot   {full[1]:.4f}")
    print(f"    must find it    {addr:.4f}   gap {full[1] - addr:+.4f}")

    print("\n  READING")
    flat = full[DEPTH] > 0.9
    print(f"    C2 chain is flat (level 3 > 0.9):  {flat}  ({full[DEPTH]:.4f})")
    print(f"    C3 quality holds at depth:         {qual[DEPTH] > 0.9}  "
          f"({qual[DEPTH]:.4f})")
    if flat:
        print("\n    PREDICTION CONFIRMED. Depth was never the problem; erosion")
        print("    of the re-derived intermediate was, and per-constituent")
        print("    areas trained to attractor strength remove it. Composition")
        print("    chains without measurable loss to depth 3.")
    else:
        print("\n    PREDICTION REFUTED. Erosion is gone and the chain still")
        print("    decays, so something about DEPTH itself is unaccounted for.")
        print("    That is the finding, and it outranks everything above.")


if __name__ == "__main__":
    main()
