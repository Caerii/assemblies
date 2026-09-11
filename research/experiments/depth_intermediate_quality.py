"""Are re-derived intermediates CORRECT-BUT-ERODED? (the ceiling on clean-up)

THE DECISION THIS MAKES
-----------------------
`compositional_depth_chain.py` found that composition given a perfect parent is
EXACT at every depth (step_2 = step_3 = 1.0000) while the full chain from the
leaf decays 0.9167 -> 0.6250 -> 0.3646, sub-geometrically. So every bit of the
loss is in the quality of the re-derived intermediate, and the obvious remedy is
attractor CLEAN-UP: let each intermediate settle into its own basin before
projecting onward.

That remedy is only worth building if the intermediates are eroded rather than
wrong. rank-1 cannot tell the difference -- it asks "is the top match correct"
and is blind to a constituent that is the right one but half dissolved. So:

    QUALITY   overlap between the re-derived C_L and the STORED C_L
    RANK-1    whether the re-derived C_L still identifies as item m

  QUALITY high, RANK-1 high      the intermediate is fine; the loss is
                                 downstream and clean-up buys nothing.
  QUALITY falling, RANK-1 high   CORRECT BUT ERODED -- clean-up has real
                                 headroom, and its ceiling is how much of the
                                 gap between QUALITY and 1.0 an attractor can
                                 close.
  QUALITY ~ RANK-1, both falling the intermediate is genuinely the WRONG
                                 assembly by level 2, and no amount of settling
                                 recovers it -- the problem is upstream and
                                 clean-up is the wrong fix.

WHY THE INTERMEDIATES CANNOT SELF-CORRECT TODAY, stated so it can be checked
-----------------------------------------------------------------------------
At PARENT_ROUNDS=6 the potentiation factor is (1+beta)^T = 1.8. The measured
threshold for an assembly to survive its own recurrence is roughly 3-6 at these
sizes and RISES with n (`recurrent_assembly_decay.py`). So the intermediates are
not attractors at all -- they are one-shot settling states with no basin to fall
back into. That is why a degraded C1 stays degraded and hands the damage
forward.

Note this is the same lower wall that is IRRELEVANT to depth-1 recall (measured:
recall 0.9167 at parentT=1, where potentiation is 1.1). Depth 1 re-drives the
parent from its stimulus every time, so the parent never has to hold itself. A
chain has no stimulus at level 2+. The lower wall does not govern retrieval; it
governs ERROR CORRECTION, and that is exactly what depth needs.

SELF-CORRECTION IS THEREFORE MEASURED DIRECTLY TOO. After reading the eroded
intermediate, this file lets it recur on itself for a few rounds and re-measures
QUALITY. If settling RAISES quality, clean-up works on this substrate as-is; if
it lowers it, the intermediate is being pulled toward some other assembly and
the fix needs the persistence threshold cleared first.

PRE-REGISTERED
--------------
Q1 QUALITY at level 1 is well below 1.0 even though rank-1 there is ~0.92 --
   i.e. erosion is present from the very first re-derivation.
Q2 QUALITY falls FASTER than rank-1 across levels. That gap is the headroom.
Q3 Settling an eroded intermediate on its own recurrence RAISES its quality. If
   Q3 fails, clean-up cannot be bolted on at retrieval time and the constituent
   areas have to be trained above the persistence threshold first -- which
   conflicts with the shared-target upper wall and is a harder design problem.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, pinned, probe, rank1, read, similarity, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS, MERGE_ROUNDS = 6, 1
DEPTH = 3
SETTLE_ROUNDS = 3           # for the Q3 clean-up probe

LEAF = "A"
PARTNER = {1: "B", 2: "D", 3: "E"}
STIM_OF = {"A": "a", "B": "b", "D": "d", "E": "e"}
LEVEL = {L: f"C{L}" for L in range(1, DEPTH + 1)}


def trial(seed: int):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in list(STIM_OF) + list(LEVEL.values()):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        for pre in STIM_OF.values():
            brain.add_stimulus(f"{pre}{m}", K)
    for m in range(M_ITEMS):
        for area, pre in STIM_OF.items():
            build(brain, f"{pre}{m}", area, PARENT_ROUNDS)

    stored = {}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER[1], LEVEL[1],
              stim_a=f"a{m}", stim_b=f"b{m}", rounds=MERGE_ROUNDS)
        stored.setdefault(1, {})[m] = read(brain, LEVEL[1])
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            with pinned(brain, LEVEL[L - 1], stored[L - 1][m]):
                merge(brain, LEVEL[L - 1], PARTNER[L], LEVEL[L],
                      stim_b=f"{STIM_OF[PARTNER[L]]}{m}", rounds=MERGE_ROUNDS,
                      unstimulated_source_mode="require-fixed")
                stored.setdefault(L, {})[m] = read(brain, LEVEL[L])

    def settle(src, tgt):
        brain.project({}, {src: [tgt]})
        for _ in range(MERGE_ROUNDS - 1):
            brain.project({}, {src: [tgt], tgt: [tgt]})

    qual = {L: [] for L in LEVEL}
    hits = {L: 0 for L in LEVEL}
    clean = {L: [] for L in LEVEL}
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, LEVEL[1])
            for L in range(1, DEPTH + 1):
                if L > 1:
                    settle(LEVEL[L - 1], LEVEL[L])
                live = read(brain, LEVEL[L])
                qual[L].append(similarity(live, stored[L][m]))
                hits[L] += rank1(live, stored[L]) == m
        # Q3: does self-recurrence REPAIR an eroded intermediate? Measured in a
        # separate probe so the clean-up cannot contaminate the chain above.
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, LEVEL[1])
            before = similarity(read(brain, LEVEL[1]), stored[1][m])
            for _ in range(SETTLE_ROUNDS):
                brain.project({}, {LEVEL[1]: [LEVEL[1]]})
            after = similarity(read(brain, LEVEL[1]), stored[1][m])
            clean[1].append(after - before)

    return (
        {L: statistics.mean(qual[L]) for L in LEVEL},
        {L: hits[L] / M_ITEMS for L in LEVEL},
        statistics.mean(clean[1]),
        {L: spread(stored[L].values()) for L in LEVEL},
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    trials = M_ITEMS * len(SEEDS)
    qual = {L: statistics.mean(r[0][L] for r in res) for L in LEVEL}
    rk = {L: statistics.mean(r[1][L] for r in res) for L in LEVEL}
    repair = statistics.mean(r[2] for r in res)
    sp = {L: statistics.mean(r[3][L] for r in res) for L in LEVEL}

    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS}, depth {DEPTH}")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials "
          f"(MIN_TRIALS={MIN_TRIALS}); rank-1 chance {1 / M_ITEMS:.4f}, "
          f"overlap chance {K / N:.3f}\n")
    print(f"  {'level':>6}{'QUALITY (overlap)':>20}{'RANK-1':>10}"
          f"{'gap':>8}{'distinct':>11}")
    for L in range(1, DEPTH + 1):
        print(f"  {L:>6}{qual[L]:>20.4f}{rk[L]:>10.4f}"
              f"{rk[L] - qual[L]:>8.4f}{sp[L]:>11.4f}")

    print(f"\n  Q3  self-recurrence x{SETTLE_ROUNDS} on the eroded level-1 "
          f"intermediate: quality {repair:+.4f}")

    print("\n  READING")
    eroded = qual[1] < 0.9 and rk[1] > 0.5
    faster = (rk[1] - qual[1]) < (rk[DEPTH] - qual[DEPTH])
    print(f"    Q1 eroded from the first re-derivation:  {eroded}")
    print(f"    Q2 quality falls faster than rank-1:     {faster}")
    print(f"    Q3 settling repairs the intermediate:    {repair > 0.01}")
    if eroded and repair > 0.01:
        print("\n    CLEAN-UP IS THE RIGHT FIX and works as-is: intermediates")
        print("    are correct-but-eroded and self-recurrence repairs them.")
        print("    Next: insert settling at every level and re-run the chain.")
    elif eroded:
        print("\n    Clean-up has HEADROOM but self-recurrence does not deliver")
        print("    it -- the intermediates are not attractors at this training")
        print("    level ((1+b)^T = %.1f). The constituent areas must clear the"
              % ((1 + BETA) ** PARENT_ROUNDS))
        print("    persistence threshold first, which fights the shared-target")
        print("    upper wall. That is the real design problem.")
    else:
        print("\n    NOT erosion. The intermediate is not merely degraded, so")
        print("    clean-up is the wrong fix and the loss is upstream.")


if __name__ == "__main__":
    main()
