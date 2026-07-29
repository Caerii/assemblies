"""Does composition survive a SECOND level? (on the harness, at the window)

WHY THIS SUPERSEDES compositional_depth.py
-------------------------------------------
The old file read at chance at every depth INCLUDING depth 1, and its own
docstring marks it NOT INTERPRETABLE. It was measuring three defects rather than
depth: it read `brain.areas[X].winners` (compact indices) against stored
assemblies (neuron IDs), its parents were built by the non-recurrent default
path, and it trained outside the window. All three are now unwritable --
`_substrate` owns the readout, the opt-in, and the window.

Depth is worth asking only now that depth 1 is established: one-parent recall
0.9688 / 0.8854 against chance 0.0625 over 96 trials, constituents distinct at
0.0715 (`merge_recall_powered.py`, parentT=6 mergeR=1). Those are the settings
used here, unchanged, so any fall-off is about DEPTH and not about regime.

THE STRUCTURAL QUESTION, which is not about training at all
------------------------------------------------------------
Level 1 merges two STIMULUS-DRIVEN parents. Level 2 cannot: its left parent C1
is composed, has no stimulus, and can only be PINNED. `ops.merge` documents what
that costs in its own source -- a projection into a FIXED area is short-circuited
before plasticity, so the back-projection is never written (measured 0.000
fixed vs 3.47x driven). If that is what kills level 2, then this substrate
composes one level and not two for a reason that has nothing to do with depth,
capacity, or curriculum, and the fix belongs in the OPERATION.

THREE PROBES, because "depth fails" is not a location
------------------------------------------------------
    L1 = merge(A, B) -> C1        both parents driven
    L2 = merge(C1, D) -> C2       C1 pinned, D driven

  (a) cue A, read C1        does one level recall? (replicates the known result
                            inside this file, so a failure below cannot be
                            blamed on setup)
  (b) PIN true C1, read C2  does a level whose parent is COMPOSED recall, GIVEN
                            a perfect copy of that parent?
  (c) cue A, read C2        the real depth-2 path: re-derive C1 from A, then
                            reach C2 through it

PRE-REGISTERED
--------------
D1 (a) reproduces depth 1 -- recall well above chance, constituents distinct.
   If it does not, nothing below is interpretable and the regime moved.
D2 (b) is the discriminator. If it FAILS while (a) passes, the loss is
   STRUCTURAL -- the pinned-parent plasticity short-circuit -- and no amount of
   training or capacity fixes it. If it PASSES, composition is not the barrier
   and the problem (if any) is in re-deriving the parent.
D3 (c) <= (b). If (c) is much worse than (b) there is a SECOND, independent loss
   in re-deriving C1, on top of whatever (b) shows; that would be compounding
   error, which is quantitative and might yield to more items or larger n.

DISTINCTNESS IS REPORTED AT BOTH LEVELS. If level 2 collapses, its recall
numbers mean nothing regardless of what they say -- the upper wall applies to
C2 exactly as it does to C1, and C2 is a shared target too.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, pinned, probe, rank1, read, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
#: The measured window from merge_recall_powered.py. Not retuned here: reusing
#: the exact depth-1 operating point is what makes a depth-2 fall-off
#: attributable to depth.
PARENT_ROUNDS, MERGE_ROUNDS = 6, 1

A, B, D, C1, C2 = "A", "B", "D", "C1", "C2"


def trial(seed: int):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, D, C1, C2):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        for pre in ("a", "b", "d"):
            brain.add_stimulus(f"{pre}{m}", K)

    for m in range(M_ITEMS):
        for pre, area in (("a", A), ("b", B), ("d", D)):
            build(brain, f"{pre}{m}", area, PARENT_ROUNDS)

    lvl1, lvl2 = {}, {}
    for m in range(M_ITEMS):
        merge(brain, A, B, C1, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=MERGE_ROUNDS)
        lvl1[m] = read(brain, C1)
    for m in range(M_ITEMS):
        with pinned(brain, C1, lvl1[m]):
            merge(brain, C1, D, C2, stim_b=f"d{m}", rounds=MERGE_ROUNDS)
            lvl2[m] = read(brain, C2)

    def settle(src, tgt):
        brain.project({}, {src: [tgt]})
        for _ in range(MERGE_ROUNDS - 1):
            brain.project({}, {src: [tgt], tgt: [tgt]})

    a_hits = b_hits = c_hits = 0
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", A, PARENT_ROUNDS)
            settle(A, C1)
            a_hits += rank1(read(brain, C1), lvl1) == m
        with probe(brain):
            with pinned(brain, C1, lvl1[m]):
                settle(C1, C2)
                b_hits += rank1(read(brain, C2), lvl2) == m
        with probe(brain):
            build(brain, f"a{m}", A, PARENT_ROUNDS)
            settle(A, C1)
            settle(C1, C2)
            c_hits += rank1(read(brain, C2), lvl2) == m

    return a_hits, b_hits, c_hits, spread(lvl1.values()), spread(lvl2.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    a, b, c = (sum(r[i] for r in res) / trials for i in range(3))
    s1 = statistics.mean(r[3] for r in res)
    s2 = statistics.mean(r[4] for r in res)

    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS} (the measured depth-1 window)")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/probe "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}\n")
    pad = 46
    print(f"  {'probe':<{pad}}{'rank-1':>9}")
    for label, v in ((f"(a) cue A -> C1          [driven parents]", a),
                     (f"(b) PIN true C1 -> C2    [composed parent]", b),
                     (f"(c) cue A -> C1 -> C2    [full depth 2]", c)):
        flag = "" if v > chance + 0.10 else "   at chance"
        print(f"  {label:<{pad}}{v:>9.4f}{flag}")
    print(f"\n  distinctness  level1={s1:.4f}  level2={s2:.4f}  "
          f"(chance {K / N:.3f}; >0.9 = collapsed)")

    print("\n  VERDICT")
    if s1 > 0.9 or s2 > 0.9:
        print("    COLLAPSED. A shared target went past the upper wall, so the")
        print("    recall rows above are not interpretable. Lower MERGE_ROUNDS")
        print("    or PARENT_ROUNDS before reading anything else.")
    elif a <= chance + 0.10:
        print("    D1 FAILED -- depth 1 does not reproduce here, so the regime")
        print("    moved and nothing below is about depth. Re-check against")
        print("    merge_recall_powered.py before touching the depth question.")
    elif b <= chance + 0.10:
        print("    STRUCTURAL. One level composes and recalls; a level whose")
        print("    parent is COMPOSED does not, even handed a perfect copy of")
        print("    that parent. This is the _fix/plasticity short-circuit that")
        print("    ops.merge documents -- the fix belongs in the OPERATION, not")
        print("    in the curriculum. D2 confirmed.")
    elif c <= chance + 0.10:
        print("    COMPOUNDING. Each level works given a clean parent; the loss")
        print("    is in RE-DERIVING the parent, so error multiplies with")
        print("    depth. Quantitative -- more items or larger n may move it.")
    else:
        print("    DEPTH 2 HOLDS. Composition survives a second level, both")
        print("    given a perfect parent and re-derived from the leaf. The")
        print("    next question is depth 3, and whether the fall-off is")
        print("    geometric in the per-level recall rate.")


if __name__ == "__main__":
    main()
