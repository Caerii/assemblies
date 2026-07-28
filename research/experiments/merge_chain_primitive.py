"""WHERE in a two-level chain does recall die? (the primitive)

`compositional_depth.py` reads at chance at every depth and
`merge_recall_control.py` showed why it cannot be trusted: even ONE merge,
recalled from ONE parent, only reaches 0.375/0.208 against chance 0.125. Before
any depth curve means anything, the loss has to be located, and "recall is bad"
is not a location.

A two-level chain has exactly three places it can be lost, and they are
separable by cueing at different points:

    L1 = merge(A, B) -> C1          both parents stimulus-driven
    L2 = merge(C1, D) -> C2         C1 has NO STIMULUS OF ITS OWN

  (a) cue A, read C1        can a merge be recalled from a parent at all?
  (b) PIN C1, read C2       can a merge whose parent is a composed assembly be
                            recalled, GIVEN a perfect copy of that parent?
  (c) cue A, read C2        the real depth case: re-derive C1 from A, then use
                            the re-derived C1 to reach C2.

The decomposition is the point. If (b) passes and (c) fails, the loss is
COMPOUNDING -- each level is individually fine and the errors multiply, which is
a quantitative problem that more repetition or a bigger n might fix. If (b)
FAILS, the loss is structural: a parent with no stimulus cannot write the
weights needed to recall its child, and no amount of training fixes it because
the mechanism never runs.

WHY (b) IS THE SUSPECT
----------------------
`ops.merge` documents the mechanism in its own source, and it is not subtle:

    "The engine SHORT-CIRCUITS a projection into a FIXED area, returning before
    plasticity is applied (see _fix). Merge is the one operation that projects
    BACK into its sources, so fixing them silently discards exactly the two-way
    connectivity merge exists to create. Measured: weight potentiation between
    the merged assembly and its parents is 0.000 with fixed parents and 3.19x
    when they are driven."

Level 1 can drive both parents from stimuli. Level 2 and above CANNOT -- a
composed constituent has no stimulus, so it must be pinned, so its
back-projection is never written. If that is what kills (b), then this substrate
can compose one level and not two, and the reason is a documented interaction
between `_fix` and plasticity rather than anything about depth or capacity.

That would make it a PRIMITIVE issue -- the thing to fix is the operation, not
the curriculum.

PRE-REGISTERED
--------------
E1 (a) passes, at roughly the 0.375 the isolated control reached.
E2 (b) FAILS, at chance, for the fixed-parent reason above.
E3 (c) fails, and no worse than (b) -- if (c) were much worse than (b) there
   would be a SECOND loss in re-deriving C1, on top of the structural one.

If E2 is refuted and (b) passes, the problem is compounding rather than
structural, and the next move is repetition and n rather than a change to merge.

Chance is 1/M_ITEMS and printed on every row.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

N, K, P, BETA = 1000, 50, 0.05, 0.10
SEEDS = (42, 7, 123)
M_ITEMS = 8
REPS, ROUNDS = 3, 2          # best cell from merge_recall_control.py

A, B, D, C1, C2 = "A", "B", "D", "C1", "C2"


def build(seed):
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, D, C1, C2):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        for pre in ("a", "b", "d"):
            brain.add_stimulus(f"{pre}{m}", K)
    for m in range(M_ITEMS):
        project(brain, f"a{m}", A, rounds=12)
        project(brain, f"b{m}", B, rounds=12)
        project(brain, f"d{m}", D, rounds=12)
    return brain


def train(brain):
    """Build both levels, interleaved across items and repeated REPS times."""
    from neural_assemblies.assembly_calculus.ops import merge

    lvl1, lvl2 = {}, {}
    for _ in range(REPS):
        for m in range(M_ITEMS):
            lvl1[m] = merge(brain, A, B, C1,
                            stim_a=f"a{m}", stim_b=f"b{m}", rounds=ROUNDS)
        for m in range(M_ITEMS):
            # C1 is a composed assembly with no stimulus, so it must be pinned.
            # This is the condition under test, not an implementation choice --
            # every level above the first is forced into it.
            brain.areas[C1].winners = np.array(lvl1[m].winners, dtype=np.int64)
            brain.areas[C1].fix_assembly()
            lvl2[m] = merge(brain, C1, D, C2, stim_b=f"d{m}", rounds=ROUNDS)
            brain.areas[C1].unfix_assembly()
    return lvl1, lvl2


def settle(brain, src, tgt):
    """Cue then settle: source-only first round, recurrence after.

    Opening with `tgt: [tgt]` lets whatever training left in the target vote for
    itself before the cue arrives, and it wins -- measured as recall pinned at
    exactly 1/M regardless of cue.
    """
    brain.project({}, {src: [tgt]})
    for _ in range(ROUNDS - 1):
        brain.project({}, {src: [tgt], tgt: [tgt]})


def score(brain, lvl1, lvl2):
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import project

    def rank1(live, table):
        return max((overlap(live, asm), m) for m, asm in table.items())[1]

    a_hits = b_hits = c_hits = 0
    for m in range(M_ITEMS):
        # (a) cue A -> C1
        with brain.read_only():
            project(brain, f"a{m}", A, rounds=4)
            settle(brain, A, C1)
            a_hits += rank1(np.array(brain.areas[C1].winners), lvl1) == m

        # (b) PIN the true C1 -> C2. Hands the level a perfect parent, so any
        # failure here cannot be blamed on the parent being reconstructed badly.
        with brain.read_only():
            brain.areas[C1].winners = np.array(lvl1[m].winners, dtype=np.int64)
            settle(brain, C1, C2)
            b_hits += rank1(np.array(brain.areas[C2].winners), lvl2) == m

        # (c) cue A -> C1 -> C2, the real depth-2 path
        with brain.read_only():
            project(brain, f"a{m}", A, rounds=4)
            settle(brain, A, C1)
            settle(brain, C1, C2)
            c_hits += rank1(np.array(brain.areas[C2].winners), lvl2) == m

    return a_hits / M_ITEMS, b_hits / M_ITEMS, c_hits / M_ITEMS


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from neural_assemblies.assembly_calculus.assembly import overlap

    chance = 1.0 / M_ITEMS
    rows, spreads = [], []
    for seed in SEEDS:
        brain = build(seed)
        lvl1, lvl2 = train(brain)
        rows.append(score(brain, lvl1, lvl2))
        spreads.append((
            statistics.mean(overlap(x, y) for x, y
                            in itertools.combinations(lvl1.values(), 2)),
            statistics.mean(overlap(x, y) for x, y
                            in itertools.combinations(lvl2.values(), 2)),
        ))

    a, b, c = (statistics.mean(r[i] for r in rows) for i in range(3))
    s1 = statistics.mean(s[0] for s in spreads)
    s2 = statistics.mean(s[1] for s in spreads)

    print(f"\n  n={N} k={K} reps={REPS} rounds={ROUNDS}, {M_ITEMS} items, "
          f"seeds {list(SEEDS)}")
    print(f"  chance = {chance:.3f}, pass = {chance + 0.05:.3f}\n")
    print(f"  {'probe':<44} {'rank-1':>8}")
    print(f"  {'(a) cue A -> C1   [driven parents]':<44} {a:>8.3f}"
          f"{'' if a > chance + 0.05 else '   at chance'}")
    print(f"  {'(b) PIN true C1 -> C2   [pinned parent]':<44} {b:>8.3f}"
          f"{'' if b > chance + 0.05 else '   at chance'}")
    print(f"  {'(c) cue A -> C1 -> C2   [full depth 2]':<44} {c:>8.3f}"
          f"{'' if c > chance + 0.05 else '   at chance'}")
    print(f"\n  distinctness: level1={s1:.3f}  level2={s2:.3f}")

    print("\n  verdict")
    if s1 > 0.9 or s2 > 0.9:
        print("    COLLAPSED -- constituents are not distinct, nothing above is")
        print("    interpretable. Fix MERGE_ROUNDS first.")
    elif a > chance + 0.05 and b <= chance + 0.05:
        print("    STRUCTURAL. One level composes and recalls; a level whose")
        print("    parent is a composed (stimulus-less) assembly does not, even")
        print("    handed a PERFECT copy of that parent. This is the _fix /")
        print("    plasticity short-circuit ops.merge documents -- the fix")
        print("    belongs in the OPERATION, not in the curriculum.")
    elif b > chance + 0.05 and c <= chance + 0.05:
        print("    COMPOUNDING. Each level works given a clean parent; the loss")
        print("    is in re-deriving the parent, so errors multiply with depth.")
        print("    That is quantitative -- more reps or larger n may fix it.")
    elif a <= chance + 0.05:
        print("    Level 1 itself does not recall, so the two-level result says")
        print("    nothing yet. Go back to merge_recall_control.py.")
    else:
        print("    All three pass. The depth harness was the problem, not the")
        print("    primitive -- re-run compositional_depth.py with these params.")


if __name__ == "__main__":
    main()
