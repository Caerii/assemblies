"""How deep can composition go before the parts stop being recoverable?

THE HYPOTHESIS THIS TESTS
--------------------------
`complexity_ladder.py` asks a formal-language question by feeding symbols one at
a time into a recurrent state, which treats this substrate as an RNN. That may
be the wrong question for it. The substrate's native operation is `merge`, and
`universality_composition.py` showed merge composes well -- overlap graded by
shared parents, on abstract symbols, including for never-seen combinations.

That run also carried a warning. At arity 3, built as merge(merge(a,b),c),
sharing the SHALLOW parent moved overlap more than sharing a DEEP one (+0.0870
vs +0.0449). A constituent's contribution attenuates with how deep it sits. If
that continues, there is a depth at which a leaf is no longer recoverable from
the root, and that is where compositional learning stalls in the substrate's own
terms. It is also the substrate-native form of the complexity question rather
than a retreat from it: a context-free derivation IS a hierarchy of composed
constituents.

WHY LEFT-BRANCHING CHAINS, AND WHAT THE FIRST VERSION GOT WRONG
----------------------------------------------------------------
The first version built a balanced binary tree and read at chance at EVERY
depth, including depth 1, with distinctness 0.022 -- constituents perfectly
distinct, recall dead. That combination means the representation was fine and
the harness was not.

The bug was structural and worth stating, because it is a fact about the
substrate rather than a slip. `merge` takes two SOURCE AREAS. In a balanced
tree both children of a node live at the same level, hence in the SAME area --
and an area holds ONE assembly, so the two children cannot be active at once.
The code silently merged each child with ITSELF in sequence and called the
result their composition.

So: composing two constituents of the same level requires two areas holding
them simultaneously. Balanced trees need per-level slot buffers; the number of
live areas grows with the fan-in you want to support. LEFT-BRANCHING chains
avoid it -- each node composes the level below with a fresh LEAF, so the two
sources are always in different areas by construction. That is also exactly the
arity-3 shape that already worked, extended.

    C1 = merge(t0, t1)      C2 = merge(C1, t2)      C3 = merge(C2, t3)   ...

THE ARCHITECTURAL CONTRAST, which is the point
-----------------------------------------------
  LAYERED    a distinct area per level: L1, L2, L3, L4. Depth costs areas.

  RECURSIVE  TWO areas, alternating, reused at every level. This is what actual
             recursion needs -- a grammar does not get a fresh cortical area per
             level of embedding, and unbounded depth cannot have one. Two rather
             than one because merge's target must differ from its source.

`phrases.py` records that recurrence into a shared area merges whatever passes
through it, so RECURSIVE is predicted to degrade faster. If it does, the honest
statement is that this substrate does hierarchy but not yet recursion -- a
specific, fixable claim rather than a verdict.

WHAT IS MEASURED
----------------
M independent chains are built from disjoint leaf sequences, so at every level
there are M constituents to tell apart and CHANCE IS 1/M, printed on every row.
Rank-1 recall without that baseline is meaningless.

  deep recall    cue with the chain's FIRST leaf -- the one embedded k levels
                 below the level-k root -- and ask whether that chain's
                 constituent is rank-1. This is the headline.

  recent recall  cue with the leaf attached at the LAST step, one level down.
                 A control: it should stay high at every depth, because its
                 distance to the root never grows. If recent recall decays too,
                 the loss is general degradation with merge count, not
                 attenuation with depth, and D2 would be measuring the wrong
                 thing.

  distinctness   mean pairwise overlap among the M constituents at a level.
                 Near 1.0 means collapse and nothing above it is interpretable.

PRE-REGISTERED PREDICTIONS
--------------------------
D1 Distinctness holds under LAYERED. If it collapses there too, check
   MERGE_ROUNDS before concluding anything.
D2 Deep recall decays monotonically with depth; recent recall does not. The
   gap between the two curves IS the depth effect, separated from wear.
D3 LAYERED stays above chance to a strictly greater depth than RECURSIVE.
   The one most worth being wrong about: if RECURSIVE matches LAYERED, reusing
   areas costs nothing and the substrate is readier for recursion than the
   collapse results suggested.
D4 Deep recall is at chance by depth 4. Under k-WTA a leaf's share of the drive
   roughly halves per level, so by depth 4 it is ~1/16 -- below what 50-of-1000
   winner-take-all should resolve.

Recorded before the first run of this version.

STATUS (2026-07-28): NO DEPTH RESULT. THE READOUT DOES NOT CLEAR ITS OWN BAR.
-----------------------------------------------------------------------------
Every row of this experiment reads at chance, at every depth, for both cue
types, under both architectures -- while distinctness sits at 0.011, meaning the
constituents are perfectly distinct and the readout finds nothing in them. Three
successive fixes (cue-then-settle ordering, two leaf slot areas so merge is not
called on an area with itself, separate deep/recent cue paths) each moved the
numbers and none lifted depth 1 off chance.

`merge_recall_control.py` then isolated the single property all of this rests
on -- can ONE merge be recalled from ONE parent -- and swept the two candidate
causes. Best cell, 8 items, chance 0.125, 3 seeds:

    reps  rounds   from A   from B   overlap
       1       2    0.292    0.125     0.012   at chance
       3       2    0.375    0.208     0.015   marginal
      10       2    0.250    0.250     0.023   marginal
       x       5    <=0.083  <=0.208   0.017   at chance (over-settled)
       x       1    ~0.125   ~0.150    0.034   at chance (too weak)

So repetition IS the right variable -- one-shot merge fails, repeated merge is
the only thing that passes -- which supports the "train properly at each level"
hypothesis. But even the best cell reaches 0.375/0.208 against a 0.175 bar, on
24 trials. That is 2-6 items above chance and the A/B asymmetry is not stable
across cells. It is nothing like the 0.781 rank-1 that
`constituent_structure.py` reports inside the full parser.

THEREFORE the at-chance rows below are a fact about this harness, not about
depth, and MUST NOT be read as "composition stalls at depth 1". Publishing them
as a depth curve would be exactly the failure this project keeps catching: a
plausible table, no error raised, the wrong question answered.

WHAT TO DO NEXT, in order:
  1. Close the gap to the parser's 0.781 before measuring depth at all. The
     parser differs in more than repetition -- its merge sources are core areas
     already shaped by a full lexicon phase, and its VP accumulates alongside
     many other projections. Find which of those the isolated setup needs.
  2. Check `constituent_structure.py`'s scoring criterion. It asks whether the
     top-scoring constituent CONTAINS the cue parent; with 32 constituents and
     2 parents each, several contain any given parent, so its effective chance
     is higher than the 0.031 it quotes. If the two experiments are not scoring
     the same thing, 0.781 is not the target to aim at.
  3. Only then re-run the depth curve.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

N, K, P, BETA = 1000, 50, 0.05, 0.10
SEEDS = (42, 7, 123)

#: See universality_composition.py: at rounds=10 a shared merge target collapses
#: to one assembly and everything reads 1.0000. phrases.py uses the same value.
MERGE_ROUNDS = 2

#: M chains, each DEPTH+1 leaves long. Chance at every level is 1/M.
M_CHAINS = 8
DEPTH = 4

LEAF_A, LEAF_B = "LEAF_A", "LEAF_B"


def chain_areas(recursive: bool) -> List[str]:
    """Area holding each level's constituent, level 1..DEPTH.

    LAYERED spends one area per level. RECURSIVE alternates between two,
    which is the bounded-resources-unbounded-depth condition; two rather than
    one because merge's target must differ from its sources.
    """
    if recursive:
        return [("RA" if lv % 2 else "RB") for lv in range(1, DEPTH + 1)]
    return [f"L{lv}" for lv in range(1, DEPTH + 1)]


def build(seed: int, recursive: bool, n_leaves: int):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    # TWO leaf areas, and this is not cosmetic. merge() takes two SOURCE AREAS
    # and an area holds ONE assembly, so merge(LEAF, LEAF, ...) merges an area
    # with itself: the two leaf children can never be co-active and the level-1
    # constituent is not their composition. That was the defect behind recall
    # sitting at exactly 1/8 -- diagnosed for balanced trees in the docstring
    # above, then left in place at level 1, where it invalidated every level
    # built on top of it.
    brain.add_area(LEAF_A, N, K, beta=BETA)
    brain.add_area(LEAF_B, N, K, beta=BETA)
    for name in dict.fromkeys(chain_areas(recursive)):
        brain.add_area(name, N, K, beta=BETA)
    for i in range(n_leaves):
        brain.add_stimulus(f"t{i}", K)
    return brain


def train_chains(brain, recursive: bool):
    """Build M left-branching chains, LEVEL BY LEVEL across all chains.

    Levels are completed for every chain before the next level starts -- that is
    the 'trained properly at each level' curriculum, and it also means a level's
    constituents compete with their true siblings rather than with a half-built
    hierarchy.

    Returns {level: {chain: assembly}} and the leaf sequence per chain.
    """
    from neural_assemblies.assembly_calculus.ops import merge, project

    areas = chain_areas(recursive)
    seqs = {m: [m * (DEPTH + 1) + j for j in range(DEPTH + 1)]
            for m in range(M_CHAINS)}

    # The chain's FIRST leaf is the left child and lives in LEAF_A; every
    # subsequently attached leaf is a right child and lives in LEAF_B.
    for m in range(M_CHAINS):
        project(brain, f"t{seqs[m][0]}", LEAF_A, rounds=12)
        for t in seqs[m][1:]:
            project(brain, f"t{t}", LEAF_B, rounds=12)

    levels: Dict[int, Dict[int, object]] = {}
    for lv in range(1, DEPTH + 1):
        tgt = areas[lv - 1]
        cur: Dict[int, object] = {}
        for m in range(M_CHAINS):
            if lv == 1:
                # Both children are stimulus-driven, so merge can write the
                # back-projection (ops.merge: fixed parents give none).
                cur[m] = merge(brain, LEAF_A, LEAF_B, tgt,
                               stim_a=f"t{seqs[m][0]}", stim_b=f"t{seqs[m][1]}",
                               rounds=MERGE_ROUNDS)
            else:
                src = areas[lv - 2]
                # Left child is a composed assembly with no stimulus of its own,
                # so it is pinned; the right child is a fresh leaf and keeps its
                # stimulus.
                brain.areas[src].winners = np.array(
                    levels[lv - 1][m].winners, dtype=np.int64)
                brain.areas[src].fix_assembly()
                cur[m] = merge(brain, src, LEAF_B, tgt,
                               stim_b=f"t{seqs[m][lv]}", rounds=MERGE_ROUNDS,
                               unstimulated_source_mode="require-fixed")
                brain.areas[src].unfix_assembly()
        levels[lv] = cur
    return levels, seqs


def recall(brain, leaf: int, level: int, levels, recursive: bool,
           *, deep: bool) -> int:
    """Drive ONE leaf, propagate to *level*, return the rank-1 chain.

    Two cue paths, because the two leaf roles enter the hierarchy at different
    points. A DEEP cue is the chain's first leaf: it sits in LEAF_A and must
    propagate all the way up, LEAF_A -> L1 -> L2 -> ... A RECENT cue is the leaf
    attached at this level: it sits in LEAF_B and feeds that level directly, one
    step. The distance the two travel is the manipulated variable.

    Probed under read_only(): a recall probe that trains would make each
    successive cue easier, which is the contamination channel measured in #32.
    """
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import project

    areas = chain_areas(recursive)
    with brain.read_only():
        if deep:
            project(brain, f"t{leaf}", LEAF_A, rounds=4)
            path = [(LEAF_A if lv == 1 else areas[lv - 2], areas[lv - 1])
                    for lv in range(1, level + 1)]
        else:
            project(brain, f"t{leaf}", LEAF_B, rounds=4)
            path = [(LEAF_B, areas[level - 1])]

        for src, tgt in path:
            # CUE FIRST, THEN SETTLE. The target still holds whatever training
            # left in it, so including `tgt: [tgt]` on the opening round lets
            # that stale assembly vote for itself before the cue arrives -- and
            # it wins, being an already-settled attractor. Measured with
            # recurrence on round 1: recall was EXACTLY 1/8 at every depth, the
            # signature of returning the same constituent regardless of cue.
            brain.project({}, {src: [tgt]})
            for _ in range(MERGE_ROUNDS - 1):
                brain.project({}, {src: [tgt], tgt: [tgt]})
        live = np.array(brain.areas[areas[level - 1]].winners, dtype=np.int64)

    scored = [(overlap(live, asm), m) for m, asm in levels[level].items()]
    return max(scored)[1] if scored else -1


def pairwise(assemblies) -> float:
    from neural_assemblies.assembly_calculus.assembly import overlap

    vals = [overlap(a, b) for a, b in itertools.combinations(assemblies, 2)]
    return statistics.mean(vals) if vals else float("nan")


def run(recursive: bool) -> Dict[int, Dict[str, float]]:
    acc: Dict[int, Dict[str, List[float]]] = {
        lv: {"deep": [], "recent": [], "distinct": []}
        for lv in range(1, DEPTH + 1)
    }
    for seed in SEEDS:
        brain = build(seed, recursive, M_CHAINS * (DEPTH + 1))
        levels, seqs = train_chains(brain, recursive)
        for lv in range(1, DEPTH + 1):
            acc[lv]["distinct"].append(pairwise(list(levels[lv].values())))
            deep = [recall(brain, seqs[m][0], lv, levels, recursive,
                           deep=True) == m for m in range(M_CHAINS)]
            recent = [recall(brain, seqs[m][lv], lv, levels, recursive,
                             deep=False) == m for m in range(M_CHAINS)]
            acc[lv]["deep"].append(sum(deep) / M_CHAINS)
            acc[lv]["recent"].append(sum(recent) / M_CHAINS)
    return {lv: {k: statistics.mean(v) for k, v in d.items()}
            for lv, d in acc.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chance = 1.0 / M_CHAINS
    print(f"\n  n={N} k={K} p={P} beta={BETA} merge_rounds={MERGE_ROUNDS}, "
          f"seeds {list(SEEDS)}")
    print(f"  {M_CHAINS} left-branching chains, depth {DEPTH}, "
          f"chance = {chance:.3f} at every level\n")

    results = {}
    for recursive in (False, True):
        name = ("RECURSIVE (2 areas, alternating)" if recursive
                else "LAYERED (one area per level)")
        results[recursive] = run(recursive)
        print(f"  {name}")
        print(f"    {'depth':<6} {'deep recall':>12} {'recent recall':>14} "
              f"{'distinctness':>13}")
        for lv in range(1, DEPTH + 1):
            r = results[recursive][lv]
            mark = "" if r["deep"] > chance + 0.05 else "   <- deep at chance"
            print(f"    {lv:<6} {r['deep']:>12.3f} {r['recent']:>14.3f} "
                  f"{r['distinct']:>13.3f}{mark}")
        print()

    print("  verdicts")

    def stall(res) -> int:
        for lv in range(1, DEPTH + 1):
            if res[lv]["deep"] <= chance + 0.05:
                return lv
        return DEPTH + 1

    s_lay, s_rec = stall(results[False]), stall(results[True])
    print(f"    D3 layered deeper : layered stalls at {s_lay}, "
          f"recursive at {s_rec} -> "
          f"{'HOLDS' if s_lay > s_rec else 'REFUTED -- reusing areas costs nothing'}")
    lay = results[False]
    mono = all(lay[lv]["deep"] >= lay[lv + 1]["deep"] for lv in range(1, DEPTH))
    held = all(lay[lv]["recent"] > chance + 0.05 for lv in range(1, DEPTH + 1))
    print(f"    D2 deep decays, recent holds : decay "
          f"{'monotone' if mono else 'NOT monotone'}, recent "
          f"{'stays above chance' if held else 'ALSO falls -- general wear, not depth'}")
    print(f"    D4 deep at chance by d=4 : "
          f"{'HOLDS' if max(s_lay, s_rec) <= 4 else 'REFUTED -- still above chance'}")
    worst = max(lay[lv]["distinct"] for lv in range(1, DEPTH + 1))
    if worst > 0.9:
        print(f"    D1 WARNING: layered distinctness peaks at {worst:.3f} -- "
              f"collapsed, rows above are not interpretable")


if __name__ == "__main__":
    main()
