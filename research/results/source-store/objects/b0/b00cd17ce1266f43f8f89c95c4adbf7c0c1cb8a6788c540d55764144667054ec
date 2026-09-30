"""Does merge's recall property hold in the REFERENCE REGIME? + weight-path map

WHY THE EARLIER "PRIMITIVE FAILURE" IS SUSPECT
-----------------------------------------------
`merge_recall_control.py` found one-parent recall at chance over 96 trials and I
filed it as a primitive failure. Then I read the reference. `merge_sim` in
`.reference/dmitropolsky-assemblies/simulations.py`:

    def merge_sim(n=100000, k=317, p=0.01, beta=0.05, max_t=50)
        b.project({"stimA":["A"]}, {})
        b.project({"stimB":["B"]}, {})
        b.project({"stimA":["A"],"stimB":["B"]}, {"A":["A","C"],"B":["B","C"]})
        for i in range(max_t-1):
            b.project({"stimA":["A"],"stimB":["B"]},
                      {"A":["A","C"],"B":["B","C"],"C":["C","A","B"]})

Our `ops.merge` is FAITHFUL to this -- same source self-recurrence
(`source_a: [source_a, target]`), same two-way feedback
(`target: [target, source_a, source_b]`), same stimulus-driven parents. The
implementation is not the difference.

The REGIME is:

                    n        k       k/n      p      beta   rounds
    reference   100000      317    0.0032   0.01     0.05       50
    what I ran    1000       50    0.0500   0.05     0.10        2

k/n is 16x denser and rounds are 25x shallower. Those are not independent
choices: I picked rounds=2 because rounds=10 collapsed the target -- but
`phrases.py` already records that collapse is a function of AREA SIZE (94 merges
at rounds=10 give pairwise overlap 0.752 at n=1000, 0.231 at n=3000, 0.002 at
n=10000). So "rounds must be small" was a consequence of n=1000, and then small
rounds starved the very weights recall needs. I tested the primitive in the one
corner where it cannot work, and the reference sets k = sqrt(n), which n=1000,
k=50 also violates (sqrt(1000) = 32).

TWO THINGS MEASURED, because behaviour alone cannot tell them apart
-------------------------------------------------------------------
1. WEIGHT PATH. After merge, sum the source->target synaptic weight landing on
   the merged assembly from parent A's assembly, and compare it to the weight
   landing on an equal-size random set of target neurons. Ratio > 1 means the
   A->C path was actually written. This separates "the weights are not there"
   from "the weights are there and the readout cannot use them" -- which is
   exactly the ambiguity every retrieval-only probe has left open.

2. RECALL. Rank-1 retrieval of the merged assembly from ONE parent, chance
   1/M_ITEMS, same as before so the numbers are comparable.

PRE-REGISTERED
--------------
R1 In the reference regime (k = sqrt(n), p=0.01, beta=0.05, rounds=50, n large)
   one-parent recall clears chance by a wide margin. If it does NOT, the
   primitive failure is real and regime was not the explanation -- and that is a
   much stronger claim than the one currently filed, because it would then hold
   at the paper's own settings.
R2 The weight ratio exceeds 1 EVERYWHERE, including the cells where recall
   fails. Merge writes the path; whether it is usable is a separate question
   governed by capacity. If instead the ratio is ~1 where recall fails, then
   merge genuinely is not writing, and the fix is in the operation.
R3 Recall improves with n at fixed k/n, and degrades as k/n rises. The
   controlling variable is capacity, not round count per se.

If R1 holds, task #42 gets downgraded from "the primitive does not hold" to "the
primitive needs its documented regime", and every experiment I ran at n=1000,
k=50 needs re-running before its result means anything.

RESULT (2026-07-28): R1 REFUTED, AND THE REASON IS THE INTERESTING PART
------------------------------------------------------------------------
    cell                  n     k     k/n  rnds  recall  w ratio  overlap
    what I ran         1000    50  0.0500     2   0.292     1.13   0.0119
    + k=sqrt(n)        1000    32  0.0320     2   0.042     1.35   0.0067
    + p, beta          1000    32  0.0320     2   0.042     1.56   0.0171
    + rounds=50        1000    32  0.0320    50   0.125     1.21   0.9903  COLLAPSED
    + n=10k           10000   100  0.0100    50   0.125     0.77   1.0000  COLLAPSED
    reference-like    30000  173  0.0058    50   0.125     0.34   1.0000  COLLAPSED

(The 0.292 in row 1 is 24-trial noise; the same cell reads chance at 96 trials.
It is left in only because it is the row every earlier experiment used.)

Moving to the reference regime does NOT rescue recall. It destroys the
constituents outright: at 50 rounds the target collapses to a single assembly
(pairwise overlap 0.99-1.00) even at n=30000, where capacity is not plausibly
the constraint.

WHY -- AND THIS IS THE THING WORTH KNOWING. Re-read what `merge_sim` actually
does: ONE pair of parents, ONE merge, into an otherwise-unused area C. It never
merges a SECOND pair into the same C. So the reference result is about a single
merged assembly CONVERGING AND STABILISING. It says nothing about many
constituents coexisting in one area, because it never asks.

Our use is the thing the reference does not cover: a constituent LEXICON, dozens
of merges sharing one target, each required to stay distinct AND remain
retrievable from its parts. Those two requirements pull in opposite directions
through the same knob:

    more rounds -> more Hebbian potentiation on parent->target, which is what
                   one-parent recall needs
    more rounds -> the shared target's recurrence pulls every constituent onto
                   the same attractor, which is what distinctness forbids

n does not buy a way out of the squeeze: n=30000 collapses just as completely as
n=1000. That is the primitive issue, stated properly, and it is a genuine gap
between the reference protocol and what a phrase lexicon requires -- not a bug
in our port, which is faithful (same source self-recurrence, same two-way
feedback, same stimulus-driven parents).

The weight column corroborates it. In the non-collapsed cells the parent->target
path is written only WEAKLY -- ratio 1.13-1.56 against a random control, where a
strong conjunctive bind should be far higher. In the collapsed cells the ratio
falls BELOW 1 (0.77, 0.34): once every constituent is the same assembly, the
"merged" cells are saturated and carry no more parent-specific weight than
random ones.

WHAT THIS DOES NOT SAY. It does not say merge is broken -- as the reference uses
it (one assembly, converging) it is fine, and universality_composition.py's
OVERLAP-STRUCTURE result stands. It says the operation lacks a mechanism for
keeping many merged assemblies separable in a shared area while writing enough
weight to retrieve them, and that mechanism has to come from somewhere:
per-constituent inhibition, a sparser target code, an explicit capacity policy,
or a different target area per constituent. Choosing among those is the next
piece of work, and it is design, not debugging.
"""

from __future__ import annotations

import itertools
import math
import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

M_ITEMS = 8
SEEDS = (42, 7, 123)
A, B, C = "A", "B", "C"

#: (n, k, p, beta, rounds). The reference row is last and is the point of the
#: file; the rows above it walk from what I ran toward it, one lever at a time,
#: so a change in behaviour can be attributed rather than guessed at.
CELLS = [
    ("what I ran",      1000,   50, 0.05, 0.10,  2),
    ("+ k=sqrt(n)",     1000,   32, 0.05, 0.10,  2),
    ("+ p, beta",       1000,   32, 0.01, 0.05,  2),
    ("+ rounds=50",     1000,   32, 0.01, 0.05, 50),
    ("+ n=10k",        10000,  100, 0.01, 0.05, 50),
    ("reference-like", 30000,  173, 0.01, 0.05, 50),
]


def weight_ratio(brain, src_area, tgt_area, src_winners, tgt_winners, rng):
    """Mean src->tgt weight onto the merged assembly vs onto a random set.

    Reads the connectome directly rather than inferring from activity. A ratio
    near 1.0 says merge did not preferentially wire this parent to this
    constituent, whatever the behavioural probe shows.
    """
    engine = brain._engine_for(brain.areas[tgt_area])
    conns = getattr(engine, "_area_conns", {})
    conn = conns.get(src_area, {}).get(tgt_area)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    rows = [int(i) for i in src_winners if int(i) < w.shape[0]]
    cols = [int(j) for j in tgt_winners if int(j) < w.shape[1]]
    if not rows or not cols:
        return float("nan")
    on = float(w[np.ix_(rows, cols)].mean())
    other = [j for j in range(w.shape[1]) if j not in set(cols)]
    if not other:
        return float("nan")
    ctrl_cols = list(rng.choice(other, size=min(len(cols), len(other)),
                                replace=False))
    off = float(w[np.ix_(rows, ctrl_cols)].mean())
    return on / off if off > 0 else float("nan")


def trial(n, k, p, beta, rounds, seed):
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import merge, project
    from neural_assemblies.core.brain import Brain

    rng = np.random.default_rng(seed)
    brain = Brain(p=p, seed=seed)
    for area in (A, B, C):
        brain.add_area(area, n, k, beta=beta)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", k)
        brain.add_stimulus(f"b{m}", k)

    parents = {}
    for m in range(M_ITEMS):
        parents[m] = project(brain, f"a{m}", A, rounds=12)
        project(brain, f"b{m}", B, rounds=12)

    stored = {}
    for m in range(M_ITEMS):
        stored[m] = merge(brain, A, B, C, stim_a=f"a{m}", stim_b=f"b{m}",
                          rounds=rounds)

    ratios = [weight_ratio(brain, A, C, parents[m].winners,
                           stored[m].winners, rng)
              for m in range(M_ITEMS)]
    ratios = [r for r in ratios if not math.isnan(r)]

    hits = 0
    for m in range(M_ITEMS):
        with brain.read_only():
            project(brain, f"a{m}", A, rounds=4)
            brain.project({}, {A: [C]})
            for _ in range(min(rounds, 10) - 1):
                brain.project({}, {A: [C], C: [C]})
            live = np.array(brain.areas[C].winners, dtype=np.int64)
        best = max((overlap(live, asm), j) for j, asm in stored.items())[1]
        hits += (best == m)

    spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(stored.values(), 2))
    return hits / M_ITEMS, (statistics.mean(ratios) if ratios else float("nan")), spread


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chance = 1.0 / M_ITEMS
    print(f"\n  {M_ITEMS} merges x {len(SEEDS)} seeds, chance = {chance:.3f}")
    print(f"  weight ratio = mean A->C weight onto the merged assembly / "
          f"onto a random set\n")
    print(f"  {'cell':<16} {'n':>6} {'k':>5} {'k/n':>7} {'rnds':>5} "
          f"{'recall':>7} {'w ratio':>8} {'overlap':>8}")

    for label, n, k, p, beta, rounds in CELLS:
        res = [trial(n, k, p, beta, rounds, s) for s in SEEDS]
        rec = statistics.mean(r[0] for r in res)
        wr = statistics.mean(r[1] for r in res)
        sp = statistics.mean(r[2] for r in res)
        note = ""
        if sp > 0.9:
            note = "  COLLAPSED"
        elif rec > chance + 0.05:
            note = "  RECALLS"
        print(f"  {label:<16} {n:>6} {k:>5} {k / n:>7.4f} {rounds:>5} "
              f"{rec:>7.3f} {wr:>8.2f} {sp:>8.4f}{note}")

    print("\n  read the two columns together: a high w ratio with chance recall")
    print("  means merge WROTE the path and the readout cannot use it (capacity);")
    print("  a ratio near 1.0 means merge never wrote it (the operation).")


if __name__ == "__main__":
    main()
