"""Does ONE merge support recall from ONE parent? (positive control)

`compositional_depth.py` reads at chance at EVERY depth including depth 1, for
both cue types, while distinctness sits at 0.011 -- the constituents are
perfectly distinct and the readout finds nothing. Three fixes to the readout
(cue-then-settle ordering, two leaf slot areas, separate deep/recent cue paths)
each changed the numbers and none lifted depth 1 off chance.

At that point the thing to do is stop adjusting the big experiment and isolate
the single property it depends on. [PNAS20] sec 3 and `ops.merge` both state it:
the merged assembly responds to EITHER source alone. If that does not hold for
one merge in isolation, nothing built on it can be measured, and the depth
result would be reporting a broken readout as a fact about composition.

THE VARIABLE THIS SWEEPS
------------------------
Repetition. `constituent_structure.py` gets rank-1 recall of 0.781 in the
parser, and the parser merges each constituent MANY times -- once per sentence
containing it, across the whole corpus. `compositional_depth.py` merges each
constituent EXACTLY ONCE. Hebbian potentiation is multiplicative and applied per
round, so a single 2-round merge may simply not write enough weight into the
parent->target fiber for one-parent retrieval, no matter how the cue is
delivered.

That is also the hypothesis under test in the parent experiment ("trained
properly at each level"), so it deserves to be measured rather than assumed.

Swept against the other candidate, MERGE_ROUNDS, because the two trade off in
opposite directions and one value has to serve both:

    too few rounds  -> weights too weak, nothing retrievable
    too many rounds -> the shared target collapses and every constituent is the
                       same assembly (measured: 1.0000 at rounds=10)

PASS CONDITION, fixed before running: rank-1 recall above 1/M_ITEMS + 0.05 from
BOTH parents. Anything less and `compositional_depth.py` is not measuring depth.

RESULT (2026-07-28): NOTHING PASSES. THE PRIMITIVE DOES NOT HOLD HERE.
-----------------------------------------------------------------------
At 8 items x 3 seeds = 24 trials, reps=3/rounds=2 read 0.375/0.208 against
chance 0.125 and this file called it PASS. That was wrong, and it was wrong in
the way small samples usually are: 0.375 is 9 of 24. Re-run at 16 items x 6
seeds = 96 trials per cell, which lowers chance to 0.0625 AND tightens the
interval:

     reps  rounds   from A   from B   overlap
        3       2   0.0417   0.0625   0.0344
       10       2   0.0729   0.0521   0.0558
       25       2   0.0729   0.0625   0.0584
        3       3   0.0729   0.0729   0.0349
       10       3   0.0833   0.1042   0.0507
       25       3   0.0625   0.0729   0.0502

Every cell is at chance. The best, 0.1042, is 10 of 96. Repetition does not
rescue it -- 25 epochs is no better than 3 -- so the earlier reading that
"repetition is the right variable" was also an artifact of the same 24 trials.

WHAT THIS MEANS. The property being tested is merge's defining one: [PNAS20]
sec 3 and `ops.merge` both state that the merged assembly responds to EITHER
source alone. On this isolated substrate, with stimulus-driven parents and
distinct non-collapsed constituents (pairwise overlap 0.03-0.06, chance 0.05),
it does not measurably hold. That is a PRIMITIVE failure, and every compositional
result that assumes retrievability is blocked behind it.

It also puts a question on the 0.781 rank-1 figure quoted in
`parser_mixins/phrases.py`. The code that produced it is not in
research/experiments/ and I could not locate it. Three possibilities, and they
need separating before 0.781 is used as a target:
  1. the parser's full training regime supplies something this setup lacks
     (its merge sources are core areas shaped by a whole lexicon phase);
  2. it scored a DIFFERENT question -- "does the top constituent CONTAIN the
     cue parent", which with 32 constituents and 2 parents each has a much
     higher chance rate than the 0.031 quoted alongside it;
  3. it is stale.
This belongs with the back-catalogue audit (#35).
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
M_ITEMS = 8          # distinct merges to discriminate among; chance = 1/8
A, B, C = "A", "B", "C"


def trial(seed: int, reps: int, rounds: int):
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import _snap, merge, project
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, C):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)

    # Parent build rounds raised 12 -> 20. Under norm_init an assembly holds
    # only while (1+beta)^T clears the population maximum; at beta=0.10, 12
    # rounds gives 3.14, which is BELOW the measured threshold (~6 at n=1e4,
    # lower but still above 3 here). See
    # research/experiments/recurrent_assembly_decay.py. Parents that dissolve
    # cannot be cued, so every recall number below was measured against
    # assemblies that were not stable in the first place.
    for m in range(M_ITEMS):
        project(brain, f"a{m}", A, rounds=20)
        project(brain, f"b{m}", B, rounds=20)

    # Interleaved repetition: every item is merged once per epoch, so no item
    # gets all its exposure at the end. Blocked repetition would confound
    # "trained more" with "trained most recently", and recency is exactly what
    # made the first readout return one constant answer.
    stored = {}
    for _ in range(reps):
        for m in range(M_ITEMS):
            stored[m] = merge(brain, A, B, C, stim_a=f"a{m}", stim_b=f"b{m}",
                              rounds=rounds)

    def cue(m: int, from_a: bool) -> bool:
        src, stim = (A, f"a{m}") if from_a else (B, f"b{m}")
        with brain.read_only():
            # rounds=20 to match the build: a cue that does not itself clear the
            # stability threshold presents a DIFFERENT assembly than the one
            # whose synapses onto C were written, so the probe would be asking
            # the wrong question however good merge was.
            project(brain, stim, src, rounds=20)
            brain.project({}, {src: [C]})
            for _ in range(rounds - 1):
                brain.project({}, {src: [C], C: [C]})
            # `_snap` NOT `brain.areas[C].winners`. The latter holds COMPACT
            # engine indices; `stored` holds NEURON IDS (every ops.* return
            # value does). Measured: the two index spaces are DISJOINT
            # (overlap 0.0), so comparing them returns exactly chance -- which
            # is what this file previously reported, for that reason and not
            # because merge failed.
            live = np.asarray(_snap(brain, C).winners, dtype=np.int64)
        return max((overlap(live, asm), j) for j, asm in stored.items())[1] == m

    from_a = sum(cue(m, True) for m in range(M_ITEMS)) / M_ITEMS
    from_b = sum(cue(m, False) for m in range(M_ITEMS)) / M_ITEMS
    spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(stored.values(), 2))
    return from_a, from_b, spread


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chance = 1.0 / M_ITEMS
    print(f"\n  {M_ITEMS} merges, n={N} k={K}, seeds {list(SEEDS)}, "
          f"chance = {chance:.3f}")
    print(f"  PASS = both cues above {chance + 0.05:.3f}\n")
    print(f"  {'reps':>5} {'rounds':>7} {'from A':>8} {'from B':>8} "
          f"{'overlap':>8}  verdict")

    best = None
    for rounds in (1, 2, 5):
        for reps in (1, 3, 10):
            res = [trial(s, reps, rounds) for s in SEEDS]
            fa = statistics.mean(r[0] for r in res)
            fb = statistics.mean(r[1] for r in res)
            sp = statistics.mean(r[2] for r in res)
            ok = fa > chance + 0.05 and fb > chance + 0.05
            note = "PASS" if ok else ("collapsed" if sp > 0.9 else "at chance")
            print(f"  {reps:>5} {rounds:>7} {fa:>8.3f} {fb:>8.3f} "
                  f"{sp:>8.3f}  {note}")
            if ok and (best is None or fa + fb > best[0]):
                best = (fa + fb, reps, rounds)

    print()
    if best:
        print(f"  BEST: reps={best[1]} rounds={best[2]}. Use these in "
              f"compositional_depth.py before reading its depth curve.")
    else:
        print("  NO CELL PASSES. One merge does not support one-parent recall")
        print("  at any (reps, rounds) tried, so the depth experiment cannot")
        print("  measure depth and its at-chance rows are about the readout.")
        print("  Next: check whether merge writes the parent->target fiber at")
        print("  all (ops.merge documents 3.47x potentiation ONLY when the")
        print("  parents are stimulus-driven).")


if __name__ == "__main__":
    main()
