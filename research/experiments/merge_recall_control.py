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
    from neural_assemblies.assembly_calculus.ops import merge, project
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, C):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)

    for m in range(M_ITEMS):
        project(brain, f"a{m}", A, rounds=12)
        project(brain, f"b{m}", B, rounds=12)

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
            project(brain, stim, src, rounds=4)
            brain.project({}, {src: [C]})
            for _ in range(rounds - 1):
                brain.project({}, {src: [C], C: [C]})
            live = np.array(brain.areas[C].winners, dtype=np.int64)
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
