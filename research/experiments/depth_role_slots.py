"""Do PER-CONSTITUENT areas give each constituent its own basin?

THE CLAIM UNDER TEST
--------------------
`depth_basin_sweep.py` refuted attractor clean-up in a shared target, and the
sign of the failure is the reason: settling an eroded intermediate made it WORSE
(repair -0.2656 at mergeR=2, -0.2250 at 3), never better, at any cell. The
mechanism is that one `C->C` fiber cannot store sixteen basins -- it stores the
AREA's dominant attractor, so self-recurrence pulls every state toward that and
away from the item being held. Deepening the basin IS the collapse: across the
sweep `w(C->C)` climbed 1.04 -> 1.49 -> 9.86 while `distinct` degraded
0.073 -> 0.176 -> 0.538, one process seen in two columns.

If that diagnosis is right, the fix is structural and specific: THE ATTRACTOR
BELONGS TO THE AREA, SO GIVE EACH CONSTITUENT ITS OWN AREA. Spreading the same
sixteen constituents over S slot areas puts 16/S assemblies in each, and at
S = 16 every area holds exactly one -- a genuine per-item basin, with nothing to
be pulled toward.

This is also what the reference does. `recursive_parser.py` keeps SUBJ, OBJ,
VERB, ADJ, PREP_P and DEP_CLAUSE as separate areas time-shared by fiber gating,
and it restores outer context across a clause by REPLAYING the input rather than
by asking one area to hold and repair many constituents. Testing S directly
turns that from an appeal to the literature into a measurement.

PRE-REGISTERED
--------------
S1 `repair` rises monotonically with S and becomes POSITIVE by S = M_ITEMS. This
   is the whole claim; S=1 reproduces the shared-target failure as its control.
S2 `distinct` improves with S, because 16/S constituents per area is less
   crowding at every step.
S3 `recall` does NOT degrade with S. If it does, per-item basins cost
   discriminability and the fix trades one problem for another.
S4 The S=1 row reproduces `depth_basin_sweep.py` (repair about -0.27 at
   mergeR=2). If it does not, this harness differs from that one somewhere that
   matters and no row is comparable.

READOUT, and its honest limitation
-----------------------------------
Retrieval settles into the slot where the item was stored, then ranks the result
against ALL sixteen stored constituents across every slot -- so the
discrimination is genuinely 1-of-16 and comparable to every earlier number here.
It does NOT model slot SEARCH: a real parser would have to find the right slot
without being told, and fiber gating is how the reference does that. What is
measured here is whether per-item basins exist at all, which is the prerequisite;
slot addressing is a separate question and is not answered by this file.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, probe, read, similarity, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS, MERGE_ROUNDS = 6, 2      # mergeR=2 so a C->C fiber EXISTS at all
SETTLE_ROUNDS = 3
SLOTS = (1, 2, 4, 8, 16)
A, B = "A", "B"


def trial(n_slots, seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    slot_of = {m: m % n_slots for m in range(M_ITEMS)}
    names = [f"C{s}" for s in range(n_slots)]

    brain = Brain(p=P, seed=seed)
    for area in (A, B, *names):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)
    for m in range(M_ITEMS):
        build(brain, f"a{m}", A, PARENT_ROUNDS)
        build(brain, f"b{m}", B, PARENT_ROUNDS)

    stored = {}
    for m in range(M_ITEMS):
        tgt = names[slot_of[m]]
        merge(brain, A, B, tgt, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=MERGE_ROUNDS)
        stored[m] = read(brain, tgt)

    hits, qual, rep = 0, [], []
    for m in range(M_ITEMS):
        tgt = names[slot_of[m]]
        with probe(brain):
            build(brain, f"a{m}", A, PARENT_ROUNDS)
            brain.project({}, {A: [tgt]})
            for _ in range(MERGE_ROUNDS - 1):
                brain.project({}, {A: [tgt], tgt: [tgt]})
            live = read(brain, tgt)
            before = similarity(live, stored[m])
            # ranked against ALL items, not just this slot's occupants
            best = max((similarity(live, asm), j) for j, asm in stored.items())[1]
            hits += best == m
            qual.append(before)
            for _ in range(SETTLE_ROUNDS):
                brain.project({}, {tgt: [tgt]})
            rep.append(similarity(read(brain, tgt), stored[m]) - before)

    # distinctness measured WITHIN each slot -- constituents in different areas
    # cannot interfere, so pooling across slots would flatter the result.
    within = [spread([stored[m] for m in range(M_ITEMS)
                      if slot_of[m] == s])
              for s in range(n_slots) if sum(1 for m in range(M_ITEMS)
                                             if slot_of[m] == s) > 1]
    sp = statistics.mean(within) if within else float("nan")
    return hits / M_ITEMS, statistics.mean(qual), statistics.mean(rep), sp


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS}, settle x{SETTLE_ROUNDS}")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}); rank-1 chance {1 / M_ITEMS:.4f}\n")
    print(f"  {'slots':>6}{'items/slot':>12}{'distinct':>10}{'recall':>9}"
          f"{'quality':>9}{'repair':>9}  note")

    rows = []
    for s in SLOTS:
        res = [trial(s, sd) for sd in SEEDS]
        rec = statistics.mean(r[0] for r in res)
        qua = statistics.mean(r[1] for r in res)
        rep = statistics.mean(r[2] for r in res)
        sp = statistics.mean(r[3] for r in res)
        rows.append((s, sp, rec, qua, rep))
        note = "REPAIRS" if rep > 0.01 else ("anti-repair" if rep < -0.01 else "")
        spt = "n/a" if s == M_ITEMS else f"{sp:.4f}"
        print(f"  {s:>6}{M_ITEMS // s:>12}{spt:>10}{rec:>9.4f}{qua:>9.4f}"
              f"{rep:>+9.4f}  {note}")

    print("\n  READING")
    reps = [r[4] for r in rows]
    print(f"    S1 repair rises with slots: "
          f"{all(b >= a - 0.02 for a, b in zip(reps, reps[1:]))}  "
          f"({reps[0]:+.4f} -> {reps[-1]:+.4f})")
    print(f"    S3 recall does not degrade:  "
          f"{rows[-1][2] >= rows[0][2] - 0.05}  "
          f"({rows[0][2]:.4f} -> {rows[-1][2]:.4f})")
    if reps[-1] > 0.01:
        print("\n    PER-ITEM BASINS EXIST. One area per constituent turns")
        print("    self-recurrence from anti-repair into repair, which is the")
        print("    prerequisite depth needs. Next: re-run the depth chain with")
        print("    slot-per-constituent storage and settling at every level.")
    elif reps[-1] > reps[0] + 0.05:
        print("\n    PARTIAL. Slots help but do not reach positive repair, so")
        print("    crowding is not the whole story -- something else erodes the")
        print("    intermediate even when an area holds ONE assembly.")
    else:
        print("\n    SLOTS DO NOT HELP. Even one assembly per area fails to")
        print("    repair, so the erosion is NOT crowding. That would refute")
        print("    the diagnosis from depth_basin_sweep.py and send the search")
        print("    back to what the cue itself delivers.")


if __name__ == "__main__":
    main()
