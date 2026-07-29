"""Merge has THREE recurrent channels. Which one collapses the area?

THE TWO REFUTATIONS THAT LED HERE
----------------------------------
`round_resolved_read.py` refuted "settling destroys the selection": identity
accuracy is FLAT across eight rounds (0.9688 at every round). It also found the
real cap, in a control column added on a hunch -- `cue_spread`, the mean pairwise
overlap of the sixteen re-cued PARENT assemblies, goes to ~1.0 for mergeR >= 5.
The parents become one assembly, so nothing downstream can distinguish anything.

`merge_backproj_gated.py` then refuted the obvious culprit. Gating `C -> A,B`
shut for the entire merge left cue_spread at 0.8266 against 0.9431 with it fully
open (T=5). The back-projection -- the thing `ops.merge` documents at length as
what makes merge a binding operation -- is NOT what destroys the parents.

WHAT IS LEFT
------------
Merge's projection map has three recurrent channels, and every experiment in
this line has moved them together as one `rounds` argument:

    project({a: [A], b: [B]}, {A: [A, C], B: [B, C], C: [C, A, B]})
                                  ^^^         ^^^        ^^^  ^^^^
                                  P           P          S     K

    P  PARENT SELF-RECURRENCE   A -> A,  B -> B
    S  TARGET SELF-RECURRENCE   C -> C
    K  BACK-PROJECTION          C -> A,  C -> B     (refuted above)

P is the untested one, and it is mechanically the best candidate. Sixteen merges
run in one brain each drive A -> A with plasticity on. The first merge potentiates
its own assembly's self-connections; by the second, that potentiated assembly
competes with the incoming stimulus for A's k-WTA, and once `(1+beta)^T` clears
the population maximum it WINS. That is the upper wall of the training window --
already measured in this repo for a shared target -- applying to the PARENT areas,
because merge makes them recurrent too.

Note this is not a criticism of `ops.merge`, which faithfully ports a reference
that merges ONE pair. Running sixteen merges through shared areas is this
project's extension, and the accumulation is a property of the extension.

WHY P WOULD BE SAFE TO GATE
---------------------------
`ops.merge`'s own docstring establishes that what parents need is to be STABLE
AND WRITABLE, that `_fix` gives stability at the cost of writability (measured:
0.000 back-projection potentiation), and that driving them with stimuli gives
both. Nothing in that argument requires A -> A. The stimulus is already holding
the parent; the self-fiber adds a second, competing drive that accumulates
across merges. Gating P should therefore cost nothing and save distinctness --
which is exactly the kind of claim that has been wrong twice today, hence the
factorial rather than a single arm.

DESIGN
------
Full 2x2 over (P, K) with S always on -- S is what buys fidelity, so closing it
would confound the question with "did it settle at all" -- swept over T. Two
S-off cells are included at T=10 as the anchor for that claim.

PRE-REGISTERED
--------------
C1 cue_spread stays near chance (~k/n = 0.05) whenever P is OFF, at every T,
   and rises with T whenever P is ON. This is the mechanism claim, and it is
   sharp: P is the only factor that should matter.
C2 With P off, fid rises with T (settling still works) while acc stays high --
   the two stop trading off, which is what every result so far has been unable
   to achieve.
C3 stored_spread stays low with P off. If it does NOT -- if the target collapses
   even when its inputs stay distinct -- then S is an independent second channel
   and the fix is the shared-target training window, not the parents.
C4 K remains irrelevant, reproducing `merge_backproj_gated.py` inside a design
   that could have shown otherwise.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, probe, rank1, read, similarity, spread,
)

N, K_, P_, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS = 6
READ_SETTLE = 4
LEAF, PARTNER, TARGET = "A", "B", "C"

#: (T, parent_self, target_self, backproj). S off only as the T=10 anchor.
GRID = [(T, p, True, k) for T in (2, 5, 10, 20)
        for p, k in ((True, True), (True, False),
                     (False, True), (False, False))]
GRID += [(10, True, False, False), (10, False, False, False)]


def merge_channels(brain, stim_a, stim_b, total, p_self, t_self, back):
    """`ops.merge`'s map with each recurrent channel independently gated."""
    stims = {stim_a: [LEAF], stim_b: [PARTNER]}
    src = {LEAF: ([LEAF] if p_self else []) + [TARGET],
           PARTNER: ([PARTNER] if p_self else []) + [TARGET]}
    brain.project(stims, dict(src))
    for _ in range(1, total):
        tgt = ([TARGET] if t_self else []) + ([LEAF, PARTNER] if back else [])
        brain.project(stims, {**src, **({TARGET: tgt} if tgt else {})})
    return read(brain, TARGET)


def trial(total, p_self, t_self, back, seed):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P_, seed=seed)
    for area in (LEAF, PARTNER, TARGET):
        brain.add_area(area, N, K_, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K_)
        brain.add_stimulus(f"b{m}", K_)
    for m in range(M_ITEMS):
        build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
        build(brain, f"b{m}", PARTNER, PARENT_ROUNDS)

    stored = {m: merge_channels(brain, f"a{m}", f"b{m}", total,
                                p_self, t_self, back)
              for m in range(M_ITEMS)}

    hits_ff = hits_st = 0
    fid_ff, fid_st, cues = [], [], []
    for m in range(M_ITEMS):
        with probe(brain):
            cues.append(build(brain, f"a{m}", LEAF, PARENT_ROUNDS))
            brain.project({}, {LEAF: [TARGET]})
            live = read(brain, TARGET)
            hits_ff += rank1(live, stored) == m
            fid_ff.append(similarity(live, stored[m]))
            for _ in range(READ_SETTLE):
                brain.project({}, {LEAF: [TARGET], TARGET: [TARGET]})
            live = read(brain, TARGET)
            hits_st += rank1(live, stored) == m
            fid_st.append(similarity(live, stored[m]))

    return (hits_ff, hits_st, statistics.mean(fid_ff), statistics.mean(fid_st),
            spread(cues), spread(stored.values()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    print(f"\n  n={N} k={K_} beta={BETA}, parentT={PARENT_ROUNDS}, ONE SHARED "
          f"target, {M_ITEMS} constituents")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  P = parent self (A->A, B->B)   S = target self (C->C)   "
          f"K = back-proj (C->A,B)")
    print(f"  P=S=K=on is `ops.merge`.  Random-pair overlap ~ k/n = "
          f"{K_ / N:.4f}\n")
    print(f"  {'T':>4}{'P':>3}{'S':>3}{'K':>3}{'cue_spr':>10}{'stor_spr':>10}"
          f"{'acc_ff':>9}{'fid_ff':>9}{'acc_set':>9}{'fid_set':>9}")

    rows = {}
    for cell in GRID:
        total, p_self, t_self, back = cell
        res = [trial(*cell, s) for s in SEEDS]
        r = (sum(x[0] for x in res) / trials, sum(x[1] for x in res) / trials,
             statistics.mean(x[2] for x in res),
             statistics.mean(x[3] for x in res),
             statistics.mean(x[4] for x in res),
             statistics.mean(x[5] for x in res))
        rows[cell] = r
        tag = "  <- ops.merge" if (p_self and t_self and back) else ""
        print(f"  {total:>4}{'Y' if p_self else 'n':>3}"
              f"{'Y' if t_self else 'n':>3}{'Y' if back else 'n':>3}"
              f"{r[4]:>10.4f}{r[5]:>10.4f}{r[0]:>9.4f}{r[2]:>9.4f}"
              f"{r[1]:>9.4f}{r[3]:>9.4f}{tag}")

    print("\n  READING")
    print(f"    C1 parent self-recurrence is the collapse channel")
    for T in (2, 5, 10, 20):
        on = rows[(T, True, True, True)][4]
        off = rows[(T, False, True, True)][4]
        print(f"       T={T:<3} cue_spread  P=on {on:.4f}   P=off {off:.4f}"
              f"   delta {on - off:+.4f}")
    print(f"    C4 back-projection stays irrelevant")
    for T in (2, 5, 10, 20):
        kon = rows[(T, False, True, True)][4]
        koff = rows[(T, False, True, False)][4]
        print(f"       T={T:<3} cue_spread (P off)  K=on {kon:.4f}   "
              f"K=off {koff:.4f}   delta {kon - koff:+.4f}")

    poff = {T: rows[(T, False, True, False)] for T in (2, 5, 10, 20)}
    print(f"    C2/C3 with P and K both off")
    for T in (2, 5, 10, 20):
        v = poff[T]
        print(f"       T={T:<3} acc_set {v[1]:.4f}  fid_set {v[3]:.4f}  "
              f"cue_spr {v[4]:.4f}  stor_spr {v[5]:.4f}")

    good = {k: v for k, v in rows.items() if v[1] > 0.90 and v[3] > 0.50}
    print(f"\n    a cell with acc > 0.90 AND fid > 0.50: "
          f"{'YES' if good else 'NO'}")
    if good:
        best = max(good, key=lambda c: good[c][3])
        v = good[best]
        print(f"       best T={best[0]} P={'Y' if best[1] else 'n'} "
              f"S={'Y' if best[2] else 'n'} K={'Y' if best[3] else 'n'}: "
              f"acc {v[1]:.4f}  fid {v[3]:.4f}  cue_spr {v[4]:.4f}")
        print(f"\n    THE CHANNELS SEPARATE. One recurrent fiber inside merge")
        print(f"    was destroying the parents, and it is not the one the")
        print(f"    operation is built around. Gate it and the target can")
        print(f"    settle as long as it likes: identity and content stop")
        print(f"    trading off, which no cell on the old diagonal could do.")
    elif max(v[4] for k, v in rows.items() if not k[1]) < 0.20:
        print(f"\n    C1 HOLDS -- P is the parent-collapse channel and gating it")
        print(f"    keeps the cues distinct -- but fidelity still does not")
        print(f"    follow. Distinctness was necessary and is not sufficient,")
        print(f"    so what limits content is a THIRD thing, measured next")
        print(f"    against stored_spread to see whether the target collapses")
        print(f"    even on distinct inputs.")
    else:
        print(f"\n    C1 FAILS: the parents collapse with P gated too. Neither")
        print(f"    of merge's recurrent fibers explains it, which leaves the")
        print(f"    stimulus-driven path itself -- and that would be a claim")
        print(f"    about repeated stimulus-driven training, not about merge.")


if __name__ == "__main__":
    main()
