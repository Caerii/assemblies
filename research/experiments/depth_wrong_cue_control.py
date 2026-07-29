"""Is the slotted depth result REAL, or is the cue doing no work?

THE CONTROL THIS SHOULD HAVE HAD
--------------------------------
`depth_chain_slotted.py` reported a perfectly flat chain -- full = 1.0000 at
levels 1, 2 and 3, quality ~1.0 -- against a shared-target baseline of
0.9167 -> 0.6250 -> 0.3646. It ran at mergeR=20 and its readout was TOLD which
slot to consult.

Both of those are now suspect. `slot_addressing_drift.py` showed that at
mergeR=20 the PARENT area collapses: merge back-projects into its sources on
every round, and 20 rounds x 16 merges converges all sixteen parents onto one
assembly (drift reads 0.9983 not because parents are stable but because they
have become the same parent, and addressing falls to chance even with the
merge-time rows). A slot with a strong attractor returns its own content
whatever you present. Told the slot, and with the parents collapsed, `full =
1.0000` is consistent with the cue doing NOTHING.

So the number may measure pattern completion rather than retrieval. That is not
a small caveat -- it is the difference between "composition chains to depth 3"
and "sixteen areas each return their contents on demand".

THE TEST
--------
Cue slot m with the WRONG item m' != m and read what comes out.

    sim_right   overlap(result, stored[m]) after cueing with item m
    sim_wrong   overlap(result, stored[m]) after cueing with item m'
    dependence  sim_right - sim_wrong

  dependence ~ 0    the slot returns its content regardless of input. The cue
                    is inert and every "recall" number at that setting is
                    VACUOUS.
  dependence high   the cue selects what comes out. Retrieval is real.

Run at BOTH mergeR=2 (where `slot_addressing_drift.py` measured working
addressing, 0.9896 with margin 2.02x) and mergeR=20 (the setting the depth
result used), so the contrast is measured rather than argued.

PRE-REGISTERED
--------------
W1 At mergeR=20, dependence is ~0 and the depth result is vacuous. This is the
   uncomfortable prediction and it is the reason for the file.
W2 At mergeR=2, dependence is substantial -- the cue matters.
W3 At mergeR=2 the chain still composes: full_3 clears chance by a wide margin.
   If W2 holds and W3 fails, then depth is real only where the cue is inert,
   which would mean the substrate cannot do both at once -- the sharpest
   possible version of the pattern seen all day.
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
PARENT_ROUNDS = 6
DEPTH = 3
LEAF = "A"
PARTNER = {1: "B", 2: "D", 3: "E"}
STIM_OF = {"A": "a", "B": "b", "D": "d", "E": "e"}


def slot(L, m):
    return f"C{L}_{m}"


def trial(merge_rounds, seed):
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
              stim_a=f"a{m}", stim_b=f"b{m}", rounds=merge_rounds)
        stored[1][m] = read(brain, slot(1, m))
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            with pinned(brain, slot(L - 1, m), stored[L - 1][m]):
                merge(brain, slot(L - 1, m), PARTNER[L], slot(L, m),
                      stim_b=f"{STIM_OF[PARTNER[L]]}{m}", rounds=merge_rounds)
                stored[L][m] = read(brain, slot(L, m))

    def settle(src, tgt):
        brain.project({}, {src: [tgt]})
        for _ in range(merge_rounds - 1):
            brain.project({}, {src: [tgt], tgt: [tgt]})

    full = {L: 0 for L in range(1, DEPTH + 1)}
    right, wrong = [], []
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, slot(1, m))
            live = read(brain, slot(1, m))
            right.append(similarity(live, stored[1][m]))
            for L in range(1, DEPTH + 1):
                if L > 1:
                    settle(slot(L - 1, m), slot(L, m))
                    live = read(brain, slot(L, m))
                full[L] += max((similarity(live, a), j)
                               for j, a in stored[L].items())[1] == m
        # WRONG CUE: drive slot m from a different item's leaf assembly.
        other = (m + 1) % M_ITEMS
        with probe(brain):
            build(brain, f"a{other}", LEAF, PARENT_ROUNDS)
            settle(LEAF, slot(1, m))
            wrong.append(similarity(read(brain, slot(1, m)), stored[1][m]))

    return (full, statistics.mean(right), statistics.mean(wrong))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS}, "
          f"one area per constituent, depth {DEPTH}")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  dependence = sim(right cue) - sim(WRONG cue); ~0 means the cue "
          f"is inert\n")
    print(f"  {'mergeR':>7}{'full_1':>9}{'full_2':>9}{'full_3':>9}"
          f"{'sim_right':>11}{'sim_wrong':>11}{'dependence':>12}  verdict")

    out = {}
    for mr in (2, 20):
        res = [trial(mr, s) for s in SEEDS]
        full = {L: sum(r[0][L] for r in res) / trials
                for L in range(1, DEPTH + 1)}
        sr = statistics.mean(r[1] for r in res)
        sw = statistics.mean(r[2] for r in res)
        dep = sr - sw
        out[mr] = (full, dep)
        verdict = "VACUOUS (cue inert)" if dep < 0.10 else "cue matters"
        print(f"  {mr:>7}{full[1]:>9.4f}{full[2]:>9.4f}{full[3]:>9.4f}"
              f"{sr:>11.4f}{sw:>11.4f}{dep:>12.4f}  {verdict}")

    print("\n  READING")
    f2, d2 = out[2]
    f20, d20 = out[20]
    print(f"    W1 mergeR=20 is vacuous:            {d20 < 0.10}   "
          f"(dependence {d20:.4f})")
    print(f"    W2 mergeR=2 cue matters:            {d2 >= 0.10}   "
          f"(dependence {d2:.4f})")
    print(f"    W3 mergeR=2 still composes to 3:    "
          f"{f2[3] > chance + 0.20}   (full_3 {f2[3]:.4f})")
    if d2 >= 0.10 and f2[3] > chance + 0.20:
        print("\n    DEPTH IS REAL at mergeR=2: the cue selects the output AND")
        print("    the chain composes to level 3. This is the operating point")
        print("    where addressing also worked (0.9896), so one setting")
        print("    satisfies all three demands.")
    elif d20 < 0.10 and f20[3] > f2[3]:
        print("\n    THE FLAT CHAIN WAS VACUOUS. Depth 'worked' only where the")
        print("    cue is inert -- the slots were returning their contents, not")
        print("    retrieving them. The mergeR=20 depth result is WITHDRAWN.")
    else:
        print("\n    Mixed. Read the columns directly; neither clean story holds.")


if __name__ == "__main__":
    main()
