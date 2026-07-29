"""How many composed constituents fit in one area, and is it still MERGE?

WHERE THIS COMES FROM
---------------------
`merge_recurrence_channels.py` decomposed `ops.merge`'s projection map into its
three recurrent channels and swept them factorially. All three turn out to be
collapse channels when sixteen merges run through shared areas, with different
onsets:

    P  parent self  A->A, B->B   destroys the PARENTS by T=5
                                 (cue_spread 0.9431 -> 0.0518 when gated)
    S  target self  C->C         destroys the TARGET by T=10
                                 (stored_spread 0.1654 -> 0.3085 -> 0.9320)
    K  back-proj    C->A, C->B   only bites at T=20

The best cell had NO recurrence at all -- repeated stimulus-driven feed-forward
projection, T=10: acc 1.0000, fid 0.9069, cue_spread 0.0478. Against the best
cell available with recurrence on (0.9688 / 0.3640) that is a different regime,
not an improvement at the margin.

TWO THINGS THAT RESULT DOES NOT YET ESTABLISH
----------------------------------------------
1. IS IT STILL A MERGE? Every readout in this line has cued parent A. The
   calculus's criterion (Papadimitriou 2020, sec.3, quoted in `ops.merge`) is
   that the merged assembly responds to EITHER source alone. Cueing only A
   cannot tell a conjunctive binding from "C is a downstream readout of A", and
   with K gated there is genuinely no back-projection, so the reference's
   "strong two-way synaptic connectivity" is absent by construction. If B-cued
   recall is at chance, this is not merge and the result must be renamed.
2. DOES IT SCALE? Sixteen constituents in an area of n=1000 is not a phrase
   store. The interesting question for a parser is whether one area holds
   hundreds of distinct composed constituents, which is where a real grammar's
   VP inventory lives.

Both are measured here, because the second is only worth having if the first
holds.

THE LADDER
----------
M doubles 16 -> 256 at fixed n=1000, k=50. Chance falls as 1/M, so accuracy at
M=256 is a far stronger claim than the same number at M=16, and `stored_spread`
against the ~k/n = 0.05 random-pair floor says whether the assemblies are still
distinct or merely being told apart by a lucky readout.

Seeds are chosen so every cell clears MIN_TRIALS on its own: 6 seeds at M<=32,
3 above, giving 96 / 192 / 192 / 384 / 768 trials.

`ops.merge` runs alongside at T=2 -- its best setting from the factorial, not a
strawman -- so the comparison is against the library operation at its own
optimum.

PRE-REGISTERED
--------------
L1 acc_b (cue parent B alone) clears chance by a wide margin at M=16. This is
   the merge criterion and it gates everything else in the file. If it fails,
   the operation is a two-input feed-forward encoder and should be described as
   one.
L2 acc_a stays above 0.90 through M=128. The capacity claim.
L3 stored_spread stays near k/n. If accuracy holds while spread rises, the
   assemblies are crowding and rank-1 is coasting on a shrinking margin --
   report the margin, not the accuracy.
L4 The no-recurrence arm beats `ops.merge` at every M, and the GAP WIDENS with
   M, because collapse is driven by how many merges share the area.
L5 The T sweep at M=64 has an interior optimum. Too few rounds under-potentiate
   the fiber; too many should, if anything eventually collapses the target even
   feed-forward, start to crowd. If accuracy is flat in T, the operating point
   is not delicate, which is worth knowing.
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
PARENT_ROUNDS = 6
LEAF, PARTNER, TARGET = "A", "B", "C"

LADDER = (16, 32, 64, 128, 256)
T_SWEEP = (2, 5, 10, 20, 40)
SEEDS_SMALL = (42, 7, 123, 2024, 5, 99)
SEEDS_LARGE = (42, 7, 123)


def seeds_for(m):
    return SEEDS_SMALL if m <= 32 else SEEDS_LARGE


def merge_ff(brain, stim_a, stim_b, total):
    """Merge with every recurrent channel gated: stimuli drive, C receives."""
    stims = {stim_a: [LEAF], stim_b: [PARTNER]}
    for _ in range(total):
        brain.project(stims, {LEAF: [TARGET], PARTNER: [TARGET]})
    return read(brain, TARGET)


def merge_ref(brain, stim_a, stim_b, total):
    """`ops.merge` unchanged, for the baseline arm."""
    from neural_assemblies.assembly_calculus.ops import merge
    merge(brain, LEAF, PARTNER, TARGET, stim_a=stim_a, stim_b=stim_b,
          rounds=total)
    return read(brain, TARGET)


def trial(m_items, total, arm, seed):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P_, seed=seed)
    for area in (LEAF, PARTNER, TARGET):
        brain.add_area(area, N, K_, beta=BETA)
    for m in range(m_items):
        brain.add_stimulus(f"a{m}", K_)
        brain.add_stimulus(f"b{m}", K_)
    for m in range(m_items):
        build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
        build(brain, f"b{m}", PARTNER, PARENT_ROUNDS)

    do = merge_ff if arm == "ff" else merge_ref
    stored = {m: do(brain, f"a{m}", f"b{m}", total) for m in range(m_items)}

    hits_a = hits_b = 0
    fid, margins, cues = [], [], []
    for m in range(m_items):
        with probe(brain):
            cues.append(build(brain, f"a{m}", LEAF, PARENT_ROUNDS))
            brain.project({}, {LEAF: [TARGET]})
            live = read(brain, TARGET)
            hits_a += rank1(live, stored) == m
            fid.append(similarity(live, stored[m]))
            # L3: rank-1 can survive on a vanishing margin. Report it.
            sims = sorted((similarity(live, a) for a in stored.values()),
                          reverse=True)
            if len(sims) > 1 and sims[1] > 0:
                margins.append(sims[0] / sims[1])
        with probe(brain):
            # L1: the merge criterion -- respond to the OTHER parent alone.
            build(brain, f"b{m}", PARTNER, PARENT_ROUNDS)
            brain.project({}, {PARTNER: [TARGET]})
            hits_b += rank1(read(brain, TARGET), stored) == m

    return (hits_a, hits_b, m_items, statistics.mean(fid),
            statistics.mean(margins) if margins else float("nan"),
            spread(cues), spread(stored.values()))


def run(m_items, total, arm):
    res = [trial(m_items, total, arm, s) for s in seeds_for(m_items)]
    tot = sum(x[2] for x in res)
    return (sum(x[0] for x in res) / tot, sum(x[1] for x in res) / tot,
            statistics.mean(x[3] for x in res),
            statistics.mean(x[4] for x in res),
            statistics.mean(x[5] for x in res),
            statistics.mean(x[6] for x in res), tot)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  n={N} k={K_} beta={BETA}, parentT={PARENT_ROUNDS}, ONE SHARED "
          f"target area")
    print(f"  ff  = merge with all recurrent channels gated (T=10)")
    print(f"  ref = `ops.merge` unchanged at T=2, its best setting")
    print(f"  acc_a cues parent A; acc_b cues parent B -- the merge criterion")
    print(f"  margin = best stored match / second best.  Random-pair overlap "
          f"~ k/n = {K_ / N:.4f}\n")
    print(f"  {'M':>5}{'arm':>5}{'chance':>9}{'acc_a':>8}{'acc_b':>8}"
          f"{'fid':>8}{'margin':>9}{'cue_spr':>9}{'stor_spr':>10}{'n':>7}")

    lad = {}
    for m_items in LADDER:
        for arm in ("ff", "ref"):
            r = run(m_items, 10 if arm == "ff" else 2, arm)
            lad[(m_items, arm)] = r
            flag = "" if r[6] >= MIN_TRIALS else "  [UNDER-POWERED]"
            print(f"  {m_items:>5}{arm:>5}{1 / m_items:>9.4f}{r[0]:>8.4f}"
                  f"{r[1]:>8.4f}{r[2]:>8.4f}{r[3]:>9.2f}{r[4]:>9.4f}"
                  f"{r[5]:>10.4f}{r[6]:>7}{flag}")

    print(f"\n  T sweep at M=64, ff arm")
    print(f"  {'T':>5}{'acc_a':>8}{'acc_b':>8}{'fid':>8}{'margin':>9}"
          f"{'stor_spr':>10}")
    sweep = {}
    for total in T_SWEEP:
        r = run(64, total, "ff")
        sweep[total] = r
        print(f"  {total:>5}{r[0]:>8.4f}{r[1]:>8.4f}{r[2]:>8.4f}"
              f"{r[3]:>9.2f}{r[5]:>10.4f}")

    print("\n  READING")
    a16 = lad[(16, "ff")]
    l1 = a16[1] > (1 / 16) + 0.20
    print(f"    L1 responds to parent B alone (merge criterion):  {l1}   "
          f"acc_b {a16[1]:.4f} vs chance {1 / 16:.4f}")
    if not l1:
        print(f"       NOT A MERGE. With the back-projection gated the target")
        print(f"       is a two-input feed-forward encoder, and every capacity")
        print(f"       number below describes that object instead. Rename it,")
        print(f"       and re-open K as the channel the criterion needs.")

    l2 = lad[(128, "ff")][0] > 0.90
    print(f"    L2 acc_a holds to M=128:                          {l2}   "
          f"{lad[(128, 'ff')][0]:.4f}")
    l3 = lad[(128, "ff")][5] < 0.15
    print(f"    L3 assemblies stay distinct at M=128:             {l3}   "
          f"stored_spread {lad[(128, 'ff')][5]:.4f}, margin "
          f"{lad[(128, 'ff')][3]:.2f}x")
    gaps = [lad[(m, "ff")][0] - lad[(m, "ref")][0] for m in LADDER]
    l4 = all(g > 0 for g in gaps) and gaps[-1] > gaps[0]
    print(f"    L4 ff beats ops.merge everywhere and the gap widens: {l4}")
    print(f"       gaps by M: " +
          "  ".join(f"{m}:{g:+.3f}" for m, g in zip(LADDER, gaps)))
    best_t = max(T_SWEEP, key=lambda t: sweep[t][0])
    flat = max(sweep[t][0] for t in T_SWEEP) - min(
        sweep[t][0] for t in T_SWEEP) < 0.05
    print(f"    L5 T optimum at M=64: T={best_t} "
          f"({'flat -- operating point is not delicate' if flat else 'interior'})")

    print()
    if l1 and l2:
        print(f"    A SINGLE AREA HOLDS {128} COMPOSED CONSTITUENTS and returns")
        print(f"    the right one from EITHER parent alone. That is a phrase")
        print(f"    store at grammar scale, on the substrate, with no slots and")
        print(f"    no gating at read time -- the whole cost was three recurrent")
        print(f"    fibers inside the training operation.")
    elif l1:
        print(f"    Merge holds but capacity does not. The ceiling is between")
        print(f"    M={LADDER[0]} and M=128; read stored_spread and margin down")
        print(f"    the ladder to see whether it is crowding or interference,")
        print(f"    because those need different fixes.")
    else:
        print(f"    The capacity result stands on an operation that is not")
        print(f"    merge. Fix L1 before anything else here is worth quoting.")


if __name__ == "__main__":
    main()
