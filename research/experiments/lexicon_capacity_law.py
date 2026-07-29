"""How many words fit in one area, and what sets the ceiling?

WHY THE LADDER STOPPED
----------------------
`merge_capacity_ladder.py` established the merge criterion properly for the
first time -- the composed assembly is returned from EITHER parent alone, 1.0000
at M=16 -- and then hit a cliff:

     M   acc_a   acc_b     fid  margin  cue_spread
    16  1.0000  1.0000  0.9069    3.00      0.0478
    32  1.0000  1.0000  0.7941    1.62      0.0568
    64  0.0260  0.0156  0.4996    1.02      0.9486

`cue_spread` is the mean pairwise overlap of the re-cued PARENT assemblies, and
it jumps from 0.06 to 0.95 between M=32 and M=64. The parents had already
collapsed into one assembly before a single merge ran. So the ceiling is not in
composition at all -- it is in the LEXICON, and every composition, parsing and
depth result in this project sits on top of it.

That makes this the base of the ladder and worth isolating completely. No
merges here, no target area: just M stimuli driven into one area, and the
question of when the area stops being able to tell them apart.

THE HYPOTHESIS, WHICH THE LAST THREE FILES EARNED
--------------------------------------------------
`_substrate.build` passes `recurrent=True`, so forming each word runs
`A -> A` with plasticity for `parentT` rounds. `merge_recurrence_channels.py`
showed the identical fiber inside merge (channel P) taking cue_spread from
0.0518 to 0.9431 -- the first word's self-connections potentiate, and once
`(1+beta)^T` clears the population maximum that assembly wins the k-WTA against
each new stimulus. Nothing about that argument is specific to merge. It should
apply verbatim to lexicon construction, and it predicts that a FEED-FORWARD
build has no such ceiling.

This is the fourth time this session that a recurrent fiber has turned out to be
the collapse channel, and the third time the obvious culprit was not. Hence the
A/B rather than the assertion.

THE OTHER AXIS
--------------
If the ceiling survives the A/B, it is a coverage law, and the numbers already
point at one: M*k / n is 0.8 at M=16, 1.6 at M=32, 3.2 at M=64, so the break
sits where the area becomes oversubscribed. A coverage law makes a hard
prediction -- M_max is proportional to n at fixed k -- and n is swept here to
test it. A ceiling that does NOT move with n is not about coverage and needs a
different account.

WHAT IS MEASURED
----------------
    ident   overlap of the RE-CUED assembly with the one built, per word. This
            is stability: does presenting the stimulus again return the word.
    acc     rank-1 identity across all M stored words. This is discriminability,
            and it is the quantity a parser actually needs -- a lexicon whose
            words are stable but mutually indistinguishable is useless.
    spread  mean pairwise overlap, against the ~k/n random-pair floor.

ident and acc come apart in the interesting case: after collapse every word is
perfectly stable (it returns THE assembly every time) while acc sits at chance.
Reporting only ident would call that a success, which is exactly the shape of
error this line has already made twice.

PRE-REGISTERED
--------------
X1 The recurrent build has a ceiling between M=32 and M=64 at n=1000,
   reproducing the cliff in the file above from an independent path.
X2 The feed-forward build does NOT collapse at M=64. This is the mechanism
   claim. If it collapses too, recurrence is exonerated and the cause is
   repeated stimulus-driven training itself.
X3 If a ceiling remains, M_max scales with n at fixed k -- doubling n roughly
   doubles the last M that holds. This is the coverage law.
X4 ident stays HIGH across the collapse while acc falls to chance, confirming
   that stability is not the quantity that fails.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, probe, rank1, read, similarity, spread,
)

K_, P_, BETA = 50, 0.05, 0.10
PARENT_ROUNDS = 6
AREA = "L"

N_SWEEP = (1000, 2000, 4000)
M_SWEEP = (16, 32, 64, 128, 256)
SEEDS = (42, 7, 123)


def build_ff(brain, stim, area, rounds):
    """Feed-forward build: the stimulus drives, the area never drives itself."""
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read(brain, area)


def build_rec(brain, stim, area, rounds):
    """What `_substrate.build` does -- stimulus plus `area -> area`."""
    from neural_assemblies.assembly_calculus.ops import project
    project(brain, stim, area, rounds=rounds, recurrent=True)
    return read(brain, area)


def trial(n, m_words, mode, seed):
    from neural_assemblies.core.brain import Brain

    build = build_ff if mode == "ff" else build_rec
    brain = Brain(p=P_, seed=seed)
    brain.add_area(AREA, n, K_, beta=BETA)
    for m in range(m_words):
        brain.add_stimulus(f"w{m}", K_)

    stored = {m: build(brain, f"w{m}", AREA, PARENT_ROUNDS)
              for m in range(m_words)}

    hits, ident = 0, []
    for m in range(m_words):
        with probe(brain):
            live = build(brain, f"w{m}", AREA, PARENT_ROUNDS)
        hits += rank1(live, stored) == m
        ident.append(similarity(live, stored[m]))

    return hits, m_words, statistics.mean(ident), spread(stored.values())


def run(n, m_words, mode):
    res = [trial(n, m_words, mode, s) for s in SEEDS]
    tot = sum(x[1] for x in res)
    return (sum(x[0] for x in res) / tot,
            statistics.mean(x[2] for x in res),
            statistics.mean(x[3] for x in res), tot)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  k={K_} beta={BETA} p={P_}, buildT={PARENT_ROUNDS}, ONE area, "
          f"no merges")
    print(f"  rec = build with `area -> area` (what `_substrate.build` does)")
    print(f"  ff  = build feed-forward only, stimulus driving")
    print(f"  ident = re-cue matches what was built; acc = rank-1 across all M")
    print(f"  load = M*k/n, the coverage ratio\n")
    print(f"  {'n':>6}{'M':>6}{'mode':>6}{'load':>7}{'chance':>9}{'acc':>8}"
          f"{'ident':>8}{'spread':>9}{'floor':>8}{'trials':>8}")

    table = {}
    for n in N_SWEEP:
        for m_words in M_SWEEP:
            for mode in ("rec", "ff"):
                r = run(n, m_words, mode)
                table[(n, m_words, mode)] = r
                flag = "" if r[3] >= MIN_TRIALS else "  [UNDER-POWERED]"
                print(f"  {n:>6}{m_words:>6}{mode:>6}"
                      f"{m_words * K_ / n:>7.2f}{1 / m_words:>9.4f}"
                      f"{r[0]:>8.4f}{r[1]:>8.4f}{r[2]:>9.4f}"
                      f"{K_ / n:>8.4f}{r[3]:>8}{flag}")
        print()

    def ceiling(n, mode):
        """Largest M whose accuracy clears 0.90."""
        ok = [m for m in M_SWEEP if table[(n, m, mode)][0] > 0.90]
        return max(ok) if ok else 0

    print("  READING")
    print(f"    X1 the recurrent build has a ceiling, and where")
    for n in N_SWEEP:
        print(f"       n={n:<6} rec ceiling M={ceiling(n, 'rec'):<4}"
              f"  acc at M=64 {table[(n, 64, 'rec')][0]:.4f}"
              f"  spread {table[(n, 64, 'rec')][2]:.4f}")
    print(f"    X2 feed-forward is the control")
    for n in N_SWEEP:
        print(f"       n={n:<6} ff  ceiling M={ceiling(n, 'ff'):<4}"
              f"  acc at M=64 {table[(n, 64, 'ff')][0]:.4f}"
              f"  spread {table[(n, 64, 'ff')][2]:.4f}")

    ff_ceils = [ceiling(n, "ff") for n in N_SWEEP]
    rec_ceils = [ceiling(n, "rec") for n in N_SWEEP]
    print(f"    X3 does the ceiling scale with n?")
    print(f"       rec  " + "  ".join(f"n={n}:M={c}"
                                      for n, c in zip(N_SWEEP, rec_ceils)))
    print(f"       ff   " + "  ".join(f"n={n}:M={c}"
                                      for n, c in zip(N_SWEEP, ff_ceils)))

    collapsed = [(n, m) for n in N_SWEEP for m in M_SWEEP
                 if table[(n, m, "rec")][0] < 0.20]
    if collapsed:
        n, m = collapsed[0]
        v = table[(n, m, "rec")]
        print(f"    X4 stability survives the collapse: at n={n} M={m}, "
              f"ident {v[1]:.4f} while acc {v[0]:.4f}")

    print()
    # ff's ceiling is CENSORED at max(M_SWEEP): every ff cell passes, so the
    # sweep ran out of range before the capacity did. Comparing raw ceilings
    # would then score a censored tie at large n as "no difference", which is
    # the opposite of what an unfound ceiling means. Require ff to be no worse
    # everywhere and strictly better wherever rec's ceiling is measurable.
    top = max(M_SWEEP)
    never = all(ceiling(n, "ff") == top for n in N_SWEEP)
    dominates = all(ceiling(n, "ff") >= ceiling(n, "rec") for n in N_SWEEP)
    strict = [n for n in N_SWEEP if ceiling(n, "rec") < top]
    if never and dominates and strict:
        print(f"    ff CEILING NOT REACHED: every cell passes up to M={top} "
              f"(load {top * K_ / min(N_SWEEP):.1f}x). The number below is a "
              f"LOWER BOUND.")
        print(f"    RECURRENCE IS THE LEXICON CEILING. Building words")
        print(f"    feed-forward raises capacity at every n, so the limit was")
        print(f"    never the area's size -- it was that each word's self-")
        print(f"    connections outcompete the next word's stimulus. That is the")
        print(f"    same fiber that collapsed merge's parents and merge's")
        print(f"    target, found now at the base of the stack.")
    elif ff_ceils == rec_ceils and max(ff_ceils) > 0:
        print(f"    X2 REFUTED -- the ceiling is identical without recurrence,")
        print(f"    so it is a property of repeated stimulus-driven training,")
        print(f"    not of the self-fiber. Read X3: if the ceiling tracks n it")
        print(f"    is coverage and the fix is capacity; if it does not, it is")
        print(f"    interference between overlapping stimulus projections.")
    else:
        print(f"    Mixed across n. Read the table directly -- the two builds")
        print(f"    differ at some scales and not others, which no single")
        print(f"    mechanism explains and is itself the thing to chase.")


if __name__ == "__main__":
    main()
