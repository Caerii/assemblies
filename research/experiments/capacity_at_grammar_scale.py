"""How many composed constituents survive, and does the MARGIN survive with them?

WHY THE LADDER IS WORTH RE-RUNNING
-----------------------------------
`merge_capacity_ladder.py` hit a cliff at M=64 and the cliff was an artifact:
the lexicon was collapsing before a single merge ran (`cue_spread` 0.0568 at
M=32, 0.9486 at M=64). Both causes are now fixed and both fixes are in the
library or measured:

  * the lexicon is built FEED-FORWARD (`lexicon_capacity_law.py`: 1.0000 to
    M=256 at n=1000, 12.8x oversubscription, overlap 0.0510 vs floor 0.0500);
  * `ops.merge` takes `parent_self` / `target_self` / `back_project`, and
    `depth_corrected_substrate.py` found the operating point -- T=2-3, where
    depth 3 is flat at 1.0000 with margin 4.58x.

That old ladder also ran at T=10, which is now known to be the wrong side of
the overlap-amplification threshold. So every number in it was measured at a
setting that fails for a reason unrelated to capacity.

WHAT THE READOUT HAS TO BE THIS TIME
-------------------------------------
MARGIN, not accuracy. The old ladder's own L3 flagged this and it was never
acted on: at M=128 accuracy was chance but at M=32 the margin was already down
to 1.62x, meaning rank-1 was coasting on a gap about to close. Accuracy is a
step function of a continuous quantity, so it reports "fine, fine, fine,
broken" and hides the approach. Margin (best stored match / second best) is the
continuous quantity, and extrapolating it is the only way to say where the
ceiling is rather than that it has been passed.

`acc_b` -- cue the OTHER parent -- stays in as the merge criterion, because a
capacity claim about an operation that is no longer a merge is worthless. It
held at 1.0000 for M=16 with `back_project` gated; whether it survives crowding
is a separate question.

n is swept because the lexicon result predicts capacity is not the area's size
but what is done to it. If the merge ceiling scales with n while the lexicon
has none, the binding constraint is composition, and that is the number a
grammar has to live inside.

PRE-REGISTERED
--------------
G1 The M=64 cliff is GONE -- acc_a > 0.90 there, where the old ladder read
   0.0260. This is the direct test that the cliff was the lexicon.
G2 margin DECAYS SMOOTHLY with M rather than falling off a cliff, so the
   ceiling can be located by where margin crosses ~1.1 rather than by where
   accuracy has already failed.
G3 acc_b tracks acc_a. If acc_b falls away first, crowding costs the merge
   criterion before it costs retrieval, and the two-parent property is the
   fragile one.
G4 The ceiling scales with n. If it does not, composition capacity is set by
   something other than area size and that is the finding.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import MIN_TRIALS, probe, rank1, read, similarity, spread  # noqa: E402

K_, P_, BETA = 50, 0.05, 0.10
BUILD_ROUNDS, MERGE_ROUNDS = 6, 2
LEAF, PARTNER, TARGET = "A", "B", "C"
GATED = dict(parent_self=False, target_self=False, back_project=False)

GRID = [(1000, m) for m in (16, 32, 64, 128, 256)] + \
       [(4000, m) for m in (64, 256, 512)]


def build_ff(brain, stim, area, rounds):
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read(brain, area)


def seeds_for(m):
    return (42, 7, 123, 2024, 5, 99) if m <= 32 else (42, 7, 123)


def trial(n, m_items, seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P_, seed=seed)
    for area in (LEAF, PARTNER, TARGET):
        brain.add_area(area, n, K_, beta=BETA)
    for m in range(m_items):
        brain.add_stimulus(f"a{m}", K_)
        brain.add_stimulus(f"b{m}", K_)
    for m in range(m_items):
        build_ff(brain, f"a{m}", LEAF, BUILD_ROUNDS)
        build_ff(brain, f"b{m}", PARTNER, BUILD_ROUNDS)

    stored = {}
    for m in range(m_items):
        merge(brain, LEAF, PARTNER, TARGET, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=MERGE_ROUNDS, **GATED)
        stored[m] = read(brain, TARGET)

    hits_a = hits_b = 0
    margins = []
    for m in range(m_items):
        with probe(brain):
            build_ff(brain, f"a{m}", LEAF, BUILD_ROUNDS)
            brain.project({}, {LEAF: [TARGET]})
            live = read(brain, TARGET)
            hits_a += rank1(live, stored) == m
            sims = sorted((similarity(live, a) for a in stored.values()),
                          reverse=True)
            if len(sims) > 1 and sims[1] > 0:
                margins.append(sims[0] / sims[1])
        with probe(brain):
            build_ff(brain, f"b{m}", PARTNER, BUILD_ROUNDS)
            brain.project({}, {PARTNER: [TARGET]})
            hits_b += rank1(read(brain, TARGET), stored) == m

    return (hits_a, hits_b, m_items,
            statistics.mean(margins) if margins else float("nan"),
            spread(stored.values()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  k={K_} beta={BETA} p={P_}, feed-forward lexicon, merge with all "
          f"three channels gated, T={MERGE_ROUNDS}")
    print(f"  ONE SHARED target area. acc_a cues parent A, acc_b cues parent B "
          f"(the merge criterion).")
    print(f"  margin = best stored match / second best -- the CONTINUOUS "
          f"quantity; accuracy only reports that it already closed.\n")
    print(f"  {'n':>6}{'M':>6}{'load':>7}{'chance':>9}{'acc_a':>8}{'acc_b':>8}"
          f"{'margin':>9}{'stor_spr':>10}{'floor':>8}{'trials':>8}")

    rows = {}
    for n, m_items in GRID:
        res = [trial(n, m_items, s) for s in seeds_for(m_items)]
        tot = sum(x[2] for x in res)
        r = (sum(x[0] for x in res) / tot, sum(x[1] for x in res) / tot,
             statistics.mean(x[3] for x in res),
             statistics.mean(x[4] for x in res), tot)
        rows[(n, m_items)] = r
        flag = "" if r[4] >= MIN_TRIALS else "  [UNDER-POWERED]"
        print(f"  {n:>6}{m_items:>6}{m_items * K_ / n:>7.1f}"
              f"{1 / m_items:>9.4f}{r[0]:>8.4f}{r[1]:>8.4f}{r[2]:>9.2f}"
              f"{r[3]:>10.4f}{K_ / n:>8.4f}{r[4]:>8}{flag}")

    print("\n  READING")
    g1 = rows[(1000, 64)][0] > 0.90
    print(f"    G1 the M=64 cliff is gone:   {g1}   acc_a "
          f"{rows[(1000, 64)][0]:.4f}  (old ladder read 0.0260)")
    ms = [m for (n, m) in rows if n == 1000]
    mg = [rows[(1000, m)][2] for m in ms]
    smooth = all(mg[i] >= mg[i + 1] - 0.05 for i in range(len(mg) - 1))
    print(f"    G2 margin decays smoothly:   {smooth}   " +
          "  ".join(f"M={m}:{v:.2f}x" for m, v in zip(ms, mg)))
    g3 = all(abs(rows[k][0] - rows[k][1]) < 0.10 for k in rows)
    print(f"    G3 acc_b tracks acc_a:       {g3}")
    for k in sorted(rows):
        if abs(rows[k][0] - rows[k][1]) >= 0.10:
            print(f"       n={k[0]} M={k[1]}: acc_a {rows[k][0]:.4f} "
                  f"acc_b {rows[k][1]:.4f}  <- criterion fails first")

    def ceiling(n):
        ok = [m for (nn, m) in rows if nn == n and rows[(nn, m)][0] > 0.90]
        return max(ok) if ok else 0
    c1, c4 = ceiling(1000), ceiling(4000)
    print(f"    G4 ceiling scales with n:    n=1000 M>={c1}   n=4000 M>={c4}")

    print()
    top1 = max(m for (n, m) in rows if n == 1000)
    if c1 == top1:
        print(f"    NO CEILING FOUND AT n=1000 UP TO M={top1} "
              f"(load {top1 * K_ / 1000:.1f}x), margin "
              f"{rows[(1000, top1)][2]:.2f}x still open. The sweep ran out")
        print(f"    before the capacity did, so M={top1} is a LOWER BOUND, and")
        print(f"    the margin column is what says how much room is left.")
    else:
        nxt = min([m for (n, m) in rows if n == 1000 and m > c1], default=None)
        print(f"    CEILING between M={c1} and M={nxt} at n=1000. Read the")
        print(f"    margin column to see whether it approached smoothly (a")
        print(f"    crowding limit, extrapolable) or fell off (a collapse,")
        print(f"    which would mean something is still eroding).")


if __name__ == "__main__":
    main()
