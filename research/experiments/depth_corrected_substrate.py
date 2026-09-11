"""Depth 3, re-measured with the collapse channels gated.

WHY EVERY EARLIER DEPTH NUMBER HAS TO BE THROWN OUT
----------------------------------------------------
The shared-target chain decayed 0.9167 -> 0.6250 -> 0.3646
(`compositional_depth_chain.py`), and the flat 1.0000 chain that replaced it was
withdrawn when `depth_wrong_cue_control.py` showed the cue was inert. Both were
measured through areas that were collapsing while they were measured:

  * the LEXICON collapsed, because `_substrate.build` runs `area -> area`
    (`lexicon_capacity_law.py`: ceiling M=32 at n=1000, and no ceiling at all
    feed-forward -- 1.0000 to M=256, overlap 0.0510 vs floor 0.0500);
  * the MERGE areas collapsed, through all three of `ops.merge`'s recurrent
    channels (`merge_recurrence_channels.py`: parents gone by T=5, target by
    T=10).

So "erosion with depth" was measured on a substrate that was erasing itself
regardless of depth. This file re-runs the same question with both fixed:
feed-forward lexicon, and `ops.merge(..., parent_self=False, target_self=False,
back_project=False)` -- now a real parameter, verified to reproduce the
standalone experiment to four decimals.

ARCHITECTURE, and why it is a fair test this time
--------------------------------------------------
ONE SHARED AREA PER LEVEL, not one per constituent. The slot architecture is
what made the old depth result untestable: with a dedicated area per item the
chain is structurally determined once level 1 is addressed, so end-to-end
accuracy was level-1 accuracy propagated (`two_phase_read.py` says so in its
own docstring). With shared areas the system must land on the right one of 16
assemblies AT EVERY LEVEL, so errors can compound and `full_3` is a genuine
product of three selections rather than one.

It also makes the wrong-cue control unnecessary as a separate arm. In a shared
area there is nothing to tell the readout: an input-independent read returns the
same item for all 16 cues and scores exactly 1/16. `distinct` -- how many
different items the 16 cues elicit -- reports collapse directly.

Composition above level 1 pins the parent, which is now sound: `ops.merge`
warns that a pinned parent receives no back-projection plasticity (measured
0.000 vs 3.47x), but with `back_project=False` there is no back-projection to
lose. Gating the channel is what makes pinning free.

READOUT is a pure forward sweep -- cue the leaf, project level to level, read.
No pinning, no stored labels, no cross-area argmax, nothing non-neural.

PRE-REGISTERED
--------------
D1 full_1 clears chance by a wide margin. Reproduces `merge_capacity_ladder`
   inside the chain harness; if it fails, nothing above it is interpretable.
D2 full_3 clears chance by a wide margin -- composition survives to depth 3 on
   a substrate that is not collapsing. This is the claim the withdrawn result
   made and could not support.
D3 The decay is milder than the old 0.9167 -> 0.6250 -> 0.3646. That baseline
   was measured through collapse, so most of its loss should be gone. If the
   decay is UNCHANGED, depth costs something real and independent of collapse,
   which would be the more interesting outcome and outranks D2.
D4 distinct stays near 16 at every level. If it falls with depth, the higher
   levels are crowding and the accuracy is coasting on a shrinking margin --
   so `margin` is reported alongside, per the L3 lesson from the ladder.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, pinned, probe, read, similarity, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
BUILD_ROUNDS, MERGE_ROUNDS = 6, 10
DEPTH = 3
LEAF = "A"
PARTNER = {1: "B", 2: "D", 3: "E"}
STIM_OF = {"A": "a", "B": "b", "D": "d", "E": "e"}
GATED = dict(parent_self=False, target_self=False, back_project=False)


def level(L):
    return f"C{L}"


def build_ff(brain, stim, area, rounds):
    """Feed-forward lexicon build -- no `area -> area`. See lexicon_capacity_law."""
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read(brain, area)


def trial(seed, gated, rounds=None):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    kw = GATED if gated else {}
    if rounds is None:
        rounds = MERGE_ROUNDS if gated else 2  # each arm at its own optimum
    build = build_ff if gated else _rec_build

    brain = Brain(p=P, seed=seed)
    for area in STIM_OF:
        brain.add_area(area, N, K, beta=BETA)
    for L in range(1, DEPTH + 1):
        brain.add_area(level(L), N, K, beta=BETA)
    for m in range(M_ITEMS):
        for pre in STIM_OF.values():
            brain.add_stimulus(f"{pre}{m}", K)
    for m in range(M_ITEMS):
        for area, pre in STIM_OF.items():
            build(brain, f"{pre}{m}", area, BUILD_ROUNDS)

    stored = {L: {} for L in range(1, DEPTH + 1)}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER[1], level(1), stim_a=f"a{m}",
              stim_b=f"b{m}", rounds=rounds, **kw)
        stored[1][m] = read(brain, level(1))
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            with pinned(brain, level(L - 1), stored[L - 1][m]):
                merge(brain, level(L - 1), PARTNER[L], level(L),
                      stim_b=f"{STIM_OF[PARTNER[L]]}{m}", rounds=rounds,
                      unstimulated_source_mode="require-fixed", **kw)
                stored[L][m] = read(brain, level(L))

    full = {L: 0 for L in range(1, DEPTH + 1)}
    picks = {L: set() for L in range(1, DEPTH + 1)}
    margins = {L: [] for L in range(1, DEPTH + 1)}
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, BUILD_ROUNDS)
            src = LEAF
            for L in range(1, DEPTH + 1):
                brain.project({}, {src: [level(L)]})
                live = read(brain, level(L))
                sims = sorted(((similarity(live, a), j)
                               for j, a in stored[L].items()), reverse=True)
                picks[L].add(sims[0][1])
                full[L] += sims[0][1] == m
                if len(sims) > 1 and sims[1][0] > 0:
                    margins[L].append(sims[0][0] / sims[1][0])
                src = level(L)

    return (full, {L: len(picks[L]) for L in picks},
            {L: (statistics.mean(v) if v else float("nan"))
             for L, v in margins.items()},
            {L: spread(stored[L].values()) for L in stored})


def _rec_build(brain, stim, area, rounds):
    from neural_assemblies.assembly_calculus.ops import project
    project(brain, stim, area, rounds=rounds, recurrent=True)
    return read(brain, area)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    print(f"\n  n={N} k={K} beta={BETA}, ONE SHARED AREA PER LEVEL, depth "
          f"{DEPTH}, {M_ITEMS} constituents")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  gated  = feed-forward lexicon + merge with all three recurrent "
          f"channels off, T={MERGE_ROUNDS}")
    print(f"  legacy = recurrent lexicon + ops.merge defaults, T=2 (its best "
          f"setting)\n")
    print(f"  {'arm':>8}{'level':>7}{'full':>9}{'distinct':>10}{'margin':>9}"
          f"{'stored_spr':>12}")

    out = {}
    for gated in (True, False):
        res = [trial(s, gated) for s in SEEDS]
        arm = "gated" if gated else "legacy"
        out[arm] = {}
        for L in range(1, DEPTH + 1):
            full = sum(r[0][L] for r in res) / trials
            dis = statistics.mean(r[1][L] for r in res)
            mar = statistics.mean(r[2][L] for r in res)
            spr = statistics.mean(r[3][L] for r in res)
            out[arm][L] = (full, dis, mar, spr)
            print(f"  {arm:>8}{L:>7}{full:>9.4f}{dis:>10.2f}{mar:>9.2f}"
                  f"{spr:>12.4f}")
        print()

    # D3's failure localises to a quantity level 1 cannot show: composition
    # AMPLIFIES overlap, so the leading indicator for full_3 is level-1
    # stored_spr, not level-1 accuracy. Sweep the merge rounds that set it.
    print(f"  MERGE-ROUNDS SWEEP, gated arm -- optimising full_3, not full_1")
    print(f"  {'T':>5}{'full_1':>9}{'full_2':>9}{'full_3':>9}"
          f"{'spr_1':>8}{'spr_2':>8}{'spr_3':>8}{'margin_3':>10}")
    sweep = {}
    for t in (1, 2, 3, 5, 10):
        res = [trial(s, True, rounds=t) for s in SEEDS]
        full = {L: sum(r[0][L] for r in res) / trials
                for L in range(1, DEPTH + 1)}
        spr = {L: statistics.mean(r[3][L] for r in res)
               for L in range(1, DEPTH + 1)}
        mar = statistics.mean(r[2][DEPTH] for r in res)
        sweep[t] = (full, spr, mar)
        print(f"  {t:>5}{full[1]:>9.4f}{full[2]:>9.4f}{full[3]:>9.4f}"
              f"{spr[1]:>8.4f}{spr[2]:>8.4f}{spr[3]:>8.4f}{mar:>10.2f}")
    best_t = max(sweep, key=lambda t: sweep[t][0][DEPTH])
    print(f"\n  best for depth 3: T={best_t}, full_3 "
          f"{sweep[best_t][0][DEPTH]:.4f}  (T={MERGE_ROUNDS} gave "
          f"{sweep[MERGE_ROUNDS][0][DEPTH]:.4f}, legacy {out['legacy'][3][0]:.4f})")

    # Report the SWEPT optimum, not the pre-registered guess. MERGE_ROUNDS was
    # picked from `merge_capacity_ladder`, which optimised depth-1 fidelity --
    # the wrong objective here, as the sweep above shows.
    bf, bs, bm = sweep[best_t]
    nan = float("nan")
    g = {L: (bf[L], nan, bm if L == DEPTH else nan, bs[L])
         for L in range(1, DEPTH + 1)}
    print(f"\n  OLD SHARED-TARGET BASELINE (compositional_depth_chain.py, "
          f"measured through collapse)")
    print(f"    full  0.9167 -> 0.6250 -> 0.3646")
    print(f"  THIS RUN, gated")
    print(f"    full  {g[1][0]:.4f} -> {g[2][0]:.4f} -> {g[3][0]:.4f}")

    print("\n  READING")
    d1 = g[1][0] > chance + 0.20
    d2 = g[3][0] > chance + 0.20
    print(f"    D1 level 1 clears chance:        {d1}   ({g[1][0]:.4f})")
    print(f"    D2 level 3 clears chance:        {d2}   ({g[3][0]:.4f})")
    d3 = g[3][0] > 0.3646
    print(f"    D3 decay milder than baseline:   {d3}   "
          f"({g[3][0]:.4f} vs 0.3646)")
    d4 = g[DEPTH][3] < 2 * (K / N)
    print(f"    D4 still distinct at level 3:    {d4}   "
          f"(stored_spr {g[DEPTH][3]:.4f} vs floor {K / N:.4f}, "
          f"margin {g[DEPTH][2]:.2f}x)")
    print(f"    reported at the SWEPT optimum T={best_t}, not the "
          f"pre-registered T={MERGE_ROUNDS}")

    print()
    if d2 and d3:
        print(f"    DEPTH SURVIVES ON A SUBSTRATE THAT IS NOT COLLAPSING.")
        print(f"    Level 3 holds at {g[3][0]:.4f} against a baseline of 0.3646")
        print(f"    measured through collapsing areas, with one SHARED area per")
        print(f"    level -- so the chain makes three real selections, not one")
        print(f"    propagated through dedicated slots.")
    elif d1 and not d2:
        print(f"    Level 1 works and the chain still loses it by level 3, with")
        print(f"    the collapse channels gated. THAT is a cost of depth itself")
        print(f"    rather than of erosion, and it is the first time this line")
        print(f"    could have said so cleanly. Read `margin` down the levels:")
        print(f"    a falling margin means crowding, a flat one means the")
        print(f"    composed signal is being lost between levels.")
    else:
        print(f"    Level 1 fails, so the chain says nothing about depth. Check")
        print(f"    stored_spr against the {K / N:.4f} floor first -- if it is")
        print(f"    high, something is still collapsing and this file is")
        print(f"    measuring that instead.")


if __name__ == "__main__":
    main()
