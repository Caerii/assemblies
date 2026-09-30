"""How does composition decay with DEPTH? (chain to level 3, on the harness)

WHERE THIS PICKS UP
-------------------
`compositional_depth_v2.py` established depth 2 at 96 trials/probe, both levels
distinct:

    (a) cue A -> C1         [driven parents]     0.9479
    (b) PIN true C1 -> C2   [composed parent]    1.0000
    (c) cue A -> C1 -> C2   [full depth 2]       0.6979

Two things follow, and they set up exactly one question. First, (b)=1.0000
REFUTES the structural hypothesis: a level whose left parent is composed and
stimulus-less recalls perfectly, so the pinned-parent plasticity short-circuit
`ops.merge` documents (0.000 fixed vs 3.47x driven) does not cap depth at one.
Second, all of the depth-2 loss sits in RE-DERIVING the parent, since (c) falls
while (b) does not.

THE QUESTION: IS THE LOSS GEOMETRIC?
-------------------------------------
If each level's re-derivation fails independently at rate r, then full-chain
recall at depth L should be about r^L. At depth 2 that predicted 0.9479 x 1.0000
= 0.948 against 0.698 observed -- worse than multiplicative, from ONE point,
which is not enough to call a shape. Depth 3 is the point that decides it, and
the three readings are qualitatively different claims:

    c3 ~ c2 x (per-level)     geometric -- error compounds independently, and
                              depth is bounded by arithmetic, not by mechanism
    c3 << geometric           accelerating -- something degrades the chain
                              faster than error propagation, e.g. each extra
                              shared target crowds the ones below it
    c3 ~ c2                   saturating -- the loss is paid ONCE, at the first
                              re-derivation, and further levels are nearly free

CHAIN. Level L merges the previous constituent (PINNED, composed) with a fresh
stimulus-driven area:

    C1 = merge(A,  B)     both driven
    C2 = merge(C1, D)     C1 pinned
    C3 = merge(C2, E)     C2 pinned

PROBES, at every level, so per-level and cumulative rates are separable:
    step_L   PIN the true C_{L-1}, read C_L       one level, perfect parent
    full_L   cue A and walk the whole chain       cumulative

PRE-REGISTERED
--------------
G1 step_L stays high at EVERY level (>= 0.9). If step_3 drops while step_2 held,
   depth itself degrades single-level composition, which nothing so far
   predicts.
G2 full_3 < full_2 < full_1. Monotone, since each level adds a re-derivation.
G3 THE TEST: full_3 is compared against full_2 x step_3 (the geometric
   prediction). Reported as a ratio so it cannot be eyeballed favourably.
G4 Every level stays distinct. C2 and C3 are shared targets exactly as C1 is,
   so the upper wall applies at each; if any level collapses, its rows are
   uninterpretable and G1-G3 say nothing.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, pinned, probe, rank1, read, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS, MERGE_ROUNDS = 6, 1
DEPTH = 3

LEAF = "A"
#: right-hand partner for each level, each with its own stimulus family
PARTNER = {1: "B", 2: "D", 3: "E"}
STIM_OF = {"A": "a", "B": "b", "D": "d", "E": "e"}
LEVEL = {L: f"C{L}" for L in range(1, DEPTH + 1)}


def trial(seed: int):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in list(STIM_OF) + list(LEVEL.values()):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        for pre in STIM_OF.values():
            brain.add_stimulus(f"{pre}{m}", K)

    for m in range(M_ITEMS):
        for area, pre in STIM_OF.items():
            build(brain, f"{pre}{m}", area, PARENT_ROUNDS)

    stored = {}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER[1], LEVEL[1],
              stim_a=f"a{m}", stim_b=f"b{m}", rounds=MERGE_ROUNDS)
        stored.setdefault(1, {})[m] = read(brain, LEVEL[1])
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            with pinned(brain, LEVEL[L - 1], stored[L - 1][m]):
                merge(brain, LEVEL[L - 1], PARTNER[L], LEVEL[L],
                      stim_b=f"{STIM_OF[PARTNER[L]]}{m}", rounds=MERGE_ROUNDS,
                      unstimulated_source_mode="require-fixed")
                stored.setdefault(L, {})[m] = read(brain, LEVEL[L])

    def settle(src, tgt):
        brain.project({}, {src: [tgt]})
        for _ in range(MERGE_ROUNDS - 1):
            brain.project({}, {src: [tgt], tgt: [tgt]})

    step = {L: 0 for L in range(1, DEPTH + 1)}
    full = {L: 0 for L in range(1, DEPTH + 1)}
    for m in range(M_ITEMS):
        # per-level, perfect parent
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, LEVEL[1])
            step[1] += rank1(read(brain, LEVEL[1]), stored[1]) == m
        for L in range(2, DEPTH + 1):
            with probe(brain):
                with pinned(brain, LEVEL[L - 1], stored[L - 1][m]):
                    settle(LEVEL[L - 1], LEVEL[L])
                    step[L] += rank1(read(brain, LEVEL[L]), stored[L]) == m
        # cumulative, re-derived from the leaf at every level
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            settle(LEAF, LEVEL[1])
            full[1] += rank1(read(brain, LEVEL[1]), stored[1]) == m
            for L in range(2, DEPTH + 1):
                settle(LEVEL[L - 1], LEVEL[L])
                full[L] += rank1(read(brain, LEVEL[L]), stored[L]) == m

    sp = {L: spread(stored[L].values()) for L in range(1, DEPTH + 1)}
    return step, full, sp


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    step = {L: sum(r[0][L] for r in res) / trials for L in LEVEL}
    full = {L: sum(r[1][L] for r in res) / trials for L in LEVEL}
    sp = {L: statistics.mean(r[2][L] for r in res) for L in LEVEL}

    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS}, depth {DEPTH}")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/probe "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  distinct chance = {K / N:.3f}; >0.9 at any level voids that row\n")
    print(f"  {'level':>6}{'step (perfect parent)':>24}{'full (from leaf)':>19}"
          f"{'distinct':>11}")
    for L in range(1, DEPTH + 1):
        print(f"  {L:>6}{step[L]:>24.4f}{full[L]:>19.4f}{sp[L]:>11.4f}")

    print("\n  G3 -- IS THE LOSS GEOMETRIC?")
    for L in range(2, DEPTH + 1):
        pred = full[L - 1] * step[L]
        ratio = (full[L] / pred) if pred > 0 else float("nan")
        print(f"    full_{L} = {full[L]:.4f}   vs  full_{L - 1} x step_{L} = "
              f"{pred:.4f}   ratio {ratio:.3f}")

    print("\n  VERDICT")
    if any(v > 0.9 for v in sp.values()):
        print("    A LEVEL COLLAPSED (G4). Its rows are uninterpretable; lower")
        print("    MERGE_ROUNDS or PARENT_ROUNDS before reading anything else.")
        return
    if min(step.values()) < 0.9:
        worst = min(step, key=lambda L: step[L])
        print(f"    G1 FAILS at level {worst} (step={step[worst]:.4f}). Depth")
        print("    degrades SINGLE-LEVEL composition, which nothing so far")
        print("    predicted -- that is the finding, and it outranks G3.")
        return
    ratios = [full[L] / (full[L - 1] * step[L])
              for L in range(2, DEPTH + 1) if full[L - 1] * step[L] > 0]
    r = statistics.mean(ratios) if ratios else float("nan")
    if r > 0.9:
        print("    GEOMETRIC. Each level costs an independent re-derivation and")
        print("    nothing more, so depth is bounded by ARITHMETIC rather than")
        print("    by mechanism -- raise the per-level rate and depth follows.")
    elif r > 0.6:
        print("    SUB-GEOMETRIC. Error compounds faster than independence")
        print("    predicts; the chain degrades beyond simple propagation.")
        print("    Next: is the excess loss constant per level, or growing?")
    else:
        print("    ACCELERATING. Something degrades the chain far faster than")
        print("    error propagation -- most likely each added shared target")
        print("    crowding the ones below it. Depth is mechanism-bound here.")


if __name__ == "__main__":
    main()
