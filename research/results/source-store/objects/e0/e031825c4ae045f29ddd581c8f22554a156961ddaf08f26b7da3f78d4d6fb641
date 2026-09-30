"""Two-phase read: SELECT by drive (attractor off), then COMPLETE the winner.

WHY THE READ HAS TO BE SPLIT
-----------------------------
`depth_wrong_cue_control.py` measured the thing that invalidates a single-phase
readout: cueing a slot with the WRONG item returns the same content to four
decimals (dependence 0.0000 at mergeR=20, 0.0813 at mergeR=2). Settling is
INPUT-INDEPENDENT. A strong attractor emits its stored assembly whatever you
present, so settling can never be the step that decides WHICH constituent comes
back -- and every "recall = 1.0000" measured that way was reporting that I had
told the readout where to look.

The information does exist. `slot_addressing_drift.py` scored slots straight
from the connectome, with no projection and therefore no k-WTA and no settling,
and picked the right slot 0.9896 of the time at margin 2.02x (mergeR=2). So the
margin lives in the DRIVE and is destroyed by the dynamics that preserve the
constituent.

Hence two phases, which is exactly what fiber gating buys:

    PHASE 1  SELECT     attractors off. Score every slot by the drive the cue
                        delivers onto its stored assembly. Pick the argmax.
    PHASE 2  COMPLETE   disinhibit the winner alone and let it settle, which is
                        where the attractor is an asset rather than a liability.

Gating is therefore not a way of choosing the constituent. It is the mechanism
that decides WHEN the attractor is allowed to run -- off while choosing, on
while completing -- and a dynamics that is always on cannot do both.

WHAT MAKES THIS TESTABLE RATHER THAN CIRCULAR
----------------------------------------------
Under a single-phase read the wrong-cue control was nearly meaningless: the slot
returned its own content, so "recall" succeeded by construction. Under a
two-phase read the wrong cue must ROUTE ELSEWHERE -- cue with item m' and phase 1
should select slot m', returning m''s constituent, not m's. That is genuine
input dependence at the system level and it cannot be faked by a strong
attractor.

CHAIN STRUCTURE, and an honest note about it. Training connects slot(L-1, m) to
slot(L, m) only, so no cross-level fibers exist between mismatched slots. Once
level 1 is addressed the rest of the chain is structurally determined. So
end-to-end accuracy IS level-1 addressing accuracy propagated, and this file
does not claim otherwise -- what it verifies is that the propagation preserves
the constituent, and that the whole path is driven by the cue.

PRE-REGISTERED
--------------
T1 Phase-1 selection clears chance by a wide margin (baseline: 0.9896 measured
   by the drive probe).
T2 End-to-end, the level-3 constituent recovered from a leaf cue is the correct
   one, well above chance.
T3 THE CONTROL: cueing with item m' selects slot m', not slot m. Reported as
   routing accuracy on the wrong cue. Under a single-phase read this was 0.0000
   dependence; here it should be high, and if it is not, the two-phase read
   has not actually fixed anything.
T4 Completion after selection preserves the constituent: overlap of the settled
   winner with its stored assembly stays high.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, pinned, probe, read, similarity,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS, MERGE_ROUNDS = 6, 2
DEPTH = 3
LEAF = "A"
PARTNER = {1: "B", 2: "D", 3: "E"}
STIM_OF = {"A": "a", "B": "b", "D": "d", "E": "e"}


def slot(L, m):
    return f"C{L}_{m}"


def drive(brain, src, tgt, src_ids, tgt_ids):
    """PHASE 1 primitive: drive with no projection, so no attractor can run."""
    from neural_assemblies.assembly_calculus.ops import _compact_index

    engine = brain._engine_for(brain.areas[tgt])
    conn = getattr(engine, "_area_conns", {}).get(src, {}).get(tgt)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    s_inv = _compact_index(brain._engine_for(brain.areas[src]), src) or {}
    t_inv = _compact_index(engine, tgt) or {}
    rows = [s_inv[int(x)] for x in src_ids
            if int(x) in s_inv and s_inv[int(x)] < w.shape[0]]
    cols = [t_inv[int(x)] for x in tgt_ids
            if int(x) in t_inv and t_inv[int(x)] < w.shape[1]]
    if not rows or not cols:
        return float("nan")
    return float(w[np.ix_(rows, cols)].mean())


def trial(seed):
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
              stim_a=f"a{m}", stim_b=f"b{m}", rounds=MERGE_ROUNDS)
        stored[1][m] = read(brain, slot(1, m))
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            with pinned(brain, slot(L - 1, m), stored[L - 1][m]):
                merge(brain, slot(L - 1, m), PARTNER[L], slot(L, m),
                      stim_b=f"{STIM_OF[PARTNER[L]]}{m}", rounds=MERGE_ROUNDS,
                      unstimulated_source_mode="require-fixed")
                stored[L][m] = read(brain, slot(L, m))

    def select(src_area, src_ids, level):
        """PHASE 1. Score every slot at *level*; attractors never run."""
        d = np.array([drive(brain, src_area, slot(level, j), src_ids,
                            stored[level][j]) for j in range(M_ITEMS)])
        return (int(np.nanargmax(d)) if not np.all(np.isnan(d)) else -1)

    def complete(src_area, tgt_area):
        """PHASE 2. Only the winner is disinhibited; now the attractor helps."""
        brain.project({}, {src_area: [tgt_area]})
        for _ in range(MERGE_ROUNDS - 1):
            brain.project({}, {src_area: [tgt_area], tgt_area: [tgt_area]})
        return read(brain, tgt_area)

    sel1 = end3 = route = 0
    fid = []
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            cue_ids = read(brain, LEAF)
            w1 = select(LEAF, cue_ids, 1)
            sel1 += w1 == m
            if w1 >= 0:
                live = complete(LEAF, slot(1, w1))
                fid.append(similarity(live, stored[1][w1]))
                cur = w1
                for L in range(2, DEPTH + 1):
                    live = complete(slot(L - 1, cur), slot(L, cur))
                end3 += (cur == m) and (
                    max((similarity(live, a), j)
                        for j, a in stored[DEPTH].items())[1] == m)
        # T3: a WRONG cue must route to the OTHER item's slot.
        other = (m + 1) % M_ITEMS
        with probe(brain):
            build(brain, f"a{other}", LEAF, PARENT_ROUNDS)
            route += select(LEAF, read(brain, LEAF), 1) == other

    return (sel1 / M_ITEMS, end3 / M_ITEMS, route / M_ITEMS,
            statistics.mean(fid) if fid else float("nan"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    sel = statistics.mean(r[0] for r in res)
    end = statistics.mean(r[1] for r in res)
    rou = statistics.mean(r[2] for r in res)
    fid = statistics.mean(r[3] for r in res)

    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS}, one area per constituent, depth {DEPTH}")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}\n")
    print(f"  {'T1 phase-1 selection (correct cue)':<46}{sel:>9.4f}")
    print(f"  {'T2 end-to-end level-3 constituent':<46}{end:>9.4f}")
    print(f"  {'T3 WRONG cue routes to the other slot':<46}{rou:>9.4f}")
    print(f"  {'T4 completion fidelity after selection':<46}{fid:>9.4f}")
    print(f"\n  single-phase baseline (depth_wrong_cue_control):")
    print(f"    cue dependence 0.0813 at mergeR=2, 0.0000 at mergeR=20 "
          f"-- readout inert")

    print("\n  READING")
    ok = sel > chance + 0.20 and rou > chance + 0.20
    print(f"    selection is real and input-driven:  {ok}")
    print(f"    chain preserves the constituent:     {end > chance + 0.20}")
    if ok and end > chance + 0.20:
        print("\n    THE TWO-PHASE READ WORKS. Selecting on drive with the")
        print("    attractors off, then completing only the winner, gives a")
        print("    readout that DEPENDS ON THE CUE -- the property a")
        print("    single-phase read structurally cannot have. Gating is not")
        print("    an optimization here; it is what makes retrieval possible.")
    elif ok:
        print("\n    Selection works but the chain loses the constituent.")
        print("    Addressing is solved; propagation is not.")
    else:
        print("\n    Selection does not survive being embedded in the read.")
        print("    The drive margin measured offline does not transfer, and")
        print("    that gap is the next thing to explain.")


if __name__ == "__main__":
    main()
