"""Does PARENT DRIFT destroy addressing? (fixing the drive probe, and a sweep)

WHAT WENT WRONG LAST TIME
-------------------------
`slot_addressing_drive.py` read argmax accuracy 0.0625 and a margin of EXACTLY
1.00x, and concluded the addressing signal is absent. A direct weight check
refuted that immediately: `W[LEAF -> slot_m]` is potentiated 1.6x-3.4x on the
stored assembly against off it. The information is in the weights, so the
0.0625 was the probe.

The bug is visible in the connectome shapes: `W[A->C1_0]` came back (1202, 400)
while `W[A->C1_1]` was (1528, 100) -- LEAF's materialized count GROWS across
merges. At mergeR=20 merge back-projects into its own sources for 20 rounds
(`target: [target, source_a, source_b]`), so LEAF DRIFTS while each merge runs.
The rows that got potentiated are the drifted leaf assembly at merge time, not
the clean assembly re-cued afterwards. The old probe summed over the wrong rows.

THAT IS ALSO A HYPOTHESIS ABOUT THE SUBSTRATE, not just a bug
--------------------------------------------------------------
If drift is real, the same high mergeR that made constituents perfect attractors
(quality 1.0000, depth flat to level 3) also moves their parents out from under
the cue. Addressing would then fail for a reason that has nothing to do with
k-WTA, and it would be a THIRD instance of today's pattern: the setting that
fixes one property breaks another through a shared mechanism.

THREE MEASUREMENTS, one sweep over mergeR
------------------------------------------
    drift        overlap(re-cued LEAF, LEAF as it stood at merge time).
                 1.0 means no drift; low means the cue no longer matches the
                 rows that were potentiated.
    addr_merge   argmax-drive accuracy using the MERGE-TIME leaf rows. This is
                 the upper bound -- it asks whether the signal is in the
                 weights AT ALL, with the drift confound removed by
                 construction.
    addr_cue     the same using the RE-CUED leaf rows. This is what a real
                 system actually has available, since nothing stores the
                 merge-time assembly.

Drive is read straight from the connectome (no projection, no k-WTA, no
settling), so saturation cannot intervene. Both index spaces go through
`ops._compact_index`.

PRE-REGISTERED
--------------
P1 drift FALLS as mergeR rises. This is the mechanism claim.
P2 addr_merge is high at every mergeR -- the signal is in the weights
   regardless, which is what the 1.6x-3.4x weight check already implies.
P3 addr_cue TRACKS drift: high where drift is high, collapsing where drift is
   low. If P2 holds and P3 holds, addressing is defeated by parent drift and
   not by any property of the readout -- and the fix is to stop the parents
   moving (fix or gate the back-projection), not to add a pointer.
P4 If addr_merge is ALSO at chance, then the weight potentiation measured is
   real but not DISCRIMINATIVE -- every slot is potentiated on its own
   assembly, so the comparison across slots carries no signal. That would send
   addressing to a learned pointer, and it is the outcome that would refute the
   two-phase gated read.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _substrate import MIN_TRIALS, build, probe, read, similarity  # noqa: E402

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS = 6
MERGE_ROUNDS = (2, 5, 10, 20)
LEAF, PARTNER = "A", "B"


def slot(m):
    return f"C1_{m}"


def drive(brain, src, tgt, src_ids, tgt_ids):
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
    # MEAN not sum: the number of usable rows/cols differs per slot because the
    # areas have materialized different numbers of neurons, and a sum would then
    # rank slots by how much of the matrix happens to exist.
    return float(w[np.ix_(rows, cols)].mean())


def argmax_acc(brain, rows_by_item, stored):
    hits, margins = 0, []
    for m in range(M_ITEMS):
        d = np.array([drive(brain, LEAF, slot(j), rows_by_item[m], stored[j])
                      for j in range(M_ITEMS)], dtype=float)
        if np.all(np.isnan(d)):
            continue
        hits += int(np.nanargmax(d)) == m
        others = np.delete(d, m)
        om = np.nanmean(others)
        if om > 0 and not np.isnan(d[m]):
            margins.append(d[m] / om)
    return hits / M_ITEMS, (statistics.mean(margins) if margins else float("nan"))


def trial(merge_rounds, seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    brain.add_area(LEAF, N, K, beta=BETA)
    brain.add_area(PARTNER, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_area(slot(m), N, K, beta=BETA)
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)
    for m in range(M_ITEMS):
        build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
        build(brain, f"b{m}", PARTNER, PARENT_ROUNDS)

    stored, at_merge = {}, {}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER, slot(m), stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=merge_rounds)
        stored[m] = read(brain, slot(m))
        at_merge[m] = read(brain, LEAF)      # LEAF as it stands after merge m

    recued = {}
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            recued[m] = read(brain, LEAF)

    drift = statistics.mean(similarity(recued[m], at_merge[m])
                            for m in range(M_ITEMS))
    am, mm = argmax_acc(brain, at_merge, stored)
    ac, mc = argmax_acc(brain, recued, stored)
    return drift, am, mm, ac, mc


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    chance = 1.0 / M_ITEMS
    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS}, "
          f"{M_ITEMS} slots (one per constituent)")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials/cell "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}")
    print(f"  drift = overlap(re-cued LEAF, LEAF at merge time); 1.0 = no "
          f"drift\n")
    print(f"  {'mergeR':>7}{'drift':>8}{'addr_merge':>12}{'margin':>9}"
          f"{'addr_cue':>10}{'margin':>9}")

    rows = []
    for mr in MERGE_ROUNDS:
        res = [trial(mr, s) for s in SEEDS]
        d = statistics.mean(r[0] for r in res)
        am = statistics.mean(r[1] for r in res)
        mm = statistics.mean(r[2] for r in res)
        ac = statistics.mean(r[3] for r in res)
        mc = statistics.mean(r[4] for r in res)
        rows.append((mr, d, am, mm, ac, mc))
        print(f"  {mr:>7}{d:>8.4f}{am:>12.4f}{mm:>9.2f}{ac:>10.4f}{mc:>9.2f}")

    print("\n  READING")
    best_am = max(r[2] for r in rows)
    best_ac = max(r[4] for r in rows)
    print(f"    P2 signal is in the weights (addr_merge):  "
          f"{best_am > chance + 0.20}   best {best_am:.4f}")
    print(f"    P3 what a real cue can use  (addr_cue):    "
          f"{best_ac > chance + 0.20}   best {best_ac:.4f}")
    if best_am <= chance + 0.20:
        print("\n    P4. Even with the merge-time rows the comparison across")
        print("    slots is at chance: every slot is potentiated on its OWN")
        print("    assembly, so the potentiation is real but NOT")
        print("    DISCRIMINATIVE. The two-phase gated read cannot work, and")
        print("    addressing needs a learned pointer.")
    elif best_ac > chance + 0.20:
        print("\n    ADDRESSING IS AVAILABLE to a real cue. The two-phase read")
        print("    is on: inhibit C->C, compare drive across slots, then")
        print("    disinhibit the winner and let it complete.")
    else:
        print("\n    DRIFT IS THE BARRIER. The signal exists (P2) but the cue")
        print("    cannot reach it (P3), because merge moved the parent out")
        print("    from under it. Fix the parents during merge -- or gate the")
        print("    back-projection -- rather than adding a pointer.")


if __name__ == "__main__":
    main()
