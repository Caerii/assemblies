"""Is the addressing signal THERE, before the attractor eats it?

WHAT THIS CORRECTS
------------------
`depth_chain_slotted.py` reported slot addressing at 0.0625 -- exactly chance --
and I read that as "addressing is impossible without gating". That reading is
not supported. The probe compared each slot's POST-SETTLING state to its own
stored content, which saturates by construction: a slot with a strong basin
falls into its own occupant whatever you present, so every slot scores ~1.0 and
the argmax is arbitrary. What that measured was the probe, not the substrate.

The question it should have asked is whether the information exists BEFORE the
attractor runs. There is direct prior evidence in this repo that it does: the
mutual-inhibition work measured a 7x PRE-k-WTA separation collapsing to a ~7%
margin post-k-WTA. Same shape -- real signal in the drive, destroyed by
winner-take-all.

HOW DRIVE IS READ HERE
----------------------
Straight from the connectome, not from activity, which is what makes it immune
to the saturation above. For a cue assembly in LEAF and a slot's stored
assembly, the drive is the total synaptic weight the cue delivers onto exactly
those target neurons:

    drive(m, j) = sum of W[LEAF -> slot_j] over (cue_m rows) x (stored_j cols)

No projection is run, so no k-WTA and no settling can intervene. Both index
spaces are converted through `ops._compact_index`, the same one-way door
`_substrate.read` uses -- rows are compact in the SOURCE area, columns compact
in the TARGET area, and mixing them up is the error class this whole line of
work exists to prevent.

PRE-REGISTERED
--------------
A1 argmax_j drive(m, j) == m at well above chance (1/16 = 0.0625). This is the
   claim: the addressing signal is present in the drive.
A2 The MARGIN -- drive to the correct slot over the mean of the other 15 -- is
   substantially above 1.0. Prior work saw 7x pre-k-WTA in a different setting;
   no specific value is predicted here, only that it exceeds 1.
A3 A1 holds even though the post-settling readout was at chance, which
   localises the loss precisely at the k-WTA/settling step rather than in the
   stored weights.

IF A1 FAILS the information genuinely is not in the LEAF->slot fibers, no gating
scheme recovers it, and addressing needs a learned pointer instead -- a much
larger claim. Either way this is one measurement and it decides the branch.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _substrate import MIN_TRIALS, build, probe, read  # noqa: E402

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
PARENT_ROUNDS, MERGE_ROUNDS = 6, 20
LEAF, PARTNER = "A", "B"


def slot(m):
    return f"C1_{m}"


def fiber_drive(brain, src, tgt, src_ids, tgt_ids):
    """Total W[src->tgt] weight from *src_ids* onto *tgt_ids*. Neuron IDs in."""
    from neural_assemblies.assembly_calculus.ops import _compact_index

    engine = brain._engine_for(brain.areas[tgt])
    conn = getattr(engine, "_area_conns", {}).get(src, {}).get(tgt)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    src_inv = _compact_index(brain._engine_for(brain.areas[src]), src) or {}
    tgt_inv = _compact_index(engine, tgt) or {}
    rows = [src_inv[int(x)] for x in src_ids if int(x) in src_inv]
    cols = [tgt_inv[int(x)] for x in tgt_ids if int(x) in tgt_inv]
    rows = [r for r in rows if r < w.shape[0]]
    cols = [c for c in cols if c < w.shape[1]]
    if not rows or not cols:
        return float("nan")
    return float(w[np.ix_(rows, cols)].sum())


def trial(seed):
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

    stored = {}
    for m in range(M_ITEMS):
        merge(brain, LEAF, PARTNER, slot(m), stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=MERGE_ROUNDS)
        stored[m] = read(brain, slot(m))

    hits, margins = 0, []
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", LEAF, PARENT_ROUNDS)
            cue_ids = read(brain, LEAF)
        d = np.array([fiber_drive(brain, LEAF, slot(j), cue_ids, stored[j])
                      for j in range(M_ITEMS)], dtype=float)
        if np.all(np.isnan(d)):
            continue
        best = int(np.nanargmax(d))
        hits += best == m
        others = np.delete(d, m)
        om = np.nanmean(others)
        if om > 0:
            margins.append(d[m] / om)
    return hits / M_ITEMS, (statistics.mean(margins) if margins else float("nan"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    trials = M_ITEMS * len(SEEDS)
    acc = statistics.mean(r[0] for r in res)
    mar = statistics.mean(r[1] for r in res)
    chance = 1.0 / M_ITEMS

    print(f"\n  n={N} k={K} beta={BETA}, parentT={PARENT_ROUNDS} "
          f"mergeR={MERGE_ROUNDS}, {M_ITEMS} slots (one per constituent)")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {trials} trials "
          f"(MIN_TRIALS={MIN_TRIALS}), chance = {chance:.4f}\n")
    print(f"  {'A1 argmax drive picks the right slot':<44}{acc:>8.4f}")
    print(f"  {'A2 margin: correct drive / mean of others':<44}{mar:>8.2f}x")
    print(f"  {'post-settling readout (depth_chain_slotted)':<44}"
          f"{chance:>8.4f}   <- at chance")

    print("\n  READING")
    if acc > chance + 0.20:
        print("    A1 HOLDS. The addressing signal IS in the LEAF->slot")
        print("    weights. The post-settling probe read chance because k-WTA")
        print("    and the attractor destroy a margin that exists beforehand --")
        print("    A3. So addressing does not need new information, it needs")
        print("    the attractor held OFF while the choice is made.")
        print("    NEXT: two-phase read -- inhibit C->C, project the cue into")
        print("    all slots, select, then disinhibit only the winner and let")
        print("    it complete. That is precisely what fiber gating provides,")
        print("    and core/inhibition.py already implements the state machine.")
    else:
        print("    A1 FAILS. The information is not in these fibers, so no")
        print("    gating scheme recovers it and the two-phase read is dead.")
        print("    Addressing would need a LEARNED POINTER (cue -> index")
        print("    assembly -> opens one slot), which is a much larger claim")
        print("    and should be treated as such.")


if __name__ == "__main__":
    main()
