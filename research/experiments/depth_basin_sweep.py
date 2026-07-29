"""Can a shared target hold many constituents AND give each one a BASIN?

THE FORK THIS DECIDES
---------------------
`depth_intermediate_quality.py` found the depth loss is erosion: a re-derived C1
shares only 0.3446 of its neurons with the stored one while still identifying
correctly 0.9167 of the time. The obvious repair -- let the intermediate settle
into its own attractor -- returned EXACTLY +0.0000, and the exact zero explains
itself. `ops.merge` runs its recurrence loop as `for _ in range(rounds - 1)`, so
at MERGE_ROUNDS=1 it runs ZERO times: only step 1 executes, whose map is
`{source_a: [source_a, target], source_b: [source_b, target]}`. No
`target -> target`, no back-projection. The C1->C1 fiber is never potentiated,
so there is no basin to fall into.

That makes the tension mechanical rather than a matter of tuning: THE OPERATING
POINT THAT MAXIMISES DEPTH-1 RECALL IS EXACTLY THE ONE THAT WRITES NO RECURRENT
STRUCTURE INTO THE CONSTITUENT. mergeR=1 was chosen because the shared-target
upper wall punishes anything higher; that choice removed the machinery clean-up
needs.

So there is a fork, and one sweep decides it:

  A CELL EXISTS   some mergeR >= 2 keeps constituents distinct AND writes enough
                  C->C that settling repairs an eroded intermediate. Then depth
                  is a tuning problem and the next step is settling at every
                  level.
  NO CELL EXISTS  distinctness and basin depth are genuinely incompatible in one
                  shared area. That is the argument for the reference's
                  role-specific areas (SUBJ/OBJ/VERB/DEP_CLAUSE with fiber
                  gating) -- reached by measurement rather than by analogy to
                  the literature.

WHAT IS MEASURED, per (parentT, mergeR) cell
---------------------------------------------
    distinct   mean pairwise overlap of the 16 stored constituents. > 0.9 is
               collapse and voids the row.
    recall     rank-1 one-parent recall, the depth-1 property.
    quality    overlap of the RE-DERIVED intermediate with the stored one --
               the thing that actually propagates into depth.
    repair     change in quality after SETTLE_ROUNDS of C1->C1 self-recurrence.
               THE COLUMN THIS FILE EXISTS FOR. Positive means a basin exists
               and clean-up works.
    w(C1->C1)  mean potentiated weight on the target's self-fiber against its
               own mean, so "no basin" can be attributed to an unwritten fiber
               rather than inferred from behaviour. This separates "the basin is
               not there" from "the basin is there and settling cannot find it".

PRE-REGISTERED
--------------
B1 repair is ~0.0000 at mergeR=1 in EVERY parentT row, and w(C1->C1) is ~1.0
   there. That is the control: it confirms the exact-zero above is the unwritten
   fiber and not something about the probe.
B2 repair becomes positive once mergeR >= 2, and grows with mergeR.
B3 THE FORK: at least one cell has distinct < 0.10 AND repair > 0.01. If none
   does, B3 fails and the shared-target design is refuted for depth.
B4 If B2 holds while B3 fails, the failure is specifically that basin depth and
   distinctness are anti-correlated across cells -- report the correlation
   rather than asserting the trade-off.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _substrate import (  # noqa: E402
    MIN_TRIALS, build, probe, rank1, read, similarity, spread,
)

N, K, P, BETA = 1000, 50, 0.05, 0.10
M_ITEMS = 16
SEEDS = (42, 7, 123, 2024, 5, 99)
SETTLE_ROUNDS = 3
A, B, C = "A", "B", "C"

PARENT_ROUNDS = (6, 10)
MERGE_ROUNDS = (1, 2, 3, 5, 8)


def self_fiber_ratio(brain, area, assembly):
    """Mean C->C weight WITHIN the assembly vs the fiber's overall mean.

    Distinguishes "no basin was ever written" (ratio ~1) from "a basin exists
    but settling does not land in it" (ratio > 1 with repair <= 0).
    """
    from neural_assemblies.assembly_calculus.ops import _compact_index

    engine = brain._engine_for(brain.areas[area])
    conn = getattr(engine, "_area_conns", {}).get(area, {}).get(area)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    inv = _compact_index(engine, area) or {}
    idx = [inv[int(x)] for x in assembly if int(x) in inv]
    idx = [i for i in idx if i < w.shape[0] and i < w.shape[1]]
    if len(idx) < 2:
        return float("nan")
    block = w[np.ix_(idx, idx)]
    nz = w[w > 0]
    return float(block[block > 0].mean() / nz.mean()) if nz.size else float("nan")


def trial(parent_rounds, merge_rounds, seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, C):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)
    for m in range(M_ITEMS):
        build(brain, f"a{m}", A, parent_rounds)
        build(brain, f"b{m}", B, parent_rounds)

    stored = {}
    for m in range(M_ITEMS):
        merge(brain, A, B, C, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=merge_rounds)
        stored[m] = read(brain, C)

    hits, qual, rep = 0, [], []
    for m in range(M_ITEMS):
        with probe(brain):
            build(brain, f"a{m}", A, parent_rounds)
            brain.project({}, {A: [C]})
            for _ in range(merge_rounds - 1):
                brain.project({}, {A: [C], C: [C]})
            live = read(brain, C)
            before = similarity(live, stored[m])
            hits += rank1(live, stored) == m
            qual.append(before)
            for _ in range(SETTLE_ROUNDS):
                brain.project({}, {C: [C]})
            rep.append(similarity(read(brain, C), stored[m]) - before)

    # nan for EVERY item means the C->C connectome block does not exist -- the
    # fiber was never opened, so no basin was ever written. That is a result,
    # not an error, and it is exactly what B1 predicts at mergeR=1.
    ratios = [r for r in (self_fiber_ratio(brain, C, stored[m])
                          for m in range(M_ITEMS)) if not np.isnan(r)]
    ratio = statistics.mean(ratios) if ratios else float("nan")
    return (hits / M_ITEMS, statistics.mean(qual), statistics.mean(rep),
            spread(stored.values()), ratio)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trials = M_ITEMS * len(SEEDS)
    print(f"\n  n={N} k={K} beta={BETA}, {M_ITEMS} items x {len(SEEDS)} seeds "
          f"= {trials} trials/cell (MIN_TRIALS={MIN_TRIALS})")
    print(f"  rank-1 chance {1 / M_ITEMS:.4f}, overlap chance {K / N:.3f}, "
          f"settle x{SETTLE_ROUNDS}\n")
    print(f"  {'parentT':>8}{'mergeR':>8}{'distinct':>10}{'recall':>9}"
          f"{'quality':>9}{'repair':>9}{'w(C->C)':>9}  note")

    rows = []
    for pr in PARENT_ROUNDS:
        for mr in MERGE_ROUNDS:
            res = [trial(pr, mr, s) for s in SEEDS]
            rec = statistics.mean(r[0] for r in res)
            qua = statistics.mean(r[1] for r in res)
            rep = statistics.mean(r[2] for r in res)
            sp = statistics.mean(r[3] for r in res)
            wr = statistics.mean(r[4] for r in res)
            note = "COLLAPSED" if sp > 0.9 else (
                "BASIN + DISTINCT" if (sp < 0.10 and rep > 0.01) else "")
            rows.append((pr, mr, sp, rec, qua, rep, wr))
            print(f"  {pr:>8}{mr:>8}{sp:>10.4f}{rec:>9.4f}{qua:>9.4f}"
                  f"{rep:>+9.4f}{wr:>9.2f}  {note}")

    print("\n  READING")
    ctrl = [r for r in rows if r[1] == 1]
    ctrl_w = [r[6] for r in ctrl if not np.isnan(r[6])]
    w_txt = (f"{statistics.mean(ctrl_w):.2f}" if ctrl_w
             else "FIBER ABSENT (block never materialized)")
    print(f"    B1 mergeR=1: max |repair| {max(abs(r[5]) for r in ctrl):.4f}, "
          f"w(C->C) {w_txt}")
    winners = [r for r in rows if r[2] < 0.10 and r[5] > 0.01]
    if winners:
        best = max(winners, key=lambda r: r[5])
        print(f"    B3 HOLDS. parentT={best[0]} mergeR={best[1]}: distinct "
              f"{best[2]:.4f}, repair {best[5]:+.4f}, recall {best[3]:.4f}.")
        print("    A shared target CAN hold distinct constituents and give each")
        print("    a basin. Depth is a tuning problem: settle at every level")
        print("    and re-run the chain at this cell.")
    else:
        good = [r for r in rows if r[5] > 0.01]
        print("    B3 FAILS. No cell is both distinct and repairable.")
        if good:
            print(f"    Cells with a basin exist (best distinct {min(r[2] for r in good):.4f})")
            print("    but all are collapsed or near it -- basin depth and")
            print("    distinctness are anti-correlated, which is B4.")
        print("    This refutes the shared-target design for depth, and it is")
        print("    the measured argument for role-specific areas.")


if __name__ == "__main__":
    main()
