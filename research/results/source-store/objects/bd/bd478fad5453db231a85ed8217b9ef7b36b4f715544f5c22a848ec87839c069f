"""Does the CUE reinstate the parent assembly? (blocks every recall result)

WHY THIS EXISTS
---------------
`merge_gated_target.py` reported a column it was not looking for. `A drift` --
the overlap between a parent assembly as it stood when the merge was performed
and the SAME parent re-driven by its own stimulus afterwards -- came back at
0.0231, against a chance rate of k/n = 0.0225. Chance. And it was 0.0231 in the
rounds=2 control too, where nothing had collapsed, so it is not a downstream
consequence of the collapse.

If that number is what it appears to be, then every "recall from one parent"
measurement in this repo has been cueing with an assembly UNRELATED to the one
whose synapses onto the target were actually written. The merged constituent
would then be unrecallable for a reason that has nothing to do with merge:

    trained:  A holds assembly P_m  ->  potentiates P_m -> C_m
    probed:   A holds assembly Q_m  ->  Q_m has no potentiated path anywhere

and the readout would return chance, exactly as observed, no matter how well
merge worked. Four files cue this way -- `merge_recall_control.py`,
`merge_regime.py`, `merge_chain_primitive.py`, `merge_gated_target.py`, all via
`project(brain, stim, src, rounds=4)` -- so this is not a local defect. It sits
underneath the conclusion currently recorded as "merge's defining property does
not hold on this substrate", which is a strong claim resting on probes that may
never have presented the right input.

Measuring this is therefore not optional and not a detail. It is the positive
control that should have run before any retrieval experiment: BEFORE asking
whether C can be recovered from A, verify that A can be recovered at all.

THREE CANDIDATE CAUSES, and the design separates them
------------------------------------------------------
  C1 TOO FEW ROUNDS. Parents are BUILT with 12 rounds and CUED with 4. Four
     rounds may simply not reach the same fixed point. Swept directly.
  C2 STARTING-STATE CONTAMINATION. The cue begins from whatever A last held --
     after the merge phase, that is the LAST item's assembly, not a clean slate.
     `project` adds stimulus drive on top of it. Tested by clearing A's winners
     before cueing.
  C3 THE WEIGHTS GENUINELY MOVED. Later items, and then the merges, rewrite A.
     The assembly that existed at merge time may no longer be a fixed point of
     the area at all. This is a real fact about the substrate rather than a
     probe defect, and it has a very different consequence: it would mean a
     stimulus-driven assembly is not stable under subsequent unrelated learning,
     which undermines far more than merge.

C1 and C2 are probe defects and are fixable. C3 is not. The phase axis is what
separates them: `immediate` (re-cue the item the instant it is built, nothing
else has happened yet) isolates C1/C2 with C3 held at zero. If `immediate`
already fails, no amount of C3 is needed to explain anything and the probe is
simply wrong. If `immediate` succeeds and `after merges` fails, C3 is real and
the magnitude of the drop measures it.

TWO METRICS, because self-overlap alone cannot decide it
---------------------------------------------------------
  SELF   overlap(cue(m), parent_at_merge_time(m)); chance = k/n.
  ID     is parent_m the rank-1 best match for cue(m) among ALL M parents;
         chance = 1/M. This is the one that matters for retrieval. A cue can be
         a degraded copy -- SELF well below 1.0 -- and still be unambiguously
         the right parent, in which case the probe is usable. It is only broken
         if ID fails too.

PRE-REGISTERED
--------------
F1 `immediate` scores SELF near 1.0 and ID at 1.0. If it does not, the probe is
   broken at the most basic level and every recall result in this line of work
   is void rather than negative.
F2 SELF rises monotonically with cue rounds, and 12 (matching the build) beats
   4 (what every existing probe uses). This is C1, and it is the cheapest thing
   that could be wrong.
F3 Clearing A's winners first beats carrying them. This is C2.
F4 The `after merges` phase scores below `after build`, and the gap is C3.
F5 THE DECIDING ONE: at the best (rounds, start) cell, ID in the `after merges`
   phase clears chance by a wide margin. If it does, the probe is fixable, the
   existing at-chance recall results must be RE-RUN with the fixed probe before
   they mean anything, and the "merge does not hold" conclusion is suspended
   rather than confirmed. If ID stays at chance even at 24 rounds from a clean
   start, then C3 is severe: stimulus-driven assemblies do not survive
   subsequent learning in a shared area, and that is a much larger finding than
   anything about merge.

No claim here is worth reporting from one seed; every cell is meaned over seeds
and the trial count is printed on the header line.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

A, B, C = "A", "B", "C"

N, K, P, BETA = 2000, 45, 0.01, 0.05
M_ITEMS = 8
SEEDS = (42, 7, 123)
BUILD_ROUNDS = 12          # what every existing experiment builds parents with
MERGE_ROUNDS = 2           # the non-collapsed regime, so C3 is not confounded
                           # with the target collapse measured elsewhere
CUE_ROUNDS = (1, 2, 4, 8, 12, 24)
STARTS = ("carry", "clear")


def cue(brain, stim, area, rounds, start):
    """Drive `area` from `stim` and return the winners, changing nothing.

    `read_only()` is what makes this a probe rather than a training step: it
    blocks weight change AND recruitment, and restores winners afterwards. Using
    `frozen()` here would leave the area recruited differently after each
    measurement, which is the contamination already root-caused in the ERP work.
    """
    from neural_assemblies.assembly_calculus.ops import project

    with brain.read_only():
        if start == "clear":
            a = brain.areas[area]
            a.unfix_assembly()
            a.winners = np.asarray(a.winners)[:0]
        project(brain, stim, area, rounds=rounds)
        return np.asarray(brain.areas[area].winners, dtype=np.int64)


def score(brain, parents, rounds, start):
    """SELF overlap and rank-1 ID for one (rounds, start) cell."""
    from neural_assemblies.assembly_calculus.assembly import overlap

    self_ov, ident = [], 0
    for m in range(M_ITEMS):
        live = cue(brain, f"a{m}", A, rounds, start)
        self_ov.append(overlap(live, parents[m]))
        best = max((overlap(live, p), j) for j, p in parents.items())[1]
        ident += (best == m)
    return statistics.mean(self_ov), ident / M_ITEMS


def trial(seed):
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import merge, project
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (A, B, C):
        brain.add_area(area, N, K, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        brain.add_stimulus(f"b{m}", K)

    # PHASE `immediate`: re-cue each item the instant it is built, before
    # anything else has touched the area. This is the C3=0 condition -- any
    # failure here is the probe alone.
    parents, immediate = {}, []
    for m in range(M_ITEMS):
        parents[m] = np.asarray(
            project(brain, f"a{m}", A, rounds=BUILD_ROUNDS).winners,
            dtype=np.int64)
        project(brain, f"b{m}", B, rounds=BUILD_ROUNDS)
        immediate.append(overlap(
            cue(brain, f"a{m}", A, BUILD_ROUNDS, "carry"), parents[m]))

    pre_spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(parents.values(), 2))

    after_build = {(r, s): score(brain, parents, r, s)
                   for r in CUE_ROUNDS for s in STARTS}

    for m in range(M_ITEMS):
        merge(brain, A, B, C, stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=MERGE_ROUNDS)

    after_merge = {(r, s): score(brain, parents, r, s)
                   for r in CUE_ROUNDS for s in STARTS}

    return statistics.mean(immediate), pre_spread, after_build, after_merge


#: (n, k, p) and which file runs in it. `k*p` is the expected number of
#: stimulus synapses landing on each neuron of the area -- the signal the
#: stimulus actually delivers. Added after the first run, which found SELF
#: pinned at 0.0231 in EVERY cell: unchanged across a 24x rounds sweep and both
#: start states. A number that does not move under a 24x change in its own
#: independent variable is not measuring a dynamical process, so the file's own
#: "C3 is real" verdict was not accepted. The candidate it points to instead is
#: the REGIME: at k*p = 0.45 almost every neuron receives ZERO stimulus input,
#: the top-k is decided by random recruitment rather than by the stimulus, and
#: an assembly is CREATED on each projection rather than retrieved. If that is
#: right, fidelity is a function of k*p and the affected results are exactly the
#: low-k*p ones -- which is a much narrower claim than "every recall result is
#: void", and it is checkable one row at a time.
REGIMES = [
    (1000,  32, 0.01, "merge_regime rows 2-4"),
    (2000,  45, 0.01, "merge_gated_target, this file"),
    (10000, 100, 0.01, "merge_regime row 5"),
    (30000, 173, 0.01, "merge_regime row 6"),
    (1000,  50, 0.05, "merge_recall_control, merge_chain_primitive"),
    (1000, 100, 0.05, "denser control"),
    (1000,  50, 0.10, "denser control"),
]


def regime_fidelity(n, k, p, seed):
    """Immediate re-cue fidelity: build a parent, re-cue it, nothing between.

    The zero-interference condition. Whatever this returns is the CEILING on
    every retrieval measurement taken in this regime, because no experiment can
    recover a target from a parent the cue cannot itself reinstate.
    """
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, seed=seed)
    brain.add_area(A, n, k, beta=BETA)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", k)

    # WITHIN-EPISODE CONVERGENCE, measured separately from re-selectability
    # because they are different properties and only the first is what
    # [PNAS20] sec 2 proves: "after O(log n) rounds the set of winners
    # converges, with overlap > 0.95 between consecutive rounds". That is
    # stability WITHIN one projection. Re-selectability -- the same stimulus
    # presented later picking the same assembly again -- is a further claim,
    # and it is the one every cued retrieval probe silently depends on.
    # Convergence does not imply it: an area can settle to a fixed point that
    # is a function of the trajectory as well as the input.
    conv = []
    parents = {}
    for m in range(M_ITEMS):
        brain.project({f"a{m}": [A]}, {})
        prev = np.asarray(brain.areas[A].winners, dtype=np.int64)
        step = []
        for _ in range(BUILD_ROUNDS - 1):
            brain.project({f"a{m}": [A]}, {A: [A]})
            cur = np.asarray(brain.areas[A].winners, dtype=np.int64)
            step.append(overlap(prev, cur))
            prev = cur
        conv.append(step[-1] if step else float("nan"))
        parents[m] = prev

    self_ov, ident = [], 0
    for m in range(M_ITEMS):
        live = cue(brain, f"a{m}", A, BUILD_ROUNDS, "carry")
        self_ov.append(overlap(live, parents[m]))
        ident += max((overlap(live, q), j)
                     for j, q in parents.items())[1] == m
    spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(parents.values(), 2))
    return (statistics.mean(self_ov), ident / M_ITEMS, spread,
            statistics.mean(conv))


def sweep_regimes() -> None:
    print("\n\n  REGIME SWEEP -- is cue fidelity a function of k*p?")
    print("  k*p = expected stimulus synapses per neuron. Immediate re-cue,")
    print("  so this is the CEILING on retrieval in each regime.")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds; ID chance = "
          f"{1.0 / M_ITEMS:.4f}\n")
    print("  CONV = overlap between the LAST TWO rounds of a single projection")
    print("  ([PNAS20] sec 2 expects > 0.95). SELF = the same stimulus re-cued")
    print("  later. These are different properties and CONV does not imply SELF.\n")
    print(f"  {'n':>7}{'k':>5}{'p':>6}{'k*p':>7}{'CONV':>8}{'SELF':>8}"
          f"{'/chance':>9}{'ID':>8}  used by")
    for n, k, p, who in REGIMES:
        res = [regime_fidelity(n, k, p, s) for s in SEEDS]
        sv = statistics.mean(r[0] for r in res)
        iv = statistics.mean(r[1] for r in res)
        cv = statistics.mean(r[3] for r in res)
        ch = k / n
        print(f"  {n:>7}{k:>5}{p:>6.2f}{k * p:>7.2f}{cv:>8.4f}{sv:>8.4f}"
              f"{sv / ch:>9.1f}{iv:>8.4f}  {who}")
    print("\n  Read the two columns as a 2x2, because they mean different things:")
    print("   CONV high, SELF high -> the assembly is a stable FUNCTION of the")
    print("     stimulus. Cued retrieval is sound in this regime.")
    print("   CONV high, SELF low  -> projection settles, but to a fixed point")
    print("     that depends on the trajectory as well as the input. The area")
    print("     CREATES an assembly rather than retrieving one, and no cued")
    print("     probe can work -- this is a substrate property, not a bug.")
    print("   CONV low             -> the regime is too sparse for the stimulus")
    print("     to select anything (watch k*p, the expected stimulus synapses")
    print("     per neuron). Nothing measured in this row means anything.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = [trial(s) for s in SEEDS]
    chance_ov, chance_id = K / N, 1.0 / M_ITEMS

    print(f"\n  n={N} k={K} p={P} beta={BETA}, {M_ITEMS} items x "
          f"{len(SEEDS)} seeds = {M_ITEMS * len(SEEDS)} trials/cell")
    print(f"  build rounds={BUILD_ROUNDS}, merge rounds={MERGE_ROUNDS}")
    print(f"  chance: SELF overlap = k/n = {chance_ov:.4f}, "
          f"ID = 1/M = {chance_id:.4f}")

    imm = statistics.mean(r[0] for r in res)
    spread = statistics.mean(r[1] for r in res)
    print(f"\n  F1  immediate re-cue (built and re-cued at {BUILD_ROUNDS} "
          f"rounds, nothing in between): SELF = {imm:.4f}")
    print(f"      parents mutually distinct: pairwise overlap = {spread:.4f} "
          f"(chance {chance_ov:.4f})")
    if imm < 0.5:
        print("      *** F1 FAILS. The probe does not reproduce an assembly")
        print("      even with zero intervening learning, so every recall")
        print("      result built on this cue is VOID, not negative. ***")

    for phase, idx in (("AFTER BUILD (all items built, no merges)", 2),
                       ("AFTER MERGES", 3)):
        print(f"\n  {phase}")
        print(f"  {'cue rounds':>11}" +
              "".join(f"{s + ' SELF':>13}{s + ' ID':>11}" for s in STARTS))
        for r in CUE_ROUNDS:
            row = f"  {r:>11}"
            for s in STARTS:
                sv = statistics.mean(x[idx][(r, s)][0] for x in res)
                iv = statistics.mean(x[idx][(r, s)][1] for x in res)
                row += f"{sv:>13.4f}{iv:>11.4f}"
            print(row)

    best = max(((statistics.mean(x[3][(r, s)][1] for x in res), r, s)
                for r in CUE_ROUNDS for s in STARTS))
    print(f"\n  F5  best AFTER-MERGES cell: ID = {best[0]:.4f} at "
          f"rounds={best[1]}, start={best[2]} (chance {chance_id:.4f})")
    if best[0] > chance_id + 0.20:
        print("      The probe is FIXABLE. Every at-chance recall result in")
        print("      merge_recall_control / merge_regime / merge_chain_primitive")
        print("      / merge_gated_target used rounds=4 and must be RE-RUN at")
        print(f"      rounds={best[1]}, start={best[2]} before it means anything.")
        print("      The 'merge does not hold' conclusion is SUSPENDED.")
    else:
        print("      The probe is NOT fixable by rounds or start state. C3 is")
        print("      real: a stimulus-driven assembly does not survive")
        print("      subsequent learning in a shared area. That is a larger")
        print("      finding than anything about merge, and it is the thing to")
        print("      chase next -- UNLESS the sweep below shows fidelity")
        print("      tracking k*p, in which case this regime was simply too")
        print("      sparse for a stimulus to select anything.")
    sweep_regimes()


if __name__ == "__main__":
    main()
