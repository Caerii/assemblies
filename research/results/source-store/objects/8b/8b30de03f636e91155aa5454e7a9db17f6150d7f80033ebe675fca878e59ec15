"""Is merge's collapse caused by the TARGET'S SELF-RECURRENCE? (gating test)

THE SQUEEZE, AND WHY GATING IS THE RIGHT LEVER AT IT
-----------------------------------------------------
`merge_regime.py` established that a constituent LEXICON -- many merges sharing
one target area, each required to stay distinct AND remain retrievable from its
parts -- is pulled apart by one knob:

    more rounds -> more Hebbian potentiation on parent->target, which is what
                   one-parent recall needs
    more rounds -> the shared target's recurrence pulls every constituent onto
                   the same attractor, which is what distinctness forbids

Measured, and n does not buy a way out (overlap 0.99-1.00 at rounds=50 even at
n=30000). But notice what that argument assumes: that "rounds" is ONE knob. It
is one knob only because `ops.merge` hardcodes a single projection map in which
`target: [target, source_a, source_b]` rides along with `source_a: [.., target]`
on every round after the first. The two effects above travel on DIFFERENT
FIBERS:

    A -> C, B -> C     writes the path recall needs      (wanted, all rounds)
    C -> C             converges the assembly            (suspected collapse)
    C -> A, C -> B     the two-way bind [PNAS20] sec 3   (wanted, keep)

Fiber gating separates them. That is exactly what the primitive is for
(`core/inhibition.py`, `fiber.py`): inhibition decides WHICH PROJECTIONS HAPPEN,
and an inhibited fiber carries no signal and therefore no plasticity, while its
weights persist. So the squeeze may not be a property of merge at all -- it may
be an artifact of running every fiber open on every round.

FIVE ARMS, chosen so a failure LOCATES rather than just reports
---------------------------------------------------------------
    baseline        C->C open on every round after the first  (== ops.merge)
    clear           baseline, but C's winners cleared before each merge
    gate-late       C->C closed for the first half of the rounds, open after
    gate-none       C->C never open; C->A, C->B kept
    gate-none+clear both

`clear` is in because it is the cheap alternative explanation and the reference
does it (`prepare_targets` CLEARS an area the new word reaches, so nothing
blends with the previous occupant). If activity carry-over were the mechanism,
`clear` alone would fix it and no gating would be needed.

PRE-REGISTERED
--------------
G1 `gate-none` at 50 rounds keeps pairwise overlap near the rounds=2 level
   (< 0.10), i.e. NO collapse. Collapse is driven by C->C, not by capacity.
G2 `gate-none` weight ratio EXCEEDS the best seen so far (1.56): 50 rounds of
   A->C potentiation without the collapse that previously destroyed it (in the
   collapsed cells the ratio fell BELOW 1).
G3 `gate-none` one-parent recall clears chance. This is the real test, and it is
   separable from G1/G2: both can hold while G3 fails, and that would say the
   A->C path being WRITTEN is not sufficient for one-parent SELECTION -- which
   moves the problem to the readout/competition and away from the write.
G4 `clear` alone changes nothing. Registered so the cheap explanation is ruled
   out by measurement rather than by argument.
G5 `gate-late` lands between `baseline` and `gate-none` on BOTH axes -- the
   squeeze turned into a dial, which is the strongest form of the claim.

REFUTATION. If `gate-none` collapses too, C->C is not the mechanism, the
collapse is genuine shared-area capacity, and the work moves to the reference's
architecture (one area per constituent ROLE) rather than to gating.

TWO-STAGE, and the split is registered here so it is not a post-hoc choice
--------------------------------------------------------------------------
Stage 1 (PILOT) screens on DISTINCTNESS, where the effect is enormous
(0.99 vs 0.01) and 8 items x 3 seeds is ample. Stage 2 powers up ONLY the
surviving arms to 16 items x 6 seeds = 96 trials to test RECALL, where the
effect is small. `merge_recall_control.py` is on record that 24 trials cannot
separate 0.125 from 0.375, and that lesson is what this split exists to honour.

RESULT (2026-07-28): G1 REFUTED, AND SO ARE BOTH ALTERNATIVES
--------------------------------------------------------------
    STAGE 0 -- POSITIVE CONTROL, rounds=2 (n=2000 k=45, 8 merges x 3 seeds)
    arm                 C ovlp  onset  A ovlp  w ratio   from A   from B
    baseline            0.0053    8.0  0.0148     1.32   0.1250   0.1667
    gate-none           0.0220    8.0  0.0161     1.67   0.1667   0.2500
    feed-forward        0.0270    8.0  0.0156     1.20   0.1667   0.1667

    STAGE 1 -- rounds=50
    baseline            0.9981    1.0  0.1804     1.79   0.1250   0.1250
    clear               0.9981    1.0  0.1804     1.79   0.1250   0.1250
    gate-late           0.9963    1.0  0.1907     1.56   0.1250   0.1250
    gate-none           0.9963    1.0  0.1907     1.56   0.1250   0.1250
    gate-none+clear     0.9963    1.0  0.1907     1.56   0.1250   0.1250
    no-backproj         0.9669    1.0  0.1339     2.77   0.1250   0.1250
    feed-forward        0.9344    1.0  0.1508     1.17   0.0833   0.1667

The control passes -- every arm is distinct at rounds=2 -- so the deep table is
a fact about rounds and not about the harness. Then four things fall out, and
together they leave only one candidate standing:

  NOT C->C.        `feed-forward` closes EVERY outgoing fiber of C and still
                   collapses (0.9344). Gating the self-recurrence was the
                   hypothesis; it is wrong, and G1/G5 are refuted.
  NOT the bind.    `feed-forward` and `no-backproj` have no C->A / C->B either.
  NOT the sources. A's constituents stay distinct throughout (0.13-0.19 against
                   a collapsed 0.93-1.00 in C). The parents are fine; the target
                   is not.
  NOT capacity.    `onset` is 1.0 in every deep cell: merge 1 already lands on
                   merge 0's assembly. A capacity limit degrades as items
                   accumulate. This is immediate, at two items, in an area of
                   2000 neurons holding assemblies of 45.

So the collapse happens on the PURE FEED-FORWARD path, at the second item, with
distinct inputs. The only thing left is the target cells themselves: under 50
rounds of Hebbian potentiation the cells that win accumulate strengthened
synapses from EVERY source assembly indiscriminately, and thereafter win for any
input. That is the high-degree-hub failure this repo has already met once, in
recurrence, and already has a mechanism for -- one-time initial weight
normalisation (`norm_init`), plus the per-area `refractory_period` /
`inhibition_strength` knobs on `add_area`. Testing those is the next branch, and
note that it is an AREA-level mechanism: no arrangement of fibers can fix it,
which is why every arm above agreed.

BLOCKING FINDING, and it outranks the above: THE CUE MAY NOT REINSTATE THE
PARENT. `A drift` -- overlap between a parent as it was at merge time and the
same parent re-driven by its own stimulus afterwards -- is 0.0231, against a
chance rate of k/n = 0.0225. Chance. It is 0.0231 in the rounds=2 control too,
where nothing collapsed, so it is not a consequence of the collapse. If the cue
presents an assembly unrelated to the one whose synapses onto C were actually
written, then no recall column in this file -- or in `merge_recall_control.py`,
`merge_regime.py`, or `merge_chain_primitive.py`, which all cue the same way
with `project(brain, stim, src, rounds=4)` -- is measuring retrieval. Every
at-chance recall number in this line of work is suspect until that is settled,
which is what `merge_cue_fidelity.py` exists to do. Do not read the recall
columns above until it has run.

TWO CORRECTIONS FOUND WHILE READING, recorded so they are not re-derived
-------------------------------------------------------------------------
1. THE 0.781 FIGURE IS NOT A TARGET. `merge_recall_control.py` left three
   possibilities open for the rank-1 0.781 quoted in `parser_mixins/phrases.py`.
   It is possibility (2), and the source comment says so itself: it "asks
   whether the top-scoring stored constituent CONTAINS that parent" -- a
   strictly easier question than "is it the right constituent", and its quoted
   chance of 0.031 (= 1/32) is the chance rate for the harder one. Two further
   marks against it: no producing code exists anywhere in the repo, and
   `_shared.py` records it as EXACTLY 0.781 in all five cells of a sweep that
   moved overlap from 0.299 to 0.562 -- a metric that does not move under a 20x
   change in rounds is not measuring what it claims. Do not aim at it.
2. THE REFERENCE PARSER NEVER CALLS `merge`. `grep -c merge` on
   `.reference/dmitropolsky-assemblies/{parser,recursive_parser}.py` returns 1,
   and it is a comment. Composition there is projection plus gating, and the
   parse RESULT is `getActivatedFibers()` -- which fibers fired, i.e. a
   dependency graph -- never an assembly recalled from a part. `merge` lives
   only in `simulations.py` as a standalone one-pair demonstration. So the
   retrieval property this file tests is one the reference asserts (in the
   paper) but never itself relies on, which is consistent with it never having
   been pressured into a lexicon.
"""

from __future__ import annotations

import itertools
import math
import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

A, B, C = "A", "B", "C"

#: Pilot regime. Rounds are high on purpose: 50 is where `merge_regime.py`
#: measured total collapse, so it is the cell the gating claim has to rescue.
#: n is 2000 rather than 30000 because the distinctness effect being screened is
#: 0.99-vs-0.01 and was shown to be n-independent.
PILOT = dict(n=2000, k=45, p=0.01, beta=0.05, rounds=50, m_items=8,
             seeds=(42, 7, 123))
#: POSITIVE CONTROL, and the table is uninterpretable without it. `merge_regime`
#: measured distinct constituents at rounds=2, so if EVERY arm collapses at
#: rounds=50 the first question is whether this harness can produce distinctness
#: at all. Same brains, same readout, one parameter changed. A shallow table that
#: is distinct everywhere licenses reading the deep table as a fact about rounds;
#: a shallow table that also collapses says the harness is broken and nothing
#: above it means anything.
PILOT_SHALLOW = dict(PILOT, rounds=2)
POWERED = dict(n=2000, k=45, p=0.01, beta=0.05, rounds=50, m_items=16,
               seeds=(42, 7, 123, 2024, 5, 99))

#: (label, self_recurrence, clear_target, back_project). `self_recurrence` is
#: the fraction of rounds at the END during which C->C is open; 1.0 reproduces
#: `ops.merge`. `back_project` controls the C->A / C->B fiber.
#:
#: The last two arms are a DIAGNOSTIC FOLLOW-UP, added after the first run, and
#: are labelled as such so they are never read as pre-registered. The five
#: original arms all collapsed -- but `gate-late` and `gate-none` returned
#: IDENTICAL overlap (0.9963) and identical weight ratio (1.56) to every printed
#: digit, despite differing in 24 of 49 rounds. Two arms that differ in the
#: projection map cannot agree that exactly unless the collapse is already
#: complete before the arms diverge, i.e. the cause is NOT on the C->C fiber at
#: all. The remaining candidate is the OTHER fiber merge opens: C->A and C->B
#: write back into the parents on every round, so the sources themselves may be
#: what collapses -- after which every merge receives near-identical input and a
#: shared target is a foregone conclusion. `SOURCE overlap in A` is reported
#: alongside, because that hypothesis is a direct measurement rather than an
#: inference from the target.
ARMS = [
    ("baseline",         1.0, False, True),
    ("clear",            1.0, True,  True),
    ("gate-late",        0.5, False, True),
    ("gate-none",        0.0, False, True),
    ("gate-none+clear",  0.0, True,  True),
    # diagnostic follow-up, NOT pre-registered:
    ("no-backproj",      1.0, False, False),
    ("feed-forward",     0.0, False, False),
]


def gated_merge(brain, source_a, source_b, target, stim_a, stim_b, rounds,
                *, self_recurrence, clear_target, back_project=True):
    """`ops.merge` with the target's self-recurrence on its own switch.

    Everything else is held identical to `ops.merge`: parents are STIMULUS-
    DRIVEN (never fixed -- the engine short-circuits plasticity into a fixed
    area, which is documented there as silently discarding the very two-way
    connectivity merge exists to create), round 1 carries no target recurrence,
    and the C->A / C->B back-projection is kept on every later round because
    that back-projection IS the bind. The ONLY thing this function varies is
    whether `target` appears in its own destination list.
    """
    from neural_assemblies.assembly_calculus.ops import _snap

    if clear_target:
        area = brain.areas[target]
        area.unfix_assembly()
        area.winners = np.asarray(area.winners)[:0]

    stim = {stim_a: [source_a], stim_b: [source_b]}
    later = rounds - 1
    # Open C->C only for the final `self_recurrence` fraction of the rounds, so
    # the new constituent is first selected by its PARENTS alone and only then
    # allowed to converge.
    open_from = later - int(round(self_recurrence * later))

    brain.project(stim, {source_a: [source_a, target],
                         source_b: [source_b, target]})
    for i in range(later):
        tgt_dsts = [source_a, source_b] if back_project else []
        if i >= open_from and self_recurrence > 0:
            tgt_dsts = [target] + tgt_dsts
        proj = {source_a: [source_a, target], source_b: [source_b, target]}
        # An empty destination list is not the same as an absent key: omit the
        # target entirely when every one of its outgoing fibers is closed, so
        # the engine is never handed a source with nowhere to go.
        if tgt_dsts:
            proj[target] = tgt_dsts
        brain.project(stim, proj)
    return _snap(brain, target)


def weight_ratio(brain, src_area, tgt_area, src_winners, tgt_winners, rng):
    """Mean src->tgt weight onto the merged assembly vs onto a random set.

    Same measurement as `merge_regime.py`, kept identical so the numbers are
    directly comparable to the table recorded there. Reads the connectome rather
    than inferring from activity, which is what separates "merge did not write
    the path" from "the path is there and the readout cannot use it".
    """
    engine = brain._engine_for(brain.areas[tgt_area])
    conn = getattr(engine, "_area_conns", {}).get(src_area, {}).get(tgt_area)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    rows = [int(i) for i in src_winners if int(i) < w.shape[0]]
    cols = [int(j) for j in tgt_winners if int(j) < w.shape[1]]
    if not rows or not cols:
        return float("nan")
    on = float(w[np.ix_(rows, cols)].mean())
    other = [j for j in range(w.shape[1]) if j not in set(cols)]
    if not other:
        return float("nan")
    ctrl = list(rng.choice(other, size=min(len(cols), len(other)),
                           replace=False))
    off = float(w[np.ix_(rows, ctrl)].mean())
    return on / off if off > 0 else float("nan")


def trial(arm, cfg, seed):
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain

    _, self_rec, clear, backproj = arm
    m_items, rounds = cfg["m_items"], cfg["rounds"]
    rng = np.random.default_rng(seed)
    brain = Brain(p=cfg["p"], seed=seed)
    for area in (A, B, C):
        brain.add_area(area, cfg["n"], cfg["k"], beta=cfg["beta"])
    for m in range(m_items):
        brain.add_stimulus(f"a{m}", cfg["k"])
        brain.add_stimulus(f"b{m}", cfg["k"])

    parents_a, parents_b = {}, {}
    for m in range(m_items):
        parents_a[m] = project(brain, f"a{m}", A, rounds=12)
        parents_b[m] = project(brain, f"b{m}", B, rounds=12)

    # BEFORE any merge. Area A is itself a shared target -- 8 stimuli projected
    # into one area -- so it can collapse on its own, with no merge involved. If
    # it has, every number below is about A and nothing about merge, and the
    # arms would be expected to agree exactly (which is what the first run
    # showed). Measured rather than assumed.
    src_pre = {m: np.asarray(parents_a[m].winners) for m in range(m_items)}
    pre_spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(src_pre.values(), 2))

    stored = {}
    for m in range(m_items):
        stored[m] = gated_merge(brain, A, B, C, f"a{m}", f"b{m}", rounds,
                                self_recurrence=self_rec, clear_target=clear,
                                back_project=backproj)

    # SOURCE state after the merge phase. Merge writes C->A on every round, so
    # the parents are not passive: if THEY collapse, every later merge receives
    # near-identical input and a shared target is a foregone conclusion. Re-driven
    # under read_only() so measuring does not itself train.
    src_post = {}
    for m in range(m_items):
        with brain.read_only():
            project(brain, f"a{m}", A, rounds=4)
            src_post[m] = np.asarray(brain.areas[A].winners, dtype=np.int64)
    src_spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(src_post.values(), 2))
    src_drift = statistics.mean(
        overlap(src_pre[m], src_post[m]) for m in range(m_items))

    ratios = [weight_ratio(brain, A, C, parents_a[m].winners,
                           stored[m].winners, rng) for m in range(m_items)]
    ratios = [r for r in ratios if not math.isnan(r)]

    # ONSET: the first merge that lands on the SAME assembly as merge 0. "It
    # collapsed" is not a location; whether it goes at merge 1 or merge 6 is the
    # difference between a shared-attractor artifact and a real capacity limit.
    onset = next((m for m in range(1, m_items)
                  if overlap(stored[m], stored[0]) > 0.5), m_items)

    def cue(m, from_a):
        src, stim = (A, f"a{m}") if from_a else (B, f"b{m}")
        # Cue then settle. Opening with `C: [C]` lets whatever training left in
        # the target vote for itself BEFORE the cue arrives, and it wins --
        # measured in `merge_chain_primitive.py` as recall pinned at exactly
        # 1/M regardless of the cue. The readout is deliberately the SAME in
        # every arm, including the arms trained with C->C closed, so a recall
        # difference is attributable to training and not to the probe.
        with brain.read_only():
            project(brain, stim, src, rounds=4)
            brain.project({}, {src: [C]})
            for _ in range(min(rounds, 10) - 1):
                brain.project({}, {src: [C], C: [C]})
            live = np.asarray(brain.areas[C].winners, dtype=np.int64)
        return max((overlap(live, asm), j) for j, asm in stored.items())[1] == m

    from_a = sum(cue(m, True) for m in range(m_items)) / m_items
    from_b = sum(cue(m, False) for m in range(m_items)) / m_items
    spread = statistics.mean(
        overlap(x, y) for x, y in itertools.combinations(stored.values(), 2))
    return dict(from_a=from_a, from_b=from_b, spread=spread, onset=onset,
                src_spread=src_spread, src_drift=src_drift,
                ratio=(statistics.mean(ratios) if ratios else float("nan")))


def run(cfg, arms, header):
    chance = 1.0 / cfg["m_items"]
    print(f"\n  {header}")
    print(f"  n={cfg['n']} k={cfg['k']} p={cfg['p']} beta={cfg['beta']} "
          f"rounds={cfg['rounds']}")
    print(f"  {cfg['m_items']} merges x {len(cfg['seeds'])} seeds = "
          f"{cfg['m_items'] * len(cfg['seeds'])} trials/cell, "
          f"chance = {chance:.4f}\n")
    print(f"  TARGET columns describe area C; SOURCE columns describe area A.")
    print(f"  onset = first merge landing on merge 0's assembly "
          f"({cfg['m_items']} = never).")
    print(f"  drift = overlap between a parent before and after the merge "
          f"phase (1.0 = untouched).\n")
    print(f"  {'arm':<18}{'C ovlp':>8}{'onset':>7}{'A ovlp':>8}{'A drift':>8}"
          f"{'w ratio':>9}{'from A':>8}{'from B':>8}  note")

    surviving = []
    for arm in arms:
        res = [trial(arm, cfg, s) for s in cfg["seeds"]]
        mean = lambda key: statistics.mean(r[key] for r in res)
        sp, fa, fb = mean("spread"), mean("from_a"), mean("from_b")
        if sp > 0.9:
            note = "COLLAPSED"
        elif sp < 0.10:
            note = "distinct"
            surviving.append(arm)
        else:
            note = "partial"
        if fa > chance + 0.05 and fb > chance + 0.05:
            note += " + RECALLS"
        print(f"  {arm[0]:<18}{sp:>8.4f}{mean('onset'):>7.1f}"
              f"{mean('src_spread'):>8.4f}{mean('src_drift'):>8.4f}"
              f"{mean('ratio'):>9.2f}{fa:>8.4f}{fb:>8.4f}  {note}")
    return surviving


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    shallow = run(PILOT_SHALLOW, ARMS,
                  "STAGE 0 -- POSITIVE CONTROL at rounds=2 (must be distinct)")
    if not shallow:
        print("\n  CONTROL FAILED: nothing is distinct even at rounds=2, where")
        print("  `merge_regime.py` measured overlap 0.0119. The harness, not")
        print("  the substrate, is producing the collapse. STOP -- no row below")
        print("  is interpretable.")
        return

    surviving = run(PILOT, ARMS, "STAGE 1 -- PILOT, screening on DISTINCTNESS")
    print("\n  G1 asks whether `gate-none` is in the distinct set above.")
    print("  G4 asks whether `clear` is NOT (it should look like baseline).")
    print("  Read the SOURCE columns first: if `A ovlp` is also collapsed, the")
    print("  target's overlap is a CONSEQUENCE and gating C->C could never have")
    print("  helped, whatever the target column says.")

    if not surviving:
        print("\n  NO ARM STAYS DISTINCT AT rounds=50. C->C alone is not the")
        print("  mechanism. Which mechanism it IS depends on the source columns")
        print("  and on `feed-forward` (every C fiber closed): if that arm also")
        print("  collapses with SOURCES intact, the cause is rich-get-richer on")
        print("  the A->C fiber itself, not on any fiber this file can gate.")
        return
    if os.environ.get("GATED_STAGE2", "1") == "0":
        print("\n  stage 2 skipped (GATED_STAGE2=0)")
        return
    run(POWERED, surviving,
        "STAGE 2 -- POWERED, surviving arms only, testing RECALL")
    print("\n  G3 is the real test and is separable from G1/G2: distinct")
    print("  constituents with a written path can still fail to be SELECTED")
    print("  from one parent, which would move the problem to the readout.")


if __name__ == "__main__":
    main()
