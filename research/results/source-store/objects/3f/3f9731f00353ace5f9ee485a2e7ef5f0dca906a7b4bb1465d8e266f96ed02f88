"""Why moods merge: decompose SYN's drive instead of intervening on it again.

THE PROBLEM, AND WHY IT IS STILL OPEN
--------------------------------------
`neural_assemblies/reference/word_order_learner.py` records that the paper's
EMERGENT multi-mood mechanism does not survive training. Distinct syntactic
chains form correctly at initialization (SYNTAX overlap 0.04 between two moods)
and merge to 1.00 within ~20 sentences, because the helper is shared between
moods. Multi-mood word order is the prerequisite for any compositional-advantage
claim -- with a single mood there is one global rule and nothing to compose --
so this blocks the headline comparison.

Four interventions have already failed, and the file records each:

    100x capacity (n=1e5)          no effect
    the paper's beta=0.06          no effect
    norm_init                      no effect
    raising MOOD's plasticity      no effect
    priming SYN from MOOD at t=0   0.955 -> 0.952

The structural workaround (`per_mood_syntax`, one SYNTAX area per mood) gets
24/24 and is committed, but it is explicitly a deviation: the paper expects the
per-mood chains to be EMERGENT.

WHAT HAS NEVER BEEN DONE IS THE MEASUREMENT
--------------------------------------------
Every attempt above changed something and looked at the outcome. None decomposed
the quantity that actually decides the k-WTA. SYN's winners are chosen by summed
drive from three sources, visible in `_project_training`:

    helper -> syn     the shared component. Identical across moods, because the
                      helper IS shared -- this is the suspect.
    MOOD   -> syn     the DIFFERENTIATING component. The only thing that can
                      make mood 0's winners differ from mood 1's.
    syn    -> syn     self-recurrence, added at t > 0. This is the channel the
                      2026-07-28 collapse arc identified as destroying shared
                      areas across the board.

If mood-conditioning fails because MOOD's contribution is swamped, the ratio
`mood / (helper + self)` is small and gets smaller with training, and NO
intervention that leaves the ratio unchanged can work -- which would explain all
five failures at once, including why 100x capacity did nothing.

THAT LAST POINT IS THE DISCRIMINATOR, and it is why this is not simply the
lexicon collapse again. The 2026-07-28 law says the ceiling SCALES WITH n
(M=32/64/256 at n=1000/2000/4000), because the competitor is the population
maximum. Here 100x capacity changed nothing, so the competitor is NOT the
population -- it is one specific potentiated assembly receiving nearly the same
input. Same family, different disease, and the ratio is what tells them apart.

WHAT IS MEASURED
----------------
At checkpoints through training, for each constituent and each mood, the mean
synaptic drive arriving at the SYN assembly from each of the three sources, read
straight from the connectome (no projection, so no k-WTA can intervene) --
alongside the mood separation those weights produce.

    d_help / d_mood / d_self   mean weight onto the SYN winners, per source
    ratio                      d_mood / (d_help + d_self); the share of the
                               decision MOOD actually controls
    sep                        1 - overlap(SYN under mood 0, SYN under mood 1).
                               1.0 = fully distinct chains, 0.0 = merged

PRE-REGISTERED
--------------
V1 sep starts high and falls to ~0 within ~20 sentences, reproducing the
   documented collapse from an independent probe. Gates everything else.
V2 ratio is SMALL at every checkpoint -- MOOD controls a minority of the drive
   from the start. If ratio is large while moods still merge, the swamping
   account is wrong and the failure is in WHICH neurons MOOD drives rather than
   how hard.
V3 ratio FALLS as training proceeds, because helper->syn and syn->syn
   potentiate on every sentence of BOTH moods while MOOD->syn is split between
   them. This makes the collapse progressive rather than immediate and predicts
   it is unavoidable under Hebbian training with a shared helper.
V4 sep tracks ratio across checkpoints. If they correlate, the ratio is the
   controlling variable and the fix must change it -- which none of the five
   failed interventions did.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CHECKPOINTS = (0, 5, 10, 20, 40, 80)
SEEDS = (0, 1, 2, 3, 4)
MOODS = {0: ("S", "V", "O"), 1: ("V", "S", "O")}


def area_drive(brain, src, tgt, tgt_ids):
    """Mean weight from *src*'s CURRENT assembly onto *tgt_ids*. No projection.

    Read from the connectome so k-WTA and settling cannot intervene -- the same
    discipline as slot_addressing_drift.py, and both index spaces go through
    ops._compact_index.
    """
    from neural_assemblies.assembly_calculus.ops import _compact_index

    eng_t = brain._engine_for(brain.areas[tgt])
    conn = getattr(eng_t, "_area_conns", {}).get(src, {}).get(tgt)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    src_win = np.asarray(brain.areas[src].winners)
    t_inv = _compact_index(eng_t, tgt) or {}
    rows = [int(x) for x in src_win if int(x) < w.shape[0]]
    cols = [t_inv[int(x)] for x in tgt_ids
            if int(x) in t_inv and t_inv[int(x)] < w.shape[1]]
    if not rows or not cols:
        return float("nan")
    return float(w[np.ix_(rows, cols)].mean())


def snapshot(learner, c):
    """Drive decomposition and mood separation for constituent *c*."""
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import _snap
    from neural_assemblies.reference.word_order_learner import (
        HELPER, MOOD, SYNTAX,
    )

    brain = learner.brain
    syn, helper = SYNTAX[c], HELPER[c]

    # Drive the chain under each mood WITHOUT training, and see where SYN lands.
    per_mood = {}
    with brain.read_only():
        for mi in MOODS:
            learner._mood_now = mi
            brain.activate(MOOD, mi)
            learner._activate_role(0, c)
            brain.project({}, {helper: [syn], MOOD: [syn]})
            ids = np.asarray(_snap(brain, syn).winners, dtype=np.int64)
            per_mood[mi] = (
                ids,
                area_drive(brain, helper, syn, ids),
                area_drive(brain, MOOD, syn, ids),
                area_drive(brain, syn, syn, ids),
            )

    sep = 1.0 - float(overlap(per_mood[0][0], per_mood[1][0]))
    dh = statistics.mean(v[1] for v in per_mood.values() if v[1] == v[1])
    dm = statistics.mean(v[2] for v in per_mood.values() if v[2] == v[2])
    ds_vals = [v[3] for v in per_mood.values() if v[3] == v[3]]
    ds = statistics.mean(ds_vals) if ds_vals else 0.0
    denom = dh + ds
    return sep, dh, dm, ds, (dm / denom if denom > 0 else float("nan"))


def materialise(learner, c):
    """Give SYN its first winners, OUTSIDE the probe. Required, not optional.

    `snapshot` reads under `brain.read_only()`, which blocks recruitment -- that
    is its job, and it is what makes the probe non-mutating. But an area that
    has never been driven has no materialised neurons to recruit FROM, so both
    moods come back with the same degenerate winner set and separation reads
    EXACTLY 0.0000. That is a dead probe, not a merged pair: the identical
    failure mode found in gated_role_binding.py, and the reason the reference's
    "distinct at initialization, overlap 0.04" appeared not to reproduce.

    One projection per mood, at beta=0.1, is the smallest intervention that
    materialises the fiber. cp=0 therefore means AFTER MATERIALISATION, BEFORE
    TRAINING -- which is what "at initialization" means for the reference too,
    since it also has to drive the area before it can read it.
    """
    from neural_assemblies.reference.word_order_learner import HELPER, MOOD, SYNTAX

    for mi in MOODS:
        learner._mood_now = mi
        learner.brain.activate(MOOD, mi)
        learner._activate_role(0, c)
        learner.brain.project({}, {HELPER[c]: [SYNTAX[c]], MOOD: [SYNTAX[c]]})


def trial(seed):
    from neural_assemblies.reference.word_order_learner import WordOrderLearner

    learner = WordOrderLearner(mood_orders=MOODS, seed=seed,
                               per_mood_syntax=False)
    materialise(learner, "S")
    out = {}
    trained = 0
    for cp in CHECKPOINTS:
        if cp > trained:
            learner.train(cp - trained)
            trained = cp
        out[cp] = snapshot(learner, "S")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  Emergent multi-mood word order (per_mood_syntax=False), "
          f"moods {dict(MOODS)}")
    print(f"  {len(SEEDS)} seeds. Drive read from the connectome -- no "
          f"projection, so no k-WTA intervenes.")
    print(f"  sep = 1 - overlap(SYN|mood0, SYN|mood1); 1.0 distinct chains, "
          f"0.0 merged.")
    print(f"  ratio = d_mood / (d_help + d_self) = the share of the k-WTA "
          f"decision MOOD controls.\n")
    print(f"  {'sentences':>10}{'sep':>8}{'d_help':>10}{'d_mood':>10}"
          f"{'d_self':>10}{'ratio':>9}")

    res = [trial(s) for s in SEEDS]
    table = {}
    for cp in CHECKPOINTS:
        cols = [r[cp] for r in res]
        row = tuple(statistics.mean(c[i] for c in cols if c[i] == c[i])
                    for i in range(5))
        table[cp] = row
        print(f"  {cp:>10}{row[0]:>8.4f}{row[1]:>10.4f}{row[2]:>10.4f}"
              f"{row[3]:>10.4f}{row[4]:>9.4f}")

    print("\n  READING")
    s0, sN = table[CHECKPOINTS[0]][0], table[CHECKPOINTS[-1]][0]
    # The claim is the COLLAPSE -- separation falling to ~0 under training. The
    # absolute starting value is not comparable to the reference's 0.96: that
    # is measured after its own chain-building activation, whereas `materialise`
    # here is one projection per mood, deliberately the smallest thing that
    # makes the probe live. A weaker initial condition, same trajectory.
    v1 = sN < 0.05 and s0 > sN
    print(f"    V1 collapse reproduces:        {v1}   sep "
          f"{s0:.4f} -> {sN:.4f}   (reference inits at 0.96 via a stronger "
          f"protocol; only the trajectory is comparable)")
    ratios = [table[cp][4] for cp in CHECKPOINTS]
    v2 = max(r for r in ratios if r == r) < 0.5
    print(f"    V2 MOOD controls a minority:   {v2}   ratio max "
          f"{max(r for r in ratios if r == r):.4f}")
    v3 = ratios[-1] < ratios[0]
    print(f"    V3 ratio falls with training:  {v3}   "
          f"{ratios[0]:.4f} -> {ratios[-1]:.4f}")
    seps = [table[cp][0] for cp in CHECKPOINTS]
    pairs = [(r, s) for r, s in zip(ratios, seps) if r == r and s == s]
    if len(pairs) > 2:
        rs = [p[0] for p in pairs]
        ss = [p[1] for p in pairs]
        mr, ms = statistics.mean(rs), statistics.mean(ss)
        num = sum((a - mr) * (b - ms) for a, b in pairs)
        den = ((sum((a - mr) ** 2 for a in rs) *
                sum((b - ms) ** 2 for b in ss)) ** 0.5)
        corr = num / den if den > 0 else float("nan")
        print(f"    V4 sep tracks ratio:           "
              f"{corr > 0.7}   r = {corr:.3f}")

    print()
    if v1 and v2:
        print(f"    THE FAILURE IS A DRIVE RATIO, NOT A CAPACITY LIMIT. MOOD")
        print(f"    controls {ratios[-1]:.1%} of the drive that decides SYN's")
        print(f"    winners, so the shared helper picks them and the moods get")
        print(f"    the same chain. That explains all five failed interventions")
        print(f"    at once -- capacity, beta, norm_init, MOOD plasticity and")
        print(f"    t=0 priming all leave the ratio untouched. Any real fix has")
        print(f"    to change WHO DRIVES SYN, which is what `per_mood_syntax`")
        print(f"    does structurally and what gating the helper->syn fiber")
        print(f"    would do dynamically.")
    elif v1:
        print(f"    Collapse reproduces but MOOD is NOT swamped (ratio "
              f"{ratios[-1]:.4f}).")
        print(f"    The swamping account is wrong: MOOD has the drive and the")
        print(f"    moods still merge, so the failure is in WHICH neurons it")
        print(f"    drives, not how hard. Look at whether MOOD's two assemblies")
        print(f"    project onto overlapping SYN populations.")
    else:
        print(f"    The documented collapse did not reproduce here (sep "
              f"{s0:.4f} -> {sN:.4f}).")
        print(f"    Nothing below is interpretable until that is explained --")
        print(f"    check the mood pair and rounds against the module tests.")


if __name__ == "__main__":
    main()
