"""Head-to-head comparison of candidate P600 readouts on the emergent parser.

WHY THIS EXISTS
---------------
The shipped P600 does not separate conditions. After the readiness gate was
corrected (`gates.role_pathways_trained`) so P600 computes at all, it returns
0.9868 / 0.9866 / 0.9905 for grammatical / category-violation / novel -- the
same number to three decimals, so its Cohen's d is noise.

No Assembly Calculus or NEMO paper specifies an ERP model; this is the repo's
own extension. So the standard to hold it to is not "does it match a paper" but
(a) is it AC-native, and (b) does it reproduce the neurobiological pattern. The
repo's own reference run (`research/results/primitives/RESULTS_composed_erp.md`,
n=10000 k=100, 10 seeds, pre-norm_init) did reproduce it:

    condition            N400            P600 (anchored instability)
    grammatical          0.088 +/-0.007  0.023 +/-0.011
    category violation   0.351 +/-0.012  4.954 +/-0.094
    novel object         0.979 +/-0.010  0.538 +/-0.115

That is the canonical double dissociation -- a novel word drives N400 to ceiling
while sparing P600 (semantic, not structural); a word-category violation drives
P600 (structural). It also states the design principle the current code drifted
away from: the two components are "different readouts of the same prediction +
binding mechanism. No separate modules or ad-hoc energy functions are needed."

THE 2x2 IS THE TEST, NOT A THRESHOLD
------------------------------------
A metric that merely separates category-violation from grammatical is not
enough: it must ALSO leave novel words near grammatical, or it is not a P600, it
is a generic anomaly detector. The shipped metric currently gives novel the
HIGHEST value of the three, which is backwards. So each candidate is scored on
both contrasts:

    d(catviol vs gram)  should be LARGE and POSITIVE
    d(novel   vs gram)  should be SMALL

CANDIDATES
----------
raw_drive      pre-k-WTA drive into the role area, as `input_drive` returns it.
               This is what the shipped metric consumes; measuring it directly
               separates "the signal is absent" from "the signal is present but
               the 1-energy transform destroys it".
energy_deficit max(0, 1 - raw_drive). THE SHIPPED METRIC (its `anchored` term).
self_energy    normalized self-recurrent pre-k-WTA energy of the role area --
               the shipped metric's other term, via `phrase_stability`.
self_energy_k  the same total renormalized by k instead of by `w`.
               `_self_recurrent_energy` divides by `w`, the count of neurons
               that have EVER fired, which grows without bound during training
               while only ~k neurons carry real drive -- a mechanical route to
               zero. If this candidate separates and `self_energy` does not,
               the divisor alone is the bug.
binding_weak   1 - max overlap(role assembly the word projects to, the stored
               role assemblies). Pure AC: assembly overlap against a FIXED
               stored reference, no ad-hoc energy. The reference run measured
               this ("binding weakness", d=30.23) and set it aside as measuring
               "retrieval, not integration" -- but it is also the readout
               `_score_role_binding` already uses to drive role assignment at
               0.930 accuracy, so it is known to work on this substrate under
               norm_init, which is exactly the property the churn-based metric
               lost.

RESULT (4 seeds x 3 items per condition = 12 per cell)
------------------------------------------------------
    metric            gram   catviol     novel   d(cv/g)  d(nov/g)
    raw_drive       0.0194    0.0263    0.0160     +0.95     -0.31   <- only 2x2
    energy_deficit  0.9786    0.9595    0.9840     -2.97     +0.54   <- SHIPPED
    self_energy     0.0054    0.0000    0.0051     -1.82     -0.08
    self_energy_k   0.1486    0.0000    0.1531     -1.83     +0.04
    binding_weak    0.3667       nan    0.0875       nan     -0.80
    n400 (control)  0.9516    0.9848    0.9607     +1.77     +0.41

**ROOT CAUSE: THE CONTRAST IS CONFOUNDED WITH TARGET AREA.** Recording which
pathway each condition actually probes settles it:

    grammatical         {'NOUN_CORE->ROLE_PATIENT': 12}
    category_violation  {'VERB_CORE->VP': 12}
    novel_noun          {'NOUN_CORE->ROLE_PATIENT': 4,
                         'NOUN_CORE->ROLE_AGENT': 4, 'ADJ_CORE->VP': 4}

Grammatical and category-violation are not measured on the same brain area.
`structural_role_area` sends a VERB to VP and a NOUN to ROLE_PATIENT, so the
conditions differ in target-area identity -- different sizes, different
in-degrees -- and `input_drive`'s own docstring says comparing across areas
"reverses the sign". No choice of normalization can rescue a contrast whose two
arms measure different areas, so the earlier readings ("saturates", then "sign
inverted") were both downstream symptoms.

**The intended mechanism never runs.** `anchored_p600_live` documents a violation
as "routing a wrongly-typed core through an untrained pathway" into the role
area. Because the probe follows the INTRUDING WORD'S OWN category, a verb is
routed to VP -- where verbs belong -- so no untrained core->ROLE pathway is ever
traversed. The code measures where the violating word WANTS to go, which is
exactly not the violation.

**The fix is a design change, not a rescale.** Measure drive into the role area
the syntactic frame EXPECTED (where the parser was trying to bind), not the area
implied by the intruding word's category. That makes the two arms area-matched,
which is the precondition for any of the candidate metrics to mean anything.
It also exposes an early signal the pipeline currently discards: the mismatch
between expected and actual core/role area is a word-CATEGORY error available
before role binding -- an ELAN analogue, which is what the ERP literature
reports as the early half of the biphasic response to category violations.

`novel_noun` additionally blends three different pathways (including ADJ_CORE
for "the small dog runs"), so that arm is a mixture of area identities too.

AREA-MATCHED RESULT (`drive_expected` / `share_expected`)
---------------------------------------------------------
Probing the role area POSITION expects, rather than the one the intruding word's
category implies, area-matches grammatical against category-violation -- both
now measure ROLE_PATIENT (confirmed in the pathway table the run prints):

    metric            gram   catviol     novel   d(cv/g)  d(nov/g)
    drive_expected  0.0207    3.6406    6.6994     +0.61     +0.95
    share_expected  0.0014    0.0761    0.2357     +0.60     +1.49

The raw difference is ~175x, so the effect is not subtle -- but it runs OPPOSITE
to the design intent: a verb drives the expected patient slot far harder than a
trained noun does. Cohen's d is only ~0.6 because the variance is enormous.

TWO THINGS NOT YET RESOLVED
* The matching is INCOMPLETE. 8 of 12 novel items expect ROLE_AGENT ("the bird
  sees the cat", "the small dog runs") while every grammatical item expects
  ROLE_PATIENT, so d(nov/g) is still a cross-area comparison and must not be
  read as a dissociation. Only the gram-vs-catviol column is clean.
* A candidate mechanism for the inversion, UNCONFIRMED: `norm_init` applies a
  read-time 1/d_j divisor per postsynaptic in-degree, so a heavily TRAINED
  pathway (high in-degree) is divided down hardest, while an untrained
  VERB_CORE->ROLE_PATIENT connectome is lazily materialized at probe time with
  fresh weights and low in-degree and therefore reads LARGER. That would make
  drive systematically anti-correlate with training. Test it by comparing
  in-degrees on the two pathways before choosing any metric.

KEYSTONE: THE MUTUAL-INHIBITION COMPETITION PICKS UNTRAINED AREAS
-----------------------------------------------------------------
Running the paper's competition (project the core into every role area of
MUTUAL_INHIBITION_GROUPS[0] in ONE call, so `_apply_mutual_inhibition` fires)
and asking whether the expected area wins:

    expected area wins:  grammatical 0.333   catviol 0.000   novel 0.000

Even for GRAMMATICAL items the right area wins only a third of the time, and the
winners are ROLE_LOCATION / ROLE_GOAL / ROLE_SOURCE / ROLE_THEME. Those areas
have NO learned connectome at all:

    role area      conn from NOUN_CORE   learned weight
    ROLE_AGENT     (4614, 960)                 127181
    ROLE_PATIENT   (4614, 960)                 110295
    ROLE_ACTION    (empty)                          0
    ROLE_THEME     (empty)                          0
    ROLE_GOAL      (empty)                          0
    ROLE_SOURCE    (empty)                          0
    ROLE_LOCATION  (empty)                          0

MECHANISM, measured -- and it CORRECTS the first reading above, which claimed
untrained areas are systematically stronger. They are not. The scores mutual
inhibition actually compares, projecting "dog" into each role area:

    role area                    total_activation      pre-kWTA total
    ROLE_AGENT                             1.1277                7.60
    ROLE_PATIENT                           1.0685                6.61
    ROLE_GOAL / SOURCE / ACTION            1.0467                1.05
    ROLE_THEME / LOCATION                  1.0333                1.03

The trained areas DO score highest -- by 7%, where the pre-kWTA signal separates
them by 7x. `total_activation` sums only the top-k winners, so an untrained area
whose entire materialized population IS k (w=30) has all of it selected and
keeps 100% of its small drive, while a trained area's ~960 neurons dilute what
the top-k captures. A 7x separation collapses to a near-tie that noise flips.

So the defect is that THE DECISION VARIABLE DESTROYS THE MARGIN, not that it
points the wrong way. Same family as the lesson already recorded for the ERP
components -- post-k-WTA quantities are unreliable under `norm_init` -- but a
different mechanism (dilution by population size, not churn).

NOT YET RESOLVED: swapping the competition onto pre-k-WTA totals
(`comp_correct_pre`) produced an implausible result -- the expected area
"wins" 0.92 of the time for category VIOLATIONS versus 0.33 for grammatical
items, i.e. better for a verb in an object slot than for a correct noun. That
is almost certainly a bug in the probe rather than a finding (the per-area
loop leaves `record_activation` set, and the accuracy has the opposite sign
convention to a P600 magnitude, which also makes the automatic verdict column
wrong for it). Do not read `comp_correct_pre` as evidence until the probe is
rebuilt.

This is not an ERP problem. It is the paper's only inter-area inhibition
selecting the wrong winner, and it explains why `parser_mixins/core.py` records
that this mechanism "was inert, and role exclusivity was enforced by a Python
set instead" -- the symbolic workaround was compensating for a broken
competition. It also explains the P600 inversion: every drive-based metric here
inherits the same anti-correlation.

WHY INHIBITION IS THE RIGHT FRAME (and what the reference does)
--------------------------------------------------------------
The reference parser selects the role area by DISINHIBITION, not by looking up
the word's category -- `.reference/dmitropolsky-assemblies/parser.py`:

    "dogs": open fibers LEX<->SUBJ and LEX<->OBJ but only SUBJ disinhibited

Fibers open to BOTH candidate areas and inhibition picks the winner, so the
candidate set is identical in every condition and the contrast is area-matched
by construction. This repo keeps the same spec -- the seven role areas are
MUTUAL_INHIBITION_GROUPS[0], per the paper's "the three ROLE areas are in mutual
inhibition; this is the only use of interarea inhibition in our model". But
`anchored_p600_live` calls `input_drive(..., target_areas=[role_area])` with ONE
target, and mutual inhibition only fires when >=2 group areas are targets of the
same project() call. So the competition the paper specifies never runs, and the
categorical lookup that replaced it is what introduced the area confound.

OTHER FINDINGS
* `binding_weak` is nan for category violations, and NOT for the reason first
  recorded here. The lookup is `role_lexicons[VP]`, which is empty because VP is
  a phrase area rather than a thematic role, so there are no stored assemblies
  to compare against. It is a consequence of the area confound above, not
  evidence about untrained pathways.
* `self_energy_k` rescales grammatical 0.005 -> 0.149 but leaves category
  violation at 0.0000, so the `w` divisor is NOT the whole story -- that
  condition has genuinely zero self-recurrent energy.
* The N400 control does not reproduce the reference either: novel should
  DOMINATE (0.979 vs 0.088 grammatical) but here it is 0.9607 vs 0.9516, with
  every condition crushed into 0.95-0.98. Compression is systemic to this
  substrate, not specific to P600.

CAVEATS -- do not act on the magnitudes
* 3 items per condition per seed, 12 per cell. d=+0.95 is modest and its
  interval at that n is wide. The SIGN consistency across metrics is the
  trustworthy part.
* An alignment bug produced an entirely different (and wrong) table on the first
  two runs: `calibrate_erp_thresholds` probes every word (30 calls) but keeps
  one sample per frame at the critical position (9), so zipping rows to samples
  positionally mislabelled every row. Probes are now matched to `sample.word`,
  and the run prints "N/9 samples matched" -- if that is not 9/9, the table is
  meaningless. It also took three attempts to patch the right binding: several
  modules do `from .adapters import measure_live_integration`, so each holds its
  own reference and the live call comes from `erp.runner`.

Run: .venv/Scripts/python.exe research/experiments/p600_metric_comparison.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

CANDIDATES = (
    "raw_drive", "energy_deficit", "self_energy", "self_energy_k",
    "binding_weak", "drive_expected", "share_expected", "comp_correct",
    "comp_correct_pre", "n400",
)
PATHWAYS: Dict[str, List[str]] = {}
LABELS = ("grammatical", "category_violation", "novel_noun")


def _cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 1e-12 else 0.0


def collect(seed: int) -> Dict[str, Dict[str, List[float]]]:
    """Measure every candidate at every frame's critical position, by label."""
    from neural_assemblies.assembly_calculus.assembly import overlap as ov
    from neural_assemblies.assembly_calculus.binding import input_drive
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        MUTUAL_INHIBITION_GROUPS, ROLE_AGENT, ROLE_PATIENT,
    )

    from neural_assemblies.assembly_calculus.ops import _snap
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        get_parser_cache,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation import (
        calibrate_erp_thresholds,
    )
    import neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters as A

    parser = get_parser_cache().fork("SENTENCES", seed=seed)
    brain = parser.brain
    # The role-area competition group, straight from the paper spec: "the three
    # ROLE areas are in mutual inhibition; this is the only use of interarea
    # inhibition in our model".
    _ROLE_GROUP = [a for a in MUTUAL_INHIBITION_GROUPS[0] if a in brain.areas]
    rows: List[dict] = []
    stab_buf: List[tuple] = []

    orig_anchor = A.anchored_p600_live
    orig_stab = A.phrase_stability
    orig_measure = A.measure_live_integration

    def stab2(b, area, **kw):
        out = orig_stab(b, area, **kw)
        w = max(int(b.areas[area].w), 1) if area in b.areas else 1
        k = max(int(getattr(b.areas[area], "k", 0) or 0), 1) if area in b.areas else 1
        stab_buf.append((float(out), w, k))
        return out

    probe_word = {"w": None}

    def measure2(p, word, category, **kw):
        """Wrap the probe entry point purely to learn WHICH WORD is being probed.

        Alignment is the whole ballgame here. `calibrate_erp_thresholds` probes
        EVERY word of every frame (30 calls) but keeps one sample per frame, at
        the critical position (9 samples). Zipping rows to samples positionally
        therefore pairs sample 0 with the first word of the sentence -- silently
        mislabelling every candidate. Recording the word lets rows be matched to
        `sample.word` in order instead.
        """
        probe_word["w"] = word
        probe_word["verb_seen"] = bool(kw.get("verb_seen"))
        return orig_measure(p, word, category, **kw)

    def anchor2(p, core_area, role_area, **kw):
        """Fires once per probe, AFTER the phrase_stability loop -> flush here."""
        sources = [core_area]
        sc = kw.get("subject_core")
        if (sc and sc != core_area and sc in brain.areas
                and brain.areas[sc].winners is not None
                and len(brain.areas[sc].winners) > 0):
            sources.append(sc)
        try:
            raw = float(input_drive(
                brain, sources=sources, target_areas=[role_area],
            ).get(role_area, 0.0))
        except Exception:
            raw = float("nan")

        weak = float("nan")
        try:
            lex = (getattr(p, "role_lexicons", {}) or {}).get(role_area, {})
            if lex and core_area in brain.areas:
                with brain.frozen():
                    brain.project({}, {core_area: [role_area]})
                    asm = _snap(brain, role_area)
                weak = 1.0 - float(max(ov(asm, s) for s in lex.values()))
        except Exception:
            pass

        # ---- AREA-MATCHED probe -------------------------------------------
        # The shipped probe picks its target from the INTRUDING WORD'S category
        # (structural_role_area sends a VERB to VP), so grammatical and
        # violation land on different areas and the contrast measures area
        # identity. Here the target comes from POSITION alone -- object slot
        # after the verb, subject slot before it -- so every condition is
        # measured on the same area, which is what the reference parser gets
        # for free by opening fibers to BOTH candidate role areas and letting
        # DISINHIBITION choose ("open fibers LEX<->SUBJ and LEX<->OBJ but only
        # SUBJ disinhibited"). The role areas are a MUTUAL_INHIBITION_GROUP
        # here too, but inhibition only fires when >=2 group areas are targets
        # of the same project() call, and the shipped probe passes exactly one
        # -- so the competition the paper specifies never runs.
        expected_role = ROLE_PATIENT if probe_word.get("verb_seen") else ROLE_AGENT
        drive_exp = share_exp = float("nan")
        try:
            all_drives = input_drive(
                brain, sources=sources, target_areas=list(_ROLE_GROUP),
            )
            drive_exp = float(all_drives.get(expected_role, float("nan")))
            tot = float(sum(v for v in all_drives.values() if np.isfinite(v)))
            # Share is scale-free ACROSS areas, so it cancels the per-area
            # normalization differences that made the raw contrast unusable.
            share_exp = drive_exp / tot if tot > 1e-12 else float("nan")
        except Exception:
            pass

        # ---- THE PAPER'S ACTUAL MECHANISM: inhibitory competition ----------
        # Project the word's core into EVERY role area of the mutual-inhibition
        # group in ONE call, which is the only way `_apply_mutual_inhibition`
        # fires (it skips groups with <=1 active member). The competition is
        # winner-take-all on total_activation, so the question a P600 should ask
        # is simply: does the area the syntax EXPECTED win?
        #
        # This is area-matched BY CONSTRUCTION -- the candidate set is identical
        # in every condition -- which is what the reference gets from opening
        # fibers to both candidates and choosing by disinhibition.
        #
        # MI is DESTRUCTIVE (it sets winners=[] and w=0 on every loser), so the
        # group state is snapshotted and restored; without this the probe would
        # wipe trained role areas.
        comp_correct = float("nan")
        comp_winner = None
        # Same competition, decided on PRE-kWTA total instead of the post-kWTA
        # winner sum. total_activation sums only the top-k, so an untrained area
        # whose materialized population IS k captures all of its drive while a
        # trained area's ~960 neurons dilute what the top-k captures -- a 7x
        # pre-kWTA separation collapses to a ~7% margin that noise flips.
        comp_correct_pre = float("nan")
        try:
            pre_tot = {}
            for a in _ROLE_GROUP:
                sv = (np.array(brain.areas[a].winners, copy=True), int(brain.areas[a].w))
                with brain.frozen():
                    brain.record_activation = True
                    brain.project({}, {core_area: [a]})
                    pre_tot[a] = float(
                        (getattr(brain, "last_pre_kwta_totals", {}) or {}).get(a, 0.0))
                brain.areas[a].winners, brain.areas[a].w = sv
            if pre_tot:
                comp_correct_pre = 1.0 if max(pre_tot, key=pre_tot.get) == expected_role else 0.0
        except Exception:
            pass
        snap = {}
        try:
            for a in _ROLE_GROUP:
                ar = brain.areas[a]
                snap[a] = (np.array(ar.winners, copy=True), int(ar.w))
            with brain.frozen():
                brain.project({}, {core_area: list(_ROLE_GROUP)})
            survivors = [a for a in _ROLE_GROUP
                         if len(brain.areas[a].winners) > 0]
            comp_winner = survivors[0] if len(survivors) == 1 else None
            if comp_winner is not None:
                comp_correct = 1.0 if comp_winner == expected_role else 0.0
        except Exception:
            pass
        finally:
            for a, (w_arr, w_n) in snap.items():
                ar = brain.areas[a]
                ar.winners = w_arr
                ar.w = w_n
                try:
                    brain._engine.set_winners(a, w_arr)
                    est = getattr(brain._engine, "_areas", {}).get(a)
                    if est is not None:
                        est.w = w_n
                except Exception:
                    pass

        out = orig_anchor(p, core_area, role_area, **kw)

        if stab_buf:
            se = float(np.mean([s for s, _w, _k in stab_buf]))
            se_k = float(np.mean([s * w / k for s, w, k in stab_buf]))
        else:
            se = se_k = float("nan")
        stab_buf.clear()
        rows.append({
            "raw_drive": raw, "energy_deficit": float(out),
            "self_energy": se, "self_energy_k": se_k, "binding_weak": weak,
            "word": probe_word["w"],
            "core_area": core_area, "role_area": role_area,
            "drive_expected": drive_exp, "share_expected": share_exp,
            "comp_correct": comp_correct, "comp_winner": comp_winner,
            "comp_correct_pre": comp_correct_pre,
            "expected_role": expected_role,
        })
        return out

    A.anchored_p600_live, A.phrase_stability = anchor2, stab2
    # Several modules did `from .adapters import measure_live_integration`, so
    # each holds its OWN reference and patching the adapters attribute alone
    # never reaches the call site (0/9 matched, twice). The live call comes from
    # erp.runner, reached via erp.frames. Rebind every holder rather than
    # guessing which one is live.
    import importlib
    _HOLDERS = [
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters",
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames",
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.runner",
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp",
        "neural_assemblies.assembly_calculus.emergent.evaluation",
    ]
    _patched = []
    for _name in _HOLDERS:
        try:
            _m = importlib.import_module(_name)
        except Exception:
            continue
        if getattr(_m, "measure_live_integration", None) is not None:
            setattr(_m, "measure_live_integration", measure2)
            _patched.append(_m)
    try:
        report = calibrate_erp_thresholds(parser)
    finally:
        A.anchored_p600_live, A.phrase_stability = orig_anchor, orig_stab
        for _m in _patched:
            _m.measure_live_integration = orig_measure

    out: Dict[str, Dict[str, List[float]]] = {
        c: {lbl: [] for lbl in LABELS} for c in CANDIDATES
    }
    # Match each sample to the probe row for ITS critical word, scanning rows
    # forward so repeated words across frames consume distinct rows.
    i, matched, missed = 0, 0, 0
    for smp in report.samples:
        if smp.label not in out["raw_drive"]:
            continue
        j = i
        while j < len(rows) and rows[j].get("word") != smp.word:
            j += 1
        if j >= len(rows):
            missed += 1
            continue
        r, i = rows[j], j + 1
        matched += 1
        for c in CANDIDATES:
            out[c][smp.label].append(
                float(smp.n400) if c == "n400" else r[c]
            )
        PATHWAYS.setdefault(smp.label, []).append(
            f"exp={r['expected_role']} won={r['comp_winner']}"
        )
    print(f"  seed {seed}: {matched}/{len(report.samples)} samples matched "
          f"to probes ({len(rows)} probes total)"
          + (f", {missed} UNMATCHED" if missed else ""), flush=True)
    return out


def run(seeds: Sequence[int] = (42, 43, 44, 45)) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("P600 candidate metrics -- 2x2 dissociation on the emergent parser")
    print(f"seeds: {list(seeds)}\n")

    pooled: Dict[str, Dict[str, List[float]]] = {
        c: {lbl: [] for lbl in LABELS} for c in CANDIDATES
    }
    for sd in seeds:
        got = collect(sd)
        for c in CANDIDATES:
            for lbl in LABELS:
                pooled[c][lbl].extend(got[c][lbl])
        print(f"  seed {sd} done", flush=True)

    print(f"\n{'metric':<16}{'gram':>10}{'catviol':>10}{'novel':>10}"
          f"{'d(cv/g)':>10}{'d(nov/g)':>10}   verdict")
    print("-" * 82)
    for c in CANDIDATES:
        g, v, n = (pooled[c][l] for l in LABELS)
        d_cv, d_nv = _cohens_d(v, g), _cohens_d(n, g)
        # A P600 must move for the STRUCTURAL violation and stay put for the
        # merely-unfamiliar word. N400 is included as the positive control: it
        # is expected to do the opposite (novel >> gram).
        if c == "n400":
            verdict = "(control)"
        elif not np.isfinite(d_cv):
            verdict = "no data"
        elif d_cv > 0.8 and abs(d_nv) < abs(d_cv) / 2:
            verdict = "DISSOCIATES"
        elif d_cv > 0.8:
            verdict = "separates, but novel moves too"
        else:
            verdict = "no separation"
        print(f"{c:<16}{np.nanmean(g):>10.4f}{np.nanmean(v):>10.4f}"
              f"{np.nanmean(n):>10.4f}{d_cv:>10.2f}{d_nv:>10.2f}   {verdict}")
    # Which pathway does each condition actually probe? `structural_role_area`
    # sends a VERB to VP and a NOUN to ROLE_PATIENT, so the conditions may not
    # be measured on the same target area at all. `input_drive` normalizes PER
    # CANDIDATE and its own docstring warns that comparing raw sums across areas
    # of different size "reverses the sign" -- so if these differ by condition,
    # the whole contrast is confounded by area identity rather than by training.
    from collections import Counter
    print("\n  PATHWAY ACTUALLY PROBED PER CONDITION")
    for lbl in LABELS:
        print(f"    {lbl:<20}{dict(Counter(PATHWAYS.get(lbl, [])))}")

    print("\nd(cv/g) should be LARGE POSITIVE; d(nov/g) should be SMALL.")
    print("n400 is the positive control and should show the OPPOSITE pattern.")


if __name__ == "__main__":
    run()
