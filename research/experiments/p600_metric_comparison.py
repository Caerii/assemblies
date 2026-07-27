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
    "binding_weak", "n400",
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
            f"{r['core_area']}->{r['role_area']}"
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
