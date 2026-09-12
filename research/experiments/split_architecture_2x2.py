"""E15: the 2x2 re-run on per-value areas -- does removing merging remove the collapse?

PRE-REGISTERED (task #144), written after the mechanism (db898fb) and
before any split-architecture cell trained at scale.

THE MODEL UNDER TEST (E14's note): the corpus axis obeys a four-level law
-- reliability = per-item mass = exposure = budget x allocation -- but the
one-area feature design destroys label separability as total label
projections grow (shared SG-PL cols 2.6 -> 17.8/30 at 200 frames), which
is what collapsed uniform-200 to 0.490 and ate zipf-200's earned
exposure. The split (one area per value, mutual inhibition deciding --
the paper's ROLE-triple device) makes image merging STRUCTURALLY
impossible. If the model is right, the collapse goes with it.

CELLS: {UNIFORM, ZIPF} x FRAMES {50, 200} x seeds 42-51 (40 cells),
SLOW-scaled R1 (scaling scoped to the VALUE areas), n=3000,
split_feature_areas=True in every cell. The readout is the MI
competition (primary; the paper's semantics), with the per-area overlap
readout riding as a diagnostic. NOTE: because the READOUT differs from
E14's, cross-architecture comparisons (bars D2/E) are between different
readouts on the same corpora -- stated, not hidden; the within-run bars
(A-D1) are readout-internal.

REGISTERED BARS:
  A  STRUCTURAL: the two label images occupy DISJOINT areas in every
     cell (verified by construction check, not assumed), and the
     FUNCTIONAL discriminability guard -- mean MI margin at 200 frames
     within 0.5x-2x of its 50-frame value (the shared-area design's
     margin collapsed with merging; the split's must not).
  B  zipf-200 attested balanced >= 0.75 -- the E-series bar (levels 1-3
     of the law say the exposure is already there).
  C  MECHANISM TRANSFER: rho(mass delta, correct) over zipf-200 attested
     items positive, CI excluding zero, where delta = (mass into the PL
     area's image) - (mass into the SG area's image) -- the E12 readout
     across two weight matrices now.
  D1 THE STRONGEST PREDICTION: uniform-200 >= uniform-50 (paired, CI not
     excluding a >= 0 slope -- i.e. the negative slope must be GONE; its
     E14 value was -0.160 +/- 0.05).
  D2 cross-architecture: uniform-200-split beats E14's uniform-200
     (0.490 +/- 0.052) -- different readout, same corpus; reported with
     that caveat attached.
  GUARDS: roles 4/4; tense balanced reported per cell; secondary-readout
     agreement rate (MI answer == overlap answer) reported -- systematic
     disagreement is a finding (#24 measured cross-area drive as the
     weaker primitive in the ROLE setting).

DECISION RULE: D1 + A pass -> the merging model is CONFIRMED by
intervention; with B, the number arc CLOSES (statistics fixed at the
corpus level, architecture fixed at the area level) and the default-flip
+ CHILDES discussions open. D1 passes but B misses -> merging was real
but not the whole 200-frame story; the residual gap goes to the next
registration with the margin/agreement diagnostics as the suspect list.
D1 fails -> the merging model is REFUTED by its own intervention -- the
collapse was never (only) merging; report honestly and re-open the
suspect list with the split's diagnostics in hand.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_checkpoint_document

SEEDS = list(range(42, 52))
ARMS = ("UNIFORM", "ZIPF")
BUDGETS = (50, 200)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("SP_N", "3000"))
if os.environ.get("SP_SMOKE") == "1":
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "split_architecture_2x2_results.json")

E14_UNIFORM_200 = (0.490, 0.052)  # cross-architecture reference (D2)

from overlap_ceiling import spearman  # noqa: E402
from zipf_synthesis import FIXED_PL, FIXED_SG, ROLE_PROBES  # noqa: E402


def run_cell(arm: str, frames: int, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER, FEATURE_VALUE_LABELS, feature_value_area,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        generation as gen,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation \
        .morph_features import (
            attested_morph_sets, score_recall,
            feature_images_compact, item_afferent_mass,
        )
    from _parallel import forked_parser

    def build_and_train_pre():
        random.seed(seed)
        np.random.seed(seed)
        p = EmergentParser(n=N, k=30, seed=seed,
                           vocabulary=build_vocabulary_preset("core"),
                           fast_training=True)
        ct = CurriculumTrainer(p)
        for stage in PRE_STAGES:
            ct.train_stage(stage)
        return p

    def arm_setup(p):
        p.split_feature_areas = True
        # Scaling scoped to the VALUE areas (the shared areas never train
        # under the split, so scoping to them would be a silent no-op --
        # the dormant-selector lesson).
        value_areas = frozenset(
            feature_value_area(f, lab)
            for f in (TENSE, NUMBER)
            for lab in FEATURE_VALUE_LABELS[f])
        p.brain._engine.synaptic_scaling = value_areas
        p.brain._synaptic_scaling = value_areas
        p.brain._engine.synaptic_scaling_deferred = True
        p.morph_repetitions = 1
        gen.FRAMES_PER_STAGE = frames
        if arm == "ZIPF":
            gen.SUBJECT_SAMPLING = "zipf"

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"zs-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)
    brain = parser.brain

    sets_ = attested_morph_sets(parser)
    number = score_recall(parser.recall_number,
                          {k: sets_[k] for k in ("SG", "PL")})
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})
    fixed = score_recall(parser.recall_number,
                         {"SG": FIXED_SG, "PL": FIXED_PL})

    # Secondary-readout scoring over the SG side too, so an overlap-readout
    # "balanced" is computable (added with the per-item extension above).
    sg_overlap_ok = sg_total = 0
    for lemma in (sets_["SG"] if sets_["SG"] else FIXED_SG):
        _g, d = parser.recall_number(lemma)
        sg_total += 1
        sg_overlap_ok += d.get("overlap_answer") == "SG"

    # Bar A, structural half: the two label images by AREA. Verified,
    # not assumed: both areas exist, are distinct, and each image is
    # nonempty in its own area.
    area_sg = feature_value_area(NUMBER, "SG")
    area_pl = feature_value_area(NUMBER, "PL")
    img_sg = feature_images_compact(parser, area_sg, {"SG": "number_SG"})
    img_pl = feature_images_compact(parser, area_pl, {"PL": "number_PL"})
    structural_ok = bool(
        area_sg != area_pl
        and area_sg in brain.areas and area_pl in brain.areas
        and len(img_sg.get("SG", [])) > 0 and len(img_pl.get("PL", [])) > 0)

    exposure = getattr(parser, "_morph_exposure", {})
    items = []
    margins = []
    agree = agree_total = 0
    for form in sorted(set(FIXED_PL) | set(sets_["PL"])):
        m_pl = item_afferent_mass(parser, form, area_pl, img_pl)
        m_sg = item_afferent_mass(parser, form, area_sg, img_sg)
        got, diag = parser.recall_number(form)
        if diag.get("margin") is not None:
            margins.append(diag["margin"])
        ov_ans = diag.get("overlap_answer")
        if got is not None and ov_ans is not None:
            agree_total += 1
            agree += got == ov_ans
        items.append({
            "form": form,
            "attested": form in sets_["PL"],
            "exposure": exposure.get(f"NUMBER:{form}", 0),
            "delta": ((m_pl["PL"] - m_sg["SG"])
                      if (m_pl and m_sg) else None),
            "correct": bool(got == "PL"),
            "answer": got,
            # Added after the first full run (bars unchanged): the MI and
            # overlap readouts disagreed on ~38% of answered items and only
            # agreement COUNTS were stored, so the "which readout is the
            # bottleneck" question was unanswerable from the JSON.
            "overlap_answer": ov_ans,
            "overlap_correct": bool(ov_ans == "PL"),
            "margin": diag.get("margin"),
        })
    total_exp = sum(it["exposure"] for it in items if it["attested"])

    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want

    print(f"[E15 {arm}-{frames} seed={seed}] "
          f"num bal={number['_balanced']:.3f} (PL n={number['PL']['n']}, "
          f"exp={total_exp}) | tense={tense['_balanced']:.3f} | "
          f"margin={np.mean(margins) if margins else float('nan'):.3f} | "
          f"agree={agree}/{agree_total} | roles {g_ok}/{g_total}",
          flush=True)
    return {"number": number, "tense": tense, "fixed": fixed,
            "structural_ok": structural_ok, "items": items,
            "mean_mi_margin": float(np.mean(margins)) if margins else None,
            "readout_agree": agree, "readout_agree_total": agree_total,
            "sg_overlap_ok": sg_overlap_ok, "sg_total": sg_total,
            "total_attested_exposure": total_exp,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def cell_key(arm: str, frames: int) -> str:
    return f"{arm}-{frames}"


def main():
    if os.environ.get("SP_ANALYZE") == "1":
        with open(OUT_PATH) as f:
            raw = json.load(f)
        results = {c: {int(s): v for s, v in by.items()}
                   for c, by in raw.items()}
    else:
        from _parallel import run_cells

        cells = [(a, b, s) for a in ARMS for b in BUDGETS for s in SEEDS]
        cell_results = run_cells(run_cell, cells)
        results = {}
        for (arm, frames, seed), res in cell_results.items():
            results.setdefault(cell_key(arm, frames), {})[seed] = res
        write_checkpoint_document(Path(OUT_PATH), {
            c: {str(s): v for s, v in by.items()}
            for c, by in results.items()
        })

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    def bal(cell, s):
        return results[cell][s]["number"]["_balanced"]

    print("\n=== the split 2x2 (attested number balanced, MI readout) ===")
    for cell in sorted(results):
        print(ensemble(lambda s: bal(cell, s), SEEDS, label=f"{cell:12s}"))

    def overlap_bal(cell, s):
        r = results[cell][s]
        its = [it for it in r["items"]
               if it["attested"] and "overlap_correct" in it]
        if not its or not r.get("sg_total"):
            return float("nan")
        pl = sum(it["overlap_correct"] for it in its) / len(its)
        sg = r["sg_overlap_ok"] / r["sg_total"]
        return (pl + sg) / 2

    have_overlap = all("overlap_correct" in it
                       for c in results for s in SEEDS
                       for it in results[c][s]["items"][:1])
    if have_overlap:
        print("\n=== same cells, OVERLAP readout (the readout A/B) ===")
        for cell in sorted(results):
            print(ensemble(lambda s: overlap_bal(cell, s), SEEDS,
                           label=f"{cell:12s}"))

    print("\n=== registered bars ===")
    all_structural = all(results[c][s]["structural_ok"]
                         for c in results for s in SEEDS)
    print(f"A  structural: label images in disjoint areas, all cells: "
          f"{all_structural}")
    for cell in sorted(results):
        print(ensemble(
            lambda s: results[cell][s]["mean_mi_margin"], SEEDS,
            label=f"A  mean MI margin {cell}"))
    print(ensemble(lambda s: bal("ZIPF-200", s), SEEDS,
                   label="B  zipf-200 balanced (bar >= 0.75)"))

    def rho_mass(cell, s):
        its = [it for it in results[cell][s]["items"]
               if it["delta"] is not None and it["attested"]]
        return spearman([it["delta"] for it in its],
                        [1.0 if it["correct"] else 0.0 for it in its])

    vals = {s: rho_mass("ZIPF-200", s) for s in SEEDS}
    defined = [s for s in SEEDS if not np.isnan(vals[s])]
    print(f"C  rho(mass delta, correct) ZIPF-200 per seed: "
          f"{ {s: round(v, 3) for s, v in vals.items()} }")
    if len(defined) >= 3:
        print("   " + str(ensemble(
            lambda s: vals[s], defined,
            label=f"over {len(defined)}/{len(SEEDS)} defined seeds")))

    print(ensemble(
        lambda s: bal("UNIFORM-200", s) - bal("UNIFORM-50", s), SEEDS,
        label="D1 uniform budget slope (E14 was -0.160)"))
    u200 = ensemble(lambda s: bal("UNIFORM-200", s), SEEDS,
                    label="D2 uniform-200-split")
    print(f"{u200}  vs E14 shared-area {E14_UNIFORM_200[0]:.3f}"
          f"+/-{E14_UNIFORM_200[1]:.3f} (different readout -- caveat)")

    print("\n=== guards ===")
    for cell in sorted(results):
        print(ensemble(
            lambda s: results[cell][s]["tense"]["_balanced"], SEEDS,
            label=f"tense {cell}"))
    for cell in sorted(results):
        roles = [results[cell][s]["guards"] for s in SEEDS]
        ok = sum(r["roles_ok"] for r in roles)
        tot = sum(r["roles_total"] for r in roles)
        ag = sum(results[cell][s]["readout_agree"] for s in SEEDS)
        agt = sum(results[cell][s]["readout_agree_total"] for s in SEEDS)
        npl = results[cell][SEEDS[0]]["number"]["PL"]["n"]
        print(f"{cell}: roles {ok}/{tot} | readout agreement {ag}/{agt} "
              f"| attested PL n={npl}")

    print("\n=== zipf-200 per-item (pooled over seeds) ===")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for s in SEEDS:
        for it in results["ZIPF-200"][s]["items"]:
            if not it["attested"]:
                continue
            a = agg[it["form"]]
            a[0] += it["exposure"]
            a[1] += it["correct"]
            a[2] += 1
    for form, (e, ok, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {form:10s} exp/seed={e / n:5.1f}  acc={ok / n:.2f}")


if __name__ == "__main__":
    sys.exit(main())
