"""E13: the synthesis -- Zipfian frame allocation x slow scaling.

PRE-REGISTERED (task #142), written before any Zipf corpus ever trained.

THE CHAIN THAT LEADS HERE. E12 (#141): per-item afferent mass into the
PL-vs-SG image columns decides number recall (rho 0.846); per-form
exposure is what the mass is made of (E7/E8/E11); and the naive exposure
lever -- phase repetition -- self-defeats under scaling because it
multiplies the label-stimulus projections along with the episodes and
MERGES the label images (1-4 -> 8-17 shared cols of 30). The surviving
lever is REDISTRIBUTION: Zipfian rank-frequency subject sampling raises
per-form exposure on head forms at CONSTANT total episode and label-stim
budget -- no image-merging pressure by construction.

PRE-RUN ARITHMETIC (the E10 lesson). ~50 frames/stage, ~85% noun
subjects ~= 42 noun-subject draws; s=1.0 over ~30-40 stage nouns gives
H ~= 4.0-4.3, so rank-1 draws ~= 42/H ~= 10, x PLURAL_RATE 0.3 ~= 3 PL
episodes; rank-2 ~= 1.5; rank-3 ~= 1. E11/E12 measured the exposure
escapee ("girls", 4 eps) at accuracy 1.00 and rank ~1 mass, so the
prediction is 2-4 head PL forms entering the reliable regime while the
tail thins BELOW uniform -- redistribution has a COST, and the fixed
probe set is where it must show (E6's conjugacy read from the other
side). The rank assignment is imposed (stage noun order); what is
Zipfian is the distribution.

DESIGN: arms {UNIFORM, ZIPF} x seeds 42-51 (20 cells), SLOW-scaled R=1,
n=3000, forked pre-stages (pre-stages stay uniform in BOTH arms by
design: the checkpoint is shared and the morph phases run at SENTENCES+;
the ZIPF worker patches generation.SUBJECT_SAMPLING after the fork).
UNIFORM re-runs rather than importing E9's JSON so every quantity
(fixed probe, image cols, per-item mass) is measured identically -- and
its balanced number must reproduce E9's 0.650 +/- 0.063 as a guard.

REGISTERED BARS:
  A1 ATTESTED EXAM: paired ZIPF-UNIFORM number balanced > 0 with the CI
     excluding zero. ASPIRATIONAL: ZIPF mean >= 0.75 (the E-series bar).
  A2 FIXED PROBE (the exam that cannot shrink): paired delta on the
     FIXED_PL/FIXED_SG sets below (the uniform arm's stable attested
     items, from E11). REGISTERED READING: A1 passing with A2 flat or
     negative means part of A1 is exam shrinkage -- redistribution moves
     reliability to where the exposure goes and the decomposition must be
     reported, not averaged away.
  B  IMAGE GUARD (the E12 mechanism, now a bar): ZIPF mean shared
     SG-PL image columns <= UNIFORM mean + 1 (uniform floor was 1-4/30).
     Fails -> Zipf smuggled multiplication in; the design is wrong, not
     the theory.
  C  MECHANISM CHECK (E12 replicated under redistribution): within the
     ZIPF arm, per-item Spearman(exposure, correct) and
     Spearman(mass delta, correct) over the FIXED_PL items positive,
     ensemble CI excluding zero.
  GUARDS: UNIFORM number balanced within E9's envelope; tense paired
     delta >= -0.05; roles 4/4 on the two probe sentences.

DECISION RULE: A1+A2 both positive -> redistribution beats uniform
outright; flip SUBJECT_SAMPLING's default in a SEPARATE commit citing
this run, and the arc closes with the corpus as the lever. A1 passes, A2
flat -> the conjugacy is conserved at fixed budget: Zipf reallocates but
cannot create; the honest close is "exposure decides, budget binds", and
the scaling story moves to the episode budget (E7's axis) with Zipfian
allocation as the default SHAPE. A1 fails -> redistribution does not
even win the attested exam; the readout suspect (score_recall image
structure) is next. B fails -> stop, fix the design.
"""
from __future__ import annotations

import json
import os
import random
import sys

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

SEEDS = list(range(42, 52))
ARMS = ("UNIFORM", "ZIPF")
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("ZS_N", "3000"))
if os.environ.get("ZS_SMOKE") == "1":
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "zipf_synthesis_results.json")

#: The uniform arm's stable attested PL items (E11, seeds 42-46) and their
#: lemmas -- the exam that cannot shrink when Zipf changes what is attested.
FIXED_PL = ["boys", "girls", "babies", "ears", "hands",
            "legs", "hearts", "gifts", "windows", "winds"]
FIXED_SG = ["boy", "girl", "baby", "ear", "hand",
            "leg", "heart", "gift", "window", "wind"]

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
]

from overlap_ceiling import spearman  # noqa: E402


def run_cell(arm: str, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
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
        ss = frozenset({TENSE, NUMBER})
        p.brain._engine.synaptic_scaling = ss
        p.brain._synaptic_scaling = ss
        p.brain._engine.synaptic_scaling_deferred = True
        p.morph_repetitions = 1
        if arm == "ZIPF":
            gen.SUBJECT_SAMPLING = "zipf"

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"zs-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)

    sets_ = attested_morph_sets(parser)
    number = score_recall(parser.recall_number,
                          {k: sets_[k] for k in ("SG", "PL")})
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})
    fixed = score_recall(parser.recall_number,
                         {"SG": FIXED_SG, "PL": FIXED_PL})

    img = feature_images_compact(
        parser, NUMBER, {"SG": "number_SG", "PL": "number_PL"})
    shared_cols = (len(set(img.get("SG", [])) & set(img.get("PL", [])))
                   if len(img) == 2 else None)

    exposure = getattr(parser, "_morph_exposure", {})
    items = []
    for form in FIXED_PL:
        mass = item_afferent_mass(parser, form, NUMBER, img)
        got, _diag = parser.recall_number(form)
        items.append({
            "form": form,
            "exposure": exposure.get(f"NUMBER:{form}", 0),
            "delta": (mass["PL"] - mass["SG"]) if mass else None,
            "correct": bool(got == "PL"),
            "answer": got,
        })

    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want

    print(f"[E13 {arm} seed={seed}] num bal={number['_balanced']:.3f} "
          f"(attested PL n={number['PL']['n']}) | fixed bal="
          f"{fixed['_balanced']:.3f} | tense={tense['_balanced']:.3f} | "
          f"shared cols={shared_cols} | roles {g_ok}/{g_total}", flush=True)
    return {"number": number, "tense": tense, "fixed": fixed,
            "shared_image_cols": shared_cols, "items": items,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def main():
    from _parallel import run_cells

    cells = [(a, s) for a in ARMS for s in SEEDS]
    cell_results = run_cells(run_cell, cells)
    results: dict = {a: {} for a in ARMS}
    for (arm, seed), res in cell_results.items():
        results[arm][seed] = res
    with open(OUT_PATH, "w") as f:
        json.dump({a: {str(s): v for s, v in by.items()}
                   for a, by in results.items()}, f, indent=2)

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble, paired_delta

    def ens(arm, path, label):
        def get(s):
            v = results[arm][s]
            for k in path:
                v = v[k]
            return v
        return ensemble(get, SEEDS, label=label)

    print("\n=== registered bars ===")
    zn = ens("ZIPF", ("number", "_balanced"), "ZIPF number balanced")
    un = ens("UNIFORM", ("number", "_balanced"), "UNIFORM number balanced")
    print(un)
    print(zn)
    print(paired_delta(zn, un, label="A1 ZIPF-UNIFORM attested"))
    zf = ens("ZIPF", ("fixed", "_balanced"), "ZIPF fixed-probe balanced")
    uf = ens("UNIFORM", ("fixed", "_balanced"),
             "UNIFORM fixed-probe balanced")
    print(uf)
    print(zf)
    print(paired_delta(zf, uf, label="A2 ZIPF-UNIFORM fixed"))
    for arm in ARMS:
        print(ens(arm, ("shared_image_cols",),
                  f"B  {arm} shared SG-PL image cols"))

    def per_seed_rho(s, key):
        its = [it for it in results["ZIPF"][s]["items"]
               if it["delta"] is not None]
        x = [it[key] for it in its]
        ok = [1.0 if it["correct"] else 0.0 for it in its]
        return spearman(x, ok)

    print(ensemble(lambda s: per_seed_rho(s, "exposure"), SEEDS,
                   label="C  rho(exposure, correct) ZIPF"))
    print(ensemble(lambda s: per_seed_rho(s, "delta"), SEEDS,
                   label="C  rho(mass delta, correct) ZIPF"))

    print("\n=== guards ===")
    print(ens("ZIPF", ("tense", "_balanced"), "ZIPF tense"))
    print(ens("UNIFORM", ("tense", "_balanced"), "UNIFORM tense"))
    print("E9 SLOW R1 reference: number 0.650 +/- 0.063")
    for arm in ARMS:
        roles = [results[arm][s]["guards"] for s in SEEDS]
        ok = sum(r["roles_ok"] for r in roles)
        tot = sum(r["roles_total"] for r in roles)
        npl = [results[arm][s]["number"]["PL"]["n"] for s in SEEDS]
        print(f"{arm}: roles {ok}/{tot} | attested PL n per seed {npl}")

    print("\n=== ZIPF per-item (pooled) ===")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for s in SEEDS:
        for it in results["ZIPF"][s]["items"]:
            a = agg[it["form"]]
            a[0] += it["exposure"]
            a[1] += it["correct"]
            a[2] += 1
    for form, (e, ok, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {form:10s} exp/seed={e / n:4.1f}  acc={ok / n:.2f}")


if __name__ == "__main__":
    sys.exit(main())
