"""E14: the {sampling} x {budget} 2x2 -- the INTERACTION is the bar.

PRE-REGISTERED (task #143), written before any cell of the 2x2 trained.

THE CLAIM. E7: under UNIFORM sampling, budget cannot raise per-form
exposure (coverage widens exactly as fast as the budget -- pinned at
~1.5 at 50, 100, and 200 frames). E13: under Zipf at FIXED budget,
allocation cannot either (the corpus holds ~12 PL episodes total; the
SUM binds). But Zipf BOUNDS coverage (~9 attested PL forms at any
budget), so under Zipf -- and only under Zipf -- per-form exposure must
rise roughly linearly with budget. Neither axis alone moves reliability;
the COMPOSITION is the only thing that can. The registered bar is
therefore the interaction term, not any single cell.

PRE-RUN ARITHMETIC (the E10 lesson). zipf-200: ~200 frames x ~85% noun
subjects ~= 170 draws, x PLURAL_RATE 0.3 ~= 50 PL episodes over the
~9-12 zipf-bounded forms ~= 3-5 exposures each -- the ENTIRE attested
set enters the reliable regime (E11/E12/E13: forms at 2-4 exposures
read 0.8-1.0). uniform-200: E7 measured attested PL n -> ~40 with
exposure still ~1.2-1.5 -- flat. Each cell reports its total attested
PL exposure so this arithmetic is checked against measurement first.

THE SECOND QUESTION (rides free). Repetition failed by MERGING the
SG/PL label images (E12: 1-4 -> 8-17 shared cols under R4). A 200-frame
corpus also projects the labels ~4x more -- but through VARIED
sentences, not identical replays. The image guard therefore decides
whether merging follows sheer label-projection COUNT (bad for all
scaling) or identical replay specifically (bad only for the crutch we
abandoned).

CELLS: {UNIFORM, ZIPF} x FRAMES {50, 200} x seeds 42-51 (40 cells),
SLOW-scaled R1, n=3000. Budget and sampling are patched as module
attributes in the worker AFTER the fork (pre-stages stay at defaults in
every arm; the shared checkpoint is the point -- morph phases run at
SENTENCES). Defaults untouched on disk; the default corpus stays
sha-pinned by test_corpus_grammaticality.

REGISTERED BARS:
  I  INTERACTION: per-seed (zipf200 - zipf50) - (uni200 - uni50) on
     attested number balanced > 0, ensemble CI excluding zero.
  II zipf-200 attested balanced >= 0.75 -- the E-series bar, at last.
  III IMAGE GUARD: zipf-200 shared SG-PL image columns at the E13 floor
     (ensemble mean <= 4.0 of 30); all four cells reported so the
     count-vs-replay question is answered either way.
  IV MECHANISM: rho(mass delta, correct) over zipf-200's attested items
     positive with CI excluding zero (the E12 readout, third corpus).
  GUARDS: uni50 reproduces E13/E9 (0.650 envelope); tense within -0.05
     of uni50 in every cell; roles 4/4; attested PL n per cell reported
     (the growing-exam decomposition: uniform-200's exam GROWS ~4x,
     zipf-200's does not -- balanced numbers across cells are only
     comparable through the per-item tables).

DECISION RULE: I holds -> the pinning break is demonstrated; the arc
closes with the law "reliability = per-form mass; mass = exposure;
exposure = budget x allocation shape"; corpus-side scaling (Zipfian
real text at real size, the deferred CHILDES axis) becomes the
substrate's registered path, and the default-flip discussion opens
citing this run. I holds but II misses -> the law holds with a longer
exposure requirement than the arithmetic guessed; report the measured
exposure->accuracy curve. I fails with the exposure arithmetic
CONFIRMED (zipf200 forms really at 3-5) -> exposure rises but accuracy
does not follow at scale; the suspect is whatever breaks the mass->
accuracy link there (III's merging is candidate one). I fails with
exposure flat -> the zipf mechanism itself fails to concentrate at 200
(realization noise); fix the generator's delivery before any theory
moves.
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
BUDGETS = (50, 200)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("SB_N", "3000"))
if os.environ.get("SB_SMOKE") == "1":
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "sampling_budget_interaction_results.json")

from overlap_ceiling import spearman  # noqa: E402
from zipf_synthesis import FIXED_PL, FIXED_SG, ROLE_PROBES  # noqa: E402


def run_cell(arm: str, frames: int, seed: int) -> dict:
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
        gen.FRAMES_PER_STAGE = frames
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
    for form in sorted(set(FIXED_PL) | set(sets_["PL"])):
        mass = item_afferent_mass(parser, form, NUMBER, img)
        got, _diag = parser.recall_number(form)
        items.append({
            "form": form,
            "attested": form in sets_["PL"],
            "exposure": exposure.get(f"NUMBER:{form}", 0),
            "delta": (mass["PL"] - mass["SG"]) if mass else None,
            "correct": bool(got == "PL"),
            "answer": got,
        })
    total_exp = sum(it["exposure"] for it in items if it["attested"])

    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want

    print(f"[E14 {arm}-{frames} seed={seed}] "
          f"num bal={number['_balanced']:.3f} (PL n={number['PL']['n']}, "
          f"total exp={total_exp}) | tense={tense['_balanced']:.3f} | "
          f"shared cols={shared_cols} | roles {g_ok}/{g_total}", flush=True)
    return {"number": number, "tense": tense, "fixed": fixed,
            "shared_image_cols": shared_cols, "items": items,
            "total_attested_exposure": total_exp,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def cell_key(arm: str, frames: int) -> str:
    return f"{arm}-{frames}"


def main():
    if os.environ.get("SB_ANALYZE") == "1":
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
        with open(OUT_PATH, "w") as f:
            json.dump({c: {str(s): v for s, v in by.items()}
                       for c, by in results.items()}, f, indent=2)

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    def bal(cell, s):
        return results[cell][s]["number"]["_balanced"]

    print("\n=== the 2x2 (attested number balanced) ===")
    for cell in sorted(results):
        print(ensemble(lambda s: bal(cell, s), SEEDS, label=f"{cell:12s}"))

    print("\n=== registered bars ===")
    print(ensemble(
        lambda s: (bal("ZIPF-200", s) - bal("ZIPF-50", s))
        - (bal("UNIFORM-200", s) - bal("UNIFORM-50", s)),
        SEEDS, label="I   INTERACTION (zipf slope - uniform slope)"))
    print(ensemble(lambda s: bal("ZIPF-200", s), SEEDS,
                   label="II  zipf-200 balanced (bar >= 0.75)"))
    for cell in sorted(results):
        print(ensemble(
            lambda s: results[cell][s]["shared_image_cols"], SEEDS,
            label=f"III shared SG-PL image cols {cell}"))

    def rho_mass(cell, s):
        its = [it for it in results[cell][s]["items"]
               if it["delta"] is not None and it["attested"]]
        return spearman([it["delta"] for it in its],
                        [1.0 if it["correct"] else 0.0 for it in its])

    vals = {s: rho_mass("ZIPF-200", s) for s in SEEDS}
    defined = [s for s in SEEDS if not np.isnan(vals[s])]
    print(f"IV  rho(mass, correct) ZIPF-200 per seed: "
          f"{ {s: round(v, 3) for s, v in vals.items()} }")
    if len(defined) >= 3:
        print("    " + str(ensemble(
            lambda s: vals[s], defined,
            label=f"over {len(defined)}/{len(SEEDS)} defined seeds")))

    print("\n=== exposure arithmetic check ===")
    for cell in sorted(results):
        print(ensemble(
            lambda s: results[cell][s]["total_attested_exposure"], SEEDS,
            label=f"total attested PL exposure {cell}"))
        print(ensemble(
            lambda s: results[cell][s]["number"]["PL"]["n"], SEEDS,
            label=f"attested PL n           {cell}"))

    print("\n=== guards ===")
    for cell in sorted(results):
        print(ensemble(
            lambda s: results[cell][s]["tense"]["_balanced"], SEEDS,
            label=f"tense {cell}"))
    for cell in sorted(results):
        roles = [results[cell][s]["guards"] for s in SEEDS]
        ok = sum(r["roles_ok"] for r in roles)
        tot = sum(r["roles_total"] for r in roles)
        print(f"roles {cell}: {ok}/{tot}")
    print("E13/E9 reference: uni50 number 0.650 +/- 0.063")

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
