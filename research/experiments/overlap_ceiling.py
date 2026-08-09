"""E11: does form<->lemma core overlap cap PL accuracy? The representational test.

PRE-REGISTERED (task #140), written before any overlap was measured.

E1-E10 excluded mass, exposure contrast, substrate size, form diversity,
corpus size, scheduling, and w_max saturation for scaled-number's 0.65
ceiling. The surviving hypothesis is REPRESENTATIONAL: a plural form
("dogs") shares its GroundingContext verbatim with its lemma ("dog") and
most of its phonology, so its core assembly substantially overlaps the
lemma's -- and the lemma trained SG many times. The PL->NUMBER probe then
drives largely through SG-trained synapses; the number signal rides only
on the non-shared fraction, which is fixed at RECRUITMENT time and cannot
be enlarged by any learning-side lever. That would explain the whole
E-series pattern: every manipulation moved everything except this number.

DESIGN: one config (SLOW-scaled R1, the adoption-ready form; E9 mean
0.650 +/- 0.063), seeds 42-46, n=3000, forked pre-stages. E11 is
CORRELATIONAL -- the variance is across ITEMS, not configs. Per attested
PL form (and PAST form, as the cross-feature contrast): overlap with its
own lemma's core image, recall correctness, error direction, exposure.

MEASUREMENT DISCIPLINE. The core image is taken by replicating the recall
probe's own activation verbatim (ops.project(phon, core, rounds) inside
read_only) and snapshotting with _snap, so both operands of every overlap
are Assembly neuron-ID snapshots from the same area -- the two-index-space
trap is closed by the type, not by care. All probing inside read_only().

REGISTERED BARS:
  B0 POWER GATE (OC_GATE=1, seed 42 only, read BEFORE the full run -- the
     E3/E4 arms-must-differ lesson as a formal bar): across PL forms,
     sd(overlap) > 0 and range >= 0.15. Fails -> the item-level test is
     UNPOWERED; stop, report, and pivot to the intervention A/B
     (manipulate phon separation, watch the ceiling).
  B1 MAIN: per-seed Spearman(overlap, correct) over PL items is negative;
     ensemble over 5 seeds with CI excluding zero.
     B1' graded companion (reported, not gating): Spearman(overlap,
     margin_PL_minus_SG) -- same sign, more power than binary correct.
  B2 SEPARATION: mean overlap of FAILED PL forms minus PASSED, per seed,
     ensemble CI excluding zero (positive).
  B3 ERROR FINGERPRINT (the sharpest signature, free): failed PL forms
     answer SG specifically (modal error SG, not tie), and the SG-error
     rate rises with overlap. A readout- or noise-side ceiling has no
     reason to produce DIRECTIONAL errors.
  B4 CONFOUND CONTROL: (a) partial Spearman controlling per-form NUMBER
     exposure keeps B1's sign; (b) a placebo predictor -- overlap with a
     rotated (unrelated) lemma -- correlates with nothing.
  B5 CROSS-FEATURE: tense composed to 0.729 where number saturated; the
     hypothesis owes the explanation and makes one: PAST forms sit LOWER
     on the same overlap distribution (ensemble of per-seed mean
     overlap(PAST) - mean overlap(PL) < 0), and one pooled overlap->margin
     relation covers both features.

DECISION RULE: B1+B2 pass (B3 the clincher) -> ceiling is representational;
register interventions separately: (a) phon share for inflected surfaces
at recruitment, (b) dedicated inflection route (the papers' LEX-layer
territory). B0 passes but B1 flat -> substrate-side suspect list is empty;
attention moves to the READOUT (score_recall image structure). B0 fails ->
item-level route closed; causal A/B instead.

CAVEAT (registered): 5 seeds x ~20 PL forms makes per-form accuracy
coarse (quintiles); B1 rides on item spread. If B0 shows healthy overlap
variance but accuracy is too quantized, widen to 10 seeds (forked
checkpoints make seeds nearly free) before reading the bars.
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

SEEDS = list(range(42, 47))
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("OC_N", "3000"))
GATE = os.environ.get("OC_GATE") == "1"
if GATE:
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "overlap_ceiling_results.json")


# ---------------------------------------------------------------------------
# Rank statistics (numpy-only; scipy.stats stays off the import path)
# ---------------------------------------------------------------------------

def _rankdata(a) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman(x, y) with z partialled out (least squares on ranks)."""
    rx, ry, rz = _rankdata(x), _rankdata(y), _rankdata(z)
    A = np.vstack([np.ones_like(rz), rz]).T

    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    ex, ey = resid(rx), resid(ry)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


# ---------------------------------------------------------------------------
# Form -> lemma maps, straight from the lexicon data (the same source
# attested_morph_sets scans, so exclusions line up by construction).
# ---------------------------------------------------------------------------

def form_lemma_maps() -> "tuple[dict, dict]":
    from neural_assemblies.lexicon.data import NOUNS, VERBS

    pl_to_lemma = {e["forms"]["plural"]: e["lemma"] for e in NOUNS
                   if e.get("forms", {}).get("plural")}
    past_to_lemma = {e["forms"]["past"]: e["lemma"] for e in VERBS
                     if e.get("forms", {}).get("past")}
    return pl_to_lemma, past_to_lemma


def run_cell(seed: int) -> dict:
    from neural_assemblies.assembly_calculus.assembly import (
        overlap as assembly_overlap,
    )
    from neural_assemblies.assembly_calculus.ops import (
        project as ops_project, _snap,
    )
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation \
        .morph_features import attested_morph_sets
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

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"gm-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)
    brain = parser.brain
    rounds = max(1, int(parser.rounds))

    def core_image(word: str):
        """The recall probe's own activation, snapshotted at the core.

        Verbatim replica of _recall_morph_feature's probe prefix: project
        the word's phon into its core area for `rounds`, inside
        read_only(). _snap returns an Assembly (neuron-ID space), so
        overlaps between these snapshots are same-space by type.
        """
        core = parser._word_core_area(word)
        phon = parser.stim_map.get(word)
        if phon is None or core not in brain.areas:
            return None, core
        with brain.read_only():
            ops_project(brain, phon, core, rounds=rounds)
            return _snap(brain, core), core

    sets_ = attested_morph_sets(parser)
    pl_to_lemma, past_to_lemma = form_lemma_maps()
    exposure = getattr(parser, "_morph_exposure", {})

    out: dict = {"items": {"PL": [], "PAST": []}, "skipped": []}
    specs = (
        ("PL", sets_["PL"], pl_to_lemma, parser.recall_number, "SG",
         "NUMBER"),
        ("PAST", sets_["PAST"], past_to_lemma, parser.recall_tense,
         "PRESENT", "TENSE"),
    )
    for label, forms, lemma_map, recall, rival, feat in specs:
        # Rotated-lemma placebo: each form gets the NEXT form's lemma as a
        # frequency-comparable but unrelated predictor (B4b).
        lemmas = [lemma_map.get(f) for f in forms]
        for i, form in enumerate(forms):
            lemma = lemmas[i]
            placebo_lemma = lemmas[(i + 1) % len(forms)] if len(forms) > 1 \
                else None
            if lemma is None or lemma not in parser.stim_map:
                out["skipped"].append((label, form, "lemma unattested"))
                continue
            form_img, core_f = core_image(form)
            lemma_img, core_l = core_image(lemma)
            if form_img is None or lemma_img is None or core_f != core_l:
                out["skipped"].append((label, form, "no core image"))
                continue
            ov = float(assembly_overlap(form_img, lemma_img))
            placebo = None
            if placebo_lemma and placebo_lemma in parser.stim_map \
                    and placebo_lemma != lemma:
                pl_img, core_p = core_image(placebo_lemma)
                if pl_img is not None and core_p == core_f:
                    placebo = float(assembly_overlap(form_img, pl_img))
            got, diag = recall(form)
            scores = diag.get("scores", {})
            margin = None
            if label in scores and rival in scores:
                margin = float(scores[label] - scores[rival])
            out["items"][label].append({
                "form": form, "lemma": lemma, "overlap": ov,
                "placebo_overlap": placebo,
                "answer": got, "correct": bool(got == label),
                "margin": margin,
                "exposure_form": exposure.get(f"{feat}:{form}", 0),
                "exposure_lemma": exposure.get(f"{feat}:{lemma}", 0),
            })

    pl = out["items"]["PL"]
    n_ok = sum(it["correct"] for it in pl)
    print(f"[E11 seed={seed}] PL {n_ok}/{len(pl)} correct | "
          f"overlap mean={np.mean([it['overlap'] for it in pl]):.3f} "
          f"range=({min(it['overlap'] for it in pl):.3f},"
          f"{max(it['overlap'] for it in pl):.3f}) | "
          f"skipped={len(out['skipped'])}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Analysis against the registered bars
# ---------------------------------------------------------------------------

def _per_seed(items: "list[dict]") -> dict:
    ov = np.array([it["overlap"] for it in items])
    ok = np.array([1.0 if it["correct"] else 0.0 for it in items])
    mg = np.array([it["margin"] if it["margin"] is not None else np.nan
                   for it in items])
    exp_f = np.array([it["exposure_form"] for it in items], float)
    failed, passed = ov[ok == 0], ov[ok == 1]
    res = {
        "rho_correct": spearman(ov, ok),
        "rho_margin": (spearman(ov[~np.isnan(mg)], mg[~np.isnan(mg)])
                       if (~np.isnan(mg)).sum() >= 3 else float("nan")),
        "sep_fail_minus_pass": (float(failed.mean() - passed.mean())
                                if len(failed) and len(passed)
                                else float("nan")),
        "rho_partial_exposure": partial_spearman(ov, ok, exp_f),
        "mean_overlap": float(ov.mean()),
    }
    plac = [(it["overlap"], it["placebo_overlap"],
             1.0 if it["correct"] else 0.0)
            for it in items if it["placebo_overlap"] is not None]
    if len(plac) >= 3:
        res["rho_placebo"] = spearman([p[1] for p in plac],
                                      [p[2] for p in plac])
    return res


def main():
    from _parallel import run_cells

    cell_results = run_cells(run_cell, [(s,) for s in SEEDS])
    results = {seed: res for (seed,), res in cell_results.items()}
    with open(OUT_PATH, "w") as f:
        json.dump({str(s): v for s, v in results.items()}, f, indent=2)

    # ---- B0 power gate (read on seed 42; gates the full run) ----
    pl42 = results[42]["items"]["PL"]
    ov42 = np.array([it["overlap"] for it in pl42])
    rng = float(ov42.max() - ov42.min())
    print(f"\n=== B0 power gate (seed 42, {len(pl42)} PL items) ===")
    print(f"sd(overlap)={ov42.std():.4f}  range={rng:.4f}  "
          f"(bar: sd>0 and range>=0.15)")
    print(f"B0 {'PASSES' if ov42.std() > 0 and rng >= 0.15 else 'FAILS'}")
    if GATE:
        print("(gate mode: stopping before the registered bars)")
        return

    if len(SEEDS) < 3:
        print("(too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    stats = {s: {lab: _per_seed(results[s]["items"][lab])
                 for lab in ("PL", "PAST")} for s in SEEDS}

    print("\n=== registered bars (5-seed ensembles) ===")
    print(ensemble(lambda s: stats[s]["PL"]["rho_correct"], SEEDS,
                   label="B1  rho(overlap, correct) PL"))
    print(ensemble(lambda s: stats[s]["PL"]["rho_margin"], SEEDS,
                   label="B1' rho(overlap, margin) PL"))
    print(ensemble(lambda s: stats[s]["PL"]["sep_fail_minus_pass"], SEEDS,
                   label="B2  overlap(fail)-overlap(pass) PL"))
    print(ensemble(lambda s: stats[s]["PL"]["rho_partial_exposure"], SEEDS,
                   label="B4a partial rho | exposure PL"))
    print(ensemble(lambda s: stats[s]["PL"].get("rho_placebo", np.nan),
                   SEEDS, label="B4b placebo rho PL"))
    print(ensemble(
        lambda s: (stats[s]["PAST"]["mean_overlap"]
                   - stats[s]["PL"]["mean_overlap"]),
        SEEDS, label="B5  mean_overlap PAST - PL"))

    print("\n=== B3 error fingerprint (pooled over seeds) ===")
    sg_err = tie_err = 0
    fail_ov, sg_flag = [], []
    for s in SEEDS:
        for it in results[s]["items"]["PL"]:
            if it["correct"]:
                continue
            fail_ov.append(it["overlap"])
            is_sg = it["answer"] == "SG"
            sg_flag.append(1.0 if is_sg else 0.0)
            sg_err += is_sg
            tie_err += it["answer"] is None
    print(f"failures: {len(fail_ov)}  answered-SG: {sg_err}  "
          f"ties: {tie_err}  other: {len(fail_ov) - sg_err - tie_err}")
    # Among ALL PL items pooled: does the SG-error rate rise with overlap?
    all_ov = [it["overlap"] for s in SEEDS
              for it in results[s]["items"]["PL"]]
    all_sg = [1.0 if (not it["correct"] and it["answer"] == "SG") else 0.0
              for s in SEEDS for it in results[s]["items"]["PL"]]
    print(f"pooled rho(overlap, answered-SG) = "
          f"{spearman(all_ov, all_sg):.4f}")


if __name__ == "__main__":
    sys.exit(main())
