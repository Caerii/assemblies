"""E12: per-item drive decomposition -- where does the SG default live?

PRE-REGISTERED (task #141), written before any per-item mass was read.

E11 (#140) killed the representational hypothesis at its premise
(form<->lemma core overlap 0.00-0.03 == the unrelated-word floor, with
the self-overlap control at 1.000). What survived: 20/24 PL failures
answer SG DIRECTIONALLY with zero representational cause, and the one PL
form that escaped E7's exposure pinning ("girls", 4 eps/seed) is the one
form at 1.00 accuracy. The relocated model: each PL form's assembly is
disjoint from everything, so its number signal is only the afferent mass
its OWN ~1 training episode wrote toward the PL image; when that
marginal mass does not suffice, the probe defaults to the SG attractor
built by the area's total mass.

E12 reads that mass DIRECTLY, per item: the summed synaptic weight from
the form's core-assembly rows into the PL-image columns vs the SG-image
columns of NUMBER. Pure weight readout, no dynamics -- E10's census made
per-item. Shared image columns contribute equally to both sums, so the
difference delta = mass_PL - mass_SG is automatically the
exclusive-column contrast. Unlike the input_drive probe that could not
see role binding (it summed over ALL candidates), this readout is
WHERE-structured by construction: it sums over exactly the columns the
recall readout compares against.

INDEX-SPACE DISCIPLINE (the wmax_census pattern): rows are the form's
COMPACT core winners (area.winners immediately after the recall probe's
own activation), columns are the label images' COMPACT winners (built
with the exact _recall_morph_feature image protocol: stim projection +
rounds-1 recurrent) -- both in the weight matrix's own coordinate
system; no Assembly/neuron-ID snapshots anywhere near the matrix.

CELLS: SLOW-scaled x R in {1, 4} x seeds 42-46 (10 cells, forked
pre-stages). R1 is E11's config, so item accuracies replicate; R4 is the
arm that must explain why repetition failed the SCALED arm in E8/E9.

REGISTERED BARS:
  C1 MAIN: per-seed Spearman(delta, correct) over PL items at R1 is
     POSITIVE; ensemble over 5 seeds with CI excluding zero.
     C1' graded companion (reported): Spearman(delta, margin) where
     margin is the recall readout's own PL-minus-SG score.
  C2 "girls" -- the exposure-escapee -- ranks in the top 2 of the
     per-seed delta distribution in >= 3 of 5 seeds at R1.
  C3 THE R4 DISCRIMINATOR: paired per-item delta change R4 vs R1.
     (a) delta RISES under R4 but accuracy does not -> the mass is there
         and the ceiling is downstream (NUMBER recurrence / readout);
     (b) delta is FLAT under R4 (scaling holds the PL/SG ratio) -> named:
         normalization eats repetition's per-item gain, which is exactly
         why E8/E9's scaled arms were flat while OFF moved.
  GUARDS: per-cell PL accuracy within the E9/E11 envelope (balanced
     number ~0.65, R1 item accuracy ~4-6/10); image column overlap
     |PL image AND SG image| reported (near-total overlap would void the
     delta and the readout both).

DECISION RULE: C1 passes -> the SG default is per-item afferent mass;
the lever is corpus-side per-form exposure -- register the Zipf +
slow-scaling synthesis (repetition without swamping) as the closing
experiment of the arc. C1 fails with guards healthy -> the weight mass
does not carry the decision; the suspect moves to NUMBER's recurrent
dynamics between probe and readout (register a dynamics-on/off probe).
C3 picks (a) or (b) independently of C1 and either way retires the
"why did repetition fail under scaling" question.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_new_document

SEEDS = list(range(42, 47))
REPS = (1, 4)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("DD_N", "3000"))
if os.environ.get("DD_SMOKE") == "1":
    SEEDS = [42]
    REPS = (1,)

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "drive_decomposition_results.json")

from overlap_ceiling import spearman  # noqa: E402


def run_cell(reps: int, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.ops import project as ops_project
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
        p.morph_repetitions = reps

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"gm-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)
    brain = parser.brain
    eng = brain._engine
    rounds = max(1, int(parser.rounds))

    def compact_image(area: str, stim: str) -> "list[int]":
        """A label's image in `area` -- the recall readout's own protocol,
        winners left in COMPACT space (the weight matrix's coordinates)."""
        brain.inhibit_areas([area])
        brain.project({stim: [area]}, {})
        for _ in range(rounds - 1):
            brain.project({stim: [area]}, {area: [area]})
        return [int(c) for c in brain.areas[area].winners]

    sets_ = attested_morph_sets(parser)
    exposure = getattr(parser, "_morph_exposure", {})
    items = []
    with brain.read_only():
        img = {lab: compact_image(NUMBER, f"number_{lab}")
               for lab in ("SG", "PL")}
        shared_cols = len(set(img["SG"]) & set(img["PL"]))
        for form in sets_["PL"]:
            core = parser._word_core_area(form)
            phon = parser.stim_map.get(form)
            conn = eng._area_conns.get(core, {}).get(NUMBER)
            w = getattr(conn, "weights", None) if conn is not None else None
            if phon is None or w is None or getattr(w, "ndim", 0) != 2:
                continue
            # The recall probe's own activation, rows read in compact space.
            brain.inhibit_areas([NUMBER])
            ops_project(brain, phon, core, rounds=rounds)
            rows = [int(r) for r in brain.areas[core].winners
                    if int(r) < w.shape[0]]
            mass = {}
            for lab in ("SG", "PL"):
                cols = [c for c in img[lab] if c < w.shape[1]]
                mass[lab] = (float(np.asarray(
                    w[np.ix_(rows, cols)]).sum())
                    if rows and cols else 0.0)
            got, diag = parser.recall_number(form)
            scores = diag.get("scores", {})
            margin = (float(scores["PL"] - scores["SG"])
                      if "PL" in scores and "SG" in scores else None)
            items.append({
                "form": form,
                "mass_PL": mass["PL"], "mass_SG": mass["SG"],
                "delta": mass["PL"] - mass["SG"],
                "answer": got, "correct": bool(got == "PL"),
                "margin": margin,
                "exposure": exposure.get(f"NUMBER:{form}", 0),
            })

    n_ok = sum(it["correct"] for it in items)
    ds = [it["delta"] for it in items]
    print(f"[E12 R{reps} seed={seed}] PL {n_ok}/{len(items)} | "
          f"delta mean={np.mean(ds):+.2f} range=({min(ds):+.2f},"
          f"{max(ds):+.2f}) | shared image cols={shared_cols}/"
          f"{len(img['PL'])}", flush=True)
    return {"items": items, "shared_image_cols": shared_cols,
            "image_k": len(img["PL"])}


def main():
    from _parallel import run_cells

    cells = [(r, s) for r in REPS for s in SEEDS]
    cell_results = run_cells(run_cell, cells)
    results: dict = {}
    for (reps, seed), res in cell_results.items():
        results.setdefault(f"R{reps}", {})[seed] = res
    write_new_document(Path(OUT_PATH), {
        c: {str(s): v for s, v in by.items()}
        for c, by in results.items()
    })

    if len(SEEDS) < 3:
        print("(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    def per_seed(reps, s):
        its = results[f"R{reps}"][s]["items"]
        d = np.array([it["delta"] for it in its])
        ok = np.array([1.0 if it["correct"] else 0.0 for it in its])
        mg = np.array([it["margin"] if it["margin"] is not None else np.nan
                       for it in its])
        keep = ~np.isnan(mg)
        return {
            "rho_correct": spearman(d, ok),
            "rho_margin": (spearman(d[keep], mg[keep])
                           if keep.sum() >= 3 else float("nan")),
            "sep": (float(d[ok == 1].mean() - d[ok == 0].mean())
                    if (ok == 1).any() and (ok == 0).any()
                    else float("nan")),
            "girls_rank": (1 + int(np.sum(
                d > d[[it["form"] for it in its].index("girls")]))
                if "girls" in [it["form"] for it in its] else None),
            "mean_delta": float(d.mean()),
            "mean_mass_PL": float(np.mean([it["mass_PL"] for it in its])),
            "mean_mass_SG": float(np.mean([it["mass_SG"] for it in its])),
        }

    stats = {r: {s: per_seed(r, s) for s in SEEDS} for r in REPS}

    print("\n=== registered bars ===")
    print(ensemble(lambda s: stats[1][s]["rho_correct"], SEEDS,
                   label="C1  rho(delta, correct) R1"))
    print(ensemble(lambda s: stats[1][s]["rho_margin"], SEEDS,
                   label="C1' rho(delta, margin) R1"))
    print(ensemble(lambda s: stats[1][s]["sep"], SEEDS,
                   label="C1s delta(pass)-delta(fail) R1"))
    ranks = [stats[1][s]["girls_rank"] for s in SEEDS]
    top2 = sum(1 for r in ranks if r is not None and r <= 2)
    print(f"C2  girls delta-rank per seed: {ranks} -> top-2 in "
          f"{top2}/5 (bar >= 3)")
    if 4 in REPS:
        print(ensemble(
            lambda s: stats[4][s]["mean_delta"] - stats[1][s]["mean_delta"],
            SEEDS, label="C3  mean delta R4 - R1"))
        print(ensemble(
            lambda s: (stats[4][s]["mean_mass_PL"]
                       - stats[1][s]["mean_mass_PL"]),
            SEEDS, label="C3a mean mass_PL R4 - R1"))
        print(ensemble(
            lambda s: (stats[4][s]["mean_mass_SG"]
                       - stats[1][s]["mean_mass_SG"]),
            SEEDS, label="C3b mean mass_SG R4 - R1"))
        print(ensemble(lambda s: stats[4][s]["rho_correct"], SEEDS,
                       label="    rho(delta, correct) R4"))

    print("\n=== guards ===")
    for r in REPS:
        accs = [np.mean([it["correct"]
                         for it in results[f"R{r}"][s]["items"]])
                for s in SEEDS]
        sh = [results[f"R{r}"][s]["shared_image_cols"] for s in SEEDS]
        print(f"R{r}: PL item acc per seed "
              f"{[f'{a:.2f}' for a in accs]} | shared image cols {sh}")


if __name__ == "__main__":
    sys.exit(main())
