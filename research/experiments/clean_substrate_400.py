"""E17: zipf-400 on the CLEAN substrate -- the one unmeasured cell.

PRE-REGISTERED (task #146), after the E16 revert (f096548) and before
any clean-substrate 400-frame cell trained.

E16 measured zipf-400 only on the recurrence-damaged substrate (best
0.630) and the paired comparison convicted the substrate change itself
(-0.090 on the overlap readout at 200). The program's best-known
configuration -- split areas, feed-forward label training, within-area
overlap readout -- has therefore NEVER been measured at 400 frames.
E15's slopes say it should improve (+0.172 from 50 to 200); the E16
damage says nothing about this cell. One cell, two bars, then the arc
gets its honest verdict either way.

CELLS: ZIPF x FRAMES {200, 400} x seeds 42-51 (20 cells), split
architecture, SLOW scaling on value areas, CURRENT (post-revert) code.
Readouts: MI oneshot (the paper's commit device) and within-area
overlap (the production candidate), both per item.

REGISTERED BARS:
  R  REVERT VERIFIED BY MEASUREMENT: zipf-200 overlap replicates E15's
     0.700 +/- 0.037 (paired per-seed delta vs the E15 JSON ~ 0). If it
     does not, the revert missed something and NOTHING else here is
     interpretable.
  V  THE BAR: zipf-400 overlap balanced >= 0.75 on the ATTESTED exam.
  V' THE FIXED EXAM (the growing-exam lesson: attested n grows 30 -> 43
     at 400, so V alone conflates learning with exam drift): the same
     readout scored on FIXED_PL_200 (zipf-200's 30 attested forms,
     deterministic) + their SG lemmas. Reported alongside V; the honest
     headline is whichever is LOWER.
  GUARDS: roles 4/4; tense reported; per-item exposure->accuracy table
     at 400 (the law's shape on the clean substrate at high exposure --
     E16's flattening should NOT reproduce here; if it does, the
     flattening was never the recurrence and the attractor story needs
     rework).

DECISION RULE: R+V pass -> the number arc CLOSES at the E-series bar
with the final configuration named (split + feed-forward labels +
overlap readout + Zipf corpus at sufficient budget); adoption and
CHILDES discussions open. R passes, V fails with the exposure law
healthy -> the residual is the readout ceiling at high coverage; fit
the exposure->accuracy curve and state the corpus size the law implies
-- the arc closes with a LAW instead of a bar. R fails -> stop;
find what the revert missed before believing anything.
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
BUDGETS = (200, 400)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("CS_N", "3000"))
if os.environ.get("CS_SMOKE") == "1":
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "clean_substrate_400_results.json")

#: zipf-200's attested PL forms (deterministic corpus) -- the fixed exam.
FIXED_PL_200 = [
    "babies", "books", "boys", "chairs", "children", "clouds", "dads",
    "dogs", "eyes", "fires", "foods", "friends", "gifts", "girls",
    "heads", "hearts", "houses", "men", "mice", "moms", "mothers",
    "papers", "pictures", "rocks", "rooms", "suns", "tables", "trees",
    "women", "worlds",
]

E15_OVERLAP_200 = 0.700  # replication reference (R)

from zipf_synthesis import ROLE_PROBES  # noqa: E402


def run_cell(frames: int, seed: int) -> dict:
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
        .morph_features import attested_morph_sets, score_recall
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
        value_areas = frozenset(
            feature_value_area(f, lab)
            for f in (TENSE, NUMBER)
            for lab in FEATURE_VALUE_LABELS[f])
        p.brain._engine.synaptic_scaling = value_areas
        p.brain._synaptic_scaling = value_areas
        p.brain._engine.synaptic_scaling_deferred = True
        p.morph_repetitions = 1
        gen.FRAMES_PER_STAGE = frames
        gen.SUBJECT_SAMPLING = "zipf"

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"zs-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)

    sets_ = attested_morph_sets(parser)
    exposure = getattr(parser, "_morph_exposure", {})
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})

    def score(pl_forms, sg_forms):
        mi = {"PL": 0, "SG": 0}
        ov = {"PL": 0, "SG": 0}
        items = []
        for label, forms in (("PL", pl_forms), ("SG", sg_forms)):
            for form in forms:
                got, diag = parser.recall_number(form)
                mi[label] += got == label
                ov[label] += diag.get("overlap_answer") == label
                if label == "PL":
                    items.append({
                        "form": form,
                        "exposure": exposure.get(f"NUMBER:{form}", 0),
                        "mi_correct": bool(got == label),
                        "ov_correct": bool(
                            diag.get("overlap_answer") == label),
                    })
        n_pl, n_sg = len(pl_forms), len(sg_forms)
        return {
            "mi_balanced": (mi["PL"] / n_pl + mi["SG"] / n_sg) / 2,
            "ov_balanced": (ov["PL"] / n_pl + ov["SG"] / n_sg) / 2,
            "items": items,
        }

    attested = score(sets_["PL"], sets_["SG"])
    # Fixed exam's SG side: mapped by lexicon (irregulars: men -> man).
    from neural_assemblies.lexicon.data import NOUNS
    pl2lem = {e["forms"]["plural"]: e["lemma"] for e in NOUNS
              if e.get("forms", {}).get("plural")}
    fixed_sg = [pl2lem.get(f, f) for f in FIXED_PL_200]
    fixed = score(FIXED_PL_200, fixed_sg)

    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want

    total_exp = sum(it["exposure"] for it in attested["items"])
    print(f"[E17 zipf-{frames} seed={seed}] "
          f"attested mi={attested['mi_balanced']:.3f} "
          f"ov={attested['ov_balanced']:.3f} | fixed "
          f"mi={fixed['mi_balanced']:.3f} ov={fixed['ov_balanced']:.3f} | "
          f"tense={tense['_balanced']:.3f} | PL n={len(sets_['PL'])} "
          f"exp={total_exp} | roles {g_ok}/{g_total}", flush=True)
    return {"attested": attested, "fixed": fixed,
            "tense_balanced": tense["_balanced"],
            "attested_pl_n": len(sets_["PL"]),
            "total_attested_exposure": total_exp,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def main():
    if os.environ.get("CS_ANALYZE") == "1":
        with open(OUT_PATH) as f:
            raw = json.load(f)
        results = {int(b): {int(s): v for s, v in by.items()}
                   for b, by in raw.items()}
    else:
        from _parallel import run_cells

        cells = [(b, s) for b in BUDGETS for s in SEEDS]
        cell_results = run_cells(run_cell, cells)
        results = {}
        for (frames, seed), res in cell_results.items():
            results.setdefault(frames, {})[seed] = res
        write_checkpoint_document(Path(OUT_PATH), {
            str(b): {str(s): v for s, v in by.items()}
            for b, by in results.items()
        })

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    print("\n=== readouts x budget ===")
    for frames in BUDGETS:
        for exam in ("attested", "fixed"):
            for ro in ("mi_balanced", "ov_balanced"):
                print(ensemble(
                    lambda s: results[frames][s][exam][ro], SEEDS,
                    label=f"zipf-{frames} {exam:8s} {ro}"))

    print("\n=== registered bars ===")
    r = ensemble(lambda s: results[200][s]["attested"]["ov_balanced"],
                 SEEDS, label="R  zipf-200 overlap (E15 ref 0.700)")
    print(r)
    print(ensemble(lambda s: results[400][s]["attested"]["ov_balanced"],
                   SEEDS, label="V  zipf-400 overlap attested (>= 0.75)"))
    print(ensemble(lambda s: results[400][s]["fixed"]["ov_balanced"],
                   SEEDS, label="V' zipf-400 overlap FIXED exam"))

    print("\n=== guards ===")
    for frames in BUDGETS:
        print(ensemble(lambda s: results[frames][s]["tense_balanced"],
                       SEEDS, label=f"tense zipf-{frames}"))
        roles = [results[frames][s]["guards"] for s in SEEDS]
        ok = sum(x["roles_ok"] for x in roles)
        tot = sum(x["roles_total"] for x in roles)
        npl = results[frames][SEEDS[0]]["attested_pl_n"]
        exp = results[frames][SEEDS[0]]["total_attested_exposure"]
        print(f"zipf-{frames}: roles {ok}/{tot} | attested PL n={npl} "
              f"total exp={exp}")

    print("\n=== zipf-400 exposure -> accuracy (overlap readout, "
          "pooled) ===")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for s in SEEDS:
        for it in results[400][s]["attested"]["items"]:
            a = agg[it["form"]]
            a[0] += it["exposure"]
            a[1] += it["ov_correct"]
            a[2] += 1
    for form, (e, ok, n) in sorted(agg.items(),
                                   key=lambda kv: -kv[1][0])[:20]:
        print(f"  {form:10s} exp/seed={e / n:5.1f}  acc={ok / n:.2f}")


if __name__ == "__main__":
    sys.exit(main())
