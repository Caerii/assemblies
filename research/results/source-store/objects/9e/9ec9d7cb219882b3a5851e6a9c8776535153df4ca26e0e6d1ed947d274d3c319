"""E18: drift vs crowding -- why does a value area degrade as its OWN episodes grow?

PRE-REGISTERED (task #147), before any instrumented cell trained.

E17: on the clean substrate the budget curve is non-monotone (0.528@50
-> 0.700@200 -> 0.570@400; fixed exam agrees; 'men' at 14 episodes
reads 0.60). The split closed the MERGING channel; a second channel
lives inside a single value area (PL: 49 -> 107 episodes, worse at
more). Two suspects, each with its own discriminating measurement:

  DRIFT     The label-stimulus image is re-drawn against weights that
            keep changing, so words trained EARLY bound to an image
            that no longer exists at readout. Predicts: (D1) accuracy
            rises with a form's LAST-training position; (D2) the image
            moves during the phase, and moves MORE at 400 than at 200.
  CROWDING  Recruitment grows with episodes; k-WTA over a larger
            materialized pool dilutes probe and image. Predicts: (C1)
            value-area ever-fired counts grow strongly 200 -> 400,
            with accuracy declining even for LATE-trained forms.

ARMS (both zipf, split, slow scaling -- the E17 configuration):
  A "record"  training UNCHANGED (bit-path identical to E17 given the
              same seed); a wrapper replicates train_number's teacher
              loop in Python to log the PL episode ORDER (observation
              only -- it touches no brain state before delegating).
  B "chunk"   train_number applied in 8 chunks with the PL/SG label
              images SNAPSHOTTED between chunks (compact winners;
              compact indices are stable once materialized, so
              cross-time overlap is meaningful). CONFOUND, stated: the
              deferred-scaling flush runs per chunk instead of once;
              arm B's final accuracy is reported against arm A's as
              the comparability check, and B's image TRAJECTORY is
              evidence only if that check passes (within 0.05).

CELLS: {A, B} x FRAMES {200, 400} x seeds 42-46 (20 cells).

REGISTERED BARS:
  D1 per-seed Spearman(last-episode position, overlap-correct) over
     attested PL forms at 400 in arm A: positive, ensemble CI
     excluding zero.
  D2 arm B: total drift = 1 - overlap(first image, final image) is
     GREATER at 400 than at 200 (paired per seed, CI excluding zero),
     with the trajectory reported (per-chunk overlap to final).
  C1 ever-fired counts of NUMBER_PL/SG at 200 vs 400 reported; the
     crowding reading requires BOTH strong growth AND D1 flat (decline
     independent of training position).
DECISION RULE: D1+D2 -> drift is the channel; E19 = phase-boundary
label RE-BINDING (re-project the label stimulus once at flush time so
the image words bound to is the image read out -- one projection, and
it rhymes with deferred scaling's timescale argument). D1 flat + C1
strong -> crowding; the lever is value-area n (or k) and the fix is a
sweep, not a mechanism. Both null -> the channel is neither and the
honest state is open, with both instruments' data attached.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_checkpoint_document

SEEDS = list(range(42, 47))
ARMS = ("record", "chunk")
BUDGETS = (200, 400)
CHUNKS = 8
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("DC_N", "3000"))
if os.environ.get("DC_SMOKE") == "1":
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "drift_vs_crowding_results.json")

from overlap_ceiling import spearman  # noqa: E402


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
        .morph_features import attested_morph_sets, feature_images_compact
    from _parallel import forked_parser

    area_pl = feature_value_area(NUMBER, "PL")
    area_sg = feature_value_area(NUMBER, "SG")

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

    episode_log: list = []
    snapshots: list = []

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

        orig = p.train_number

        def record_order(sentences):
            # Observation only: replicate the teacher loop's filters to
            # log PL episode order, then delegate untouched.
            i = 0
            for sent in sentences:
                for word in sent:
                    if word not in p.stim_map:
                        continue
                    g = p.word_grounding.get(word)
                    if g is None or g.dominant_modality not in (
                            "visual", "motor"):
                        continue
                    episode_log.append((i, word, p.detect_number(word)))
                    i += 1
            return orig(sentences)

        def chunked(sentences):
            step = max(1, math.ceil(len(sentences) / CHUNKS))
            for j in range(0, len(sentences), step):
                orig(sentences[j:j + step])
                if "number_PL" in p.brain.stimuli:
                    img = feature_images_compact(
                        p, area_pl, {"PL": "number_PL"})
                    snapshots.append([int(c) for c in img.get("PL", [])])
            return None

        p.train_number = record_order if arm == "record" else chunked

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"zs-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)

    sets_ = attested_morph_sets(parser)
    exposure = getattr(parser, "_morph_exposure", {})

    # Per-form accuracy (both readouts) + last-episode position (arm A).
    pl_eps = [(i, w) for i, w, num in episode_log if num == "PL"]
    total_eps = episode_log[-1][0] + 1 if episode_log else 0
    items = []
    for form in sets_["PL"]:
        got, diag = parser.recall_number(form)
        last = max((i for i, w in pl_eps if w == form), default=None)
        items.append({
            "form": form,
            "exposure": exposure.get(f"NUMBER:{form}", 0),
            "last_pos": (last / total_eps
                         if last is not None and total_eps else None),
            "mi_correct": bool(got == "PL"),
            "ov_correct": bool(diag.get("overlap_answer") == "PL"),
        })

    # Drift trajectory (arm B): per-chunk image overlap with the FINAL.
    def ov(a, b):
        a, b = set(a), set(b)
        return len(a & b) / max(1, min(len(a), len(b)))

    drift = None
    if snapshots:
        final = snapshots[-1]
        drift = {
            "to_final": [round(ov(s, final), 4) for s in snapshots],
            "consecutive": [round(ov(a, b), 4) for a, b in
                            zip(snapshots, snapshots[1:])],
            "total_drift": round(1 - ov(snapshots[0], final), 4),
        }

    w_pl = parser.brain.areas[area_pl].get_num_ever_fired()
    w_sg = parser.brain.areas[area_sg].get_num_ever_fired()
    ov_bal_pl = float(np.mean([it["ov_correct"] for it in items]))
    print(f"[E18 {arm} zipf-{frames} seed={seed}] "
          f"PL ov acc={ov_bal_pl:.3f} (n={len(items)}) | "
          f"w PL={w_pl} SG={w_sg} | "
          f"drift={drift['total_drift'] if drift else '-'} | "
          f"eps={total_eps}", flush=True)
    return {"items": items, "drift": drift,
            "w_pl": int(w_pl), "w_sg": int(w_sg),
            "total_episodes": total_eps,
            "pl_ov_acc": ov_bal_pl}


def cell_key(arm, frames):
    return f"{arm}-{frames}"


def main():
    if os.environ.get("DC_ANALYZE") == "1":
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

    print("\n=== registered bars ===")

    def d1(s, frames):
        its = [it for it in results[f"record-{frames}"][s]["items"]
               if it["last_pos"] is not None]
        return spearman([it["last_pos"] for it in its],
                        [1.0 if it["ov_correct"] else 0.0 for it in its])

    for frames in BUDGETS:
        print(ensemble(lambda s: d1(s, frames), SEEDS,
                       label=f"D1 rho(last position, correct) @{frames}"))
    print(ensemble(
        lambda s: (results["chunk-400"][s]["drift"]["total_drift"]
                   - results["chunk-200"][s]["drift"]["total_drift"]),
        SEEDS, label="D2 total drift 400 - 200 (bar > 0)"))
    for frames in BUDGETS:
        print(ensemble(
            lambda s: results[f"chunk-{frames}"][s]["drift"]["total_drift"],
            SEEDS, label=f"   total drift @{frames}"))
    for frames in BUDGETS:
        print(ensemble(
            lambda s: results[f"record-{frames}"][s]["w_pl"], SEEDS,
            label=f"C1 NUMBER_PL ever-fired @{frames}"))
        print(ensemble(
            lambda s: results[f"record-{frames}"][s]["w_sg"], SEEDS,
            label=f"C1 NUMBER_SG ever-fired @{frames}"))

    print("\n=== comparability check (chunk-arm accuracy vs record) ===")
    for frames in BUDGETS:
        for arm in ARMS:
            print(ensemble(
                lambda s: results[f"{arm}-{frames}"][s]["pl_ov_acc"],
                SEEDS, label=f"PL overlap acc {arm}-{frames}"))

    print("\n=== drift trajectory (chunk-400, seed mean, "
          "overlap to FINAL image) ===")
    traj = np.mean([results["chunk-400"][s]["drift"]["to_final"]
                    for s in SEEDS], axis=0)
    print("  " + " ".join(f"{x:.2f}" for x in traj))


if __name__ == "__main__":
    sys.exit(main())
