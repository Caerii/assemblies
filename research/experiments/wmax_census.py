"""E10: w_max pile-up per class -- pure measurement, last suspect standing.

PRE-REGISTERED (task #139), arithmetic done BEFORE running (the E3/E4
lesson): with beta=0.10, a frequent SG form (~5 sightings/stage x R4 x ~2
rounds ~= 40 writes) grows (1.1)^40 ~= 45 > w_max=20; a rare PL form
(~12 writes) reaches ~3.1. So the cap should bind on the FREQUENT class
-- inverting the naive suspect. If SG-image weights pile at cap under
slow scaling + repetition, the mechanism harming discrimination would be
UNIFORMIZATION: capped weights are all equal, so after normalization the
SG image's internal structure flattens. If NEITHER class piles up, the
saturation suspect dies and scaled-number's 0.65 ceiling is declared
unexplained, with these histograms as the next hypothesis source.

CELLS: {SLOW-SCALED, OFF} x R in {1, 4} x seeds {42, 43, 44} (12 cells,
forked pre-stages, ~10 min). CENSUS per (fiber, class image): over the
image columns' nonzero weights -- mean, max, fraction >= 0.9 * cap.
Images taken as COMPACT winners (area.winners immediately after the
label-stimulus projection, same index space as the weight matrix columns
-- the two-index-space trap avoided by construction).

REGISTERED READINGS:
  W1 SLOW-R4 SG columns show cap fraction >> PL columns' (the
     arithmetic's prediction).
  W2 THE CEILING LINK: pile-up only matters if it distinguishes the
     saturating configs from the non-saturating ones in a way that
     tracks the measured ceilings. Reported as the full table; no
     single-number bar -- this is instrumentation, not an A/B.
"""
from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_new_document

SEEDS = [42, 43, 44]
CONFIGS = ["OFF", "SLOW"]
REPS = (1, 4)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("WC_N", "3000"))

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "wmax_census_results.json")

FIBERS = {  # feature area -> (source core, stim prefix, labels)
    "NUMBER": ("NOUN_CORE", "number_", ("SG", "PL")),
    "TENSE": ("VERB_CORE", "tense_", ("PRESENT", "PAST")),
}


def census_cell(config: str, reps: int, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
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
        if config == "SLOW":
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
    cap = 0.9 * eng.w_max
    out: dict = {}
    with brain.read_only():
        for feat, (src, prefix, labels) in FIBERS.items():
            conn = eng._area_conns.get(src, {}).get(feat)
            w = getattr(conn, "weights", None) if conn is not None else None
            if w is None or getattr(w, "ndim", 0) != 2:
                continue
            rows = min(int(eng._areas[src].w), int(w.shape[0]))
            for label in labels:
                stim = prefix + label
                if stim not in brain.stimuli:
                    continue
                brain.inhibit_areas([feat])
                brain.project({stim: [feat]}, {})
                cols = [int(c) for c in brain.areas[feat].winners
                        if int(c) < w.shape[1]]
                if not cols:
                    continue
                sub = np.asarray(w[:rows, cols])
                nz = sub[sub > 0]
                out[f"{feat}:{label}"] = {
                    "n_weights": int(nz.size),
                    "mean": float(nz.mean()) if nz.size else None,
                    "max": float(nz.max()) if nz.size else None,
                    "frac_at_cap": (float((nz >= cap).mean())
                                    if nz.size else None),
                }
    return out


def main():
    from _parallel import run_cells

    cells = [(c, r, s) for c in CONFIGS for r in REPS for s in SEEDS]
    cell_results = run_cells(census_cell, cells)
    write_new_document(Path(OUT_PATH), {
        f"{c}:R{r}:s{s}": v
        for (c, r, s), v in cell_results.items()
    })

    print("\n=== w_max census (fraction of image-column weights >= 0.9*cap) ===")
    agg: dict = defaultdict(list)
    for (c, r, _s), v in cell_results.items():
        for key, stats in v.items():
            if stats["frac_at_cap"] is not None:
                agg[(c, r, key)].append(
                    (stats["frac_at_cap"], stats["max"], stats["mean"]))
    for (c, r, key), vals in sorted(agg.items()):
        fr = np.mean([x[0] for x in vals])
        mx = np.mean([x[1] for x in vals])
        mn = np.mean([x[2] for x in vals])
        print(f"{c}:R{r} {key:16s} frac_at_cap={fr:.3f} "
              f"max={mx:6.2f} mean={mn:5.2f}")


if __name__ == "__main__":
    sys.exit(main())
