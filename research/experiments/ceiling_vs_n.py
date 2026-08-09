"""E5: does the 0.71 ceiling move with n? Crowding vs image structure.

PRE-REGISTERED (task #134). After E3/E4, gain-as-parameterized is
exhausted and the composed ceiling sits at 0.710 (sqrt-BOTH, n=10). Two
suspects remain, and substrate size discriminates them:

  * k-WTA CROWDING: the SG and PL images compete for k=30 winners in one
    NUMBER area fed by asymmetric class mass. If this binds, more neurons
    (n=6000, same k) relieve it and the ceiling RISES.
  * IMAGE STRUCTURE: few distinct plural forms make a weak, narrow PL
    image regardless of substrate size. If this binds, the ceiling is a
    CORPUS property and stays ~flat at n=6000 -- pointing the program at
    corpus form-diversity, not substrate parameters.

CONFIGS: OFF, SCALED, BOTH (sqrt gain, cap 4 -- E2's form, the best point
estimate) at n=6000, seeds 42..51. Reference points are the n=3000
ensembles already measured (E3/E4 runs, same seeds, same protocol).

REGISTERED READINGS:
  C1 "moves with n": a config's n=6000 mean exceeds its n=3000 mean by
     more than the larger of the two CIs -> crowding is implicated.
  C2 "flat": within CI -> image structure; next unit raises DISTINCT
     plural forms in the corpus (a diversity change, not a rate change --
     rates stay natural).
  C3 guards exact at n=6000 (roles 9/9, C1 identical).
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

SEEDS = list(range(42, 52))
CONFIGS = ["OFF", "SCALED", "BOTH"]
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("CN_N", "6000"))

N3000_REFERENCE = {  # measured, E3/E4 logs, same seeds/protocol at n=3000
    "OFF": (0.5300, 0.0384),
    "SCALED": (0.6500, 0.0584),
    "BOTH": (0.7100, 0.0554),
}

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "ceiling_vs_n_results.json")

from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets, score_recall  # noqa: E402

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
]


def _flags(config: str) -> dict:
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )

    scaled = frozenset({TENSE, NUMBER})
    if config == "OFF":
        return {}
    if config == "SCALED":
        return {"synaptic_scaling": scaled}
    return {"synaptic_scaling": scaled, "novelty_gain_max": 4.0}


def run_cell(config: str, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from _parallel import forked_parser

    flags = _flags(config)

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
        ss = flags.get("synaptic_scaling")
        if ss:
            p.brain._engine.synaptic_scaling = ss
            p.brain._synaptic_scaling = ss
        p.novelty_gain_max = float(flags.get("novelty_gain_max", 1.0))

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"cn-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)
    sets_ = attested_morph_sets(parser)
    number = score_recall(parser.recall_number,
                          {k: sets_[k] for k in ("SG", "PL")})
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})
    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want
    print(f"[{config} seed={seed}] num bal={number['_balanced']} "
          f"(PL={number['PL']['acc']}) | tense bal={tense['_balanced']} | "
          f"roles {g_ok}/{g_total}", flush=True)
    return {"number": number, "tense": tense,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell, [(c, s) for c in CONFIGS for s in SEEDS])
    results = defaultdict(dict)
    for (config, seed), res in cell_results.items():
        results[config][seed] = res

    with open(OUT_PATH, "w") as f:
        json.dump({c: {str(s): v for s, v in by.items()}
                   for c, by in results.items()}, f, indent=2)

    from neural_assemblies.diagnostics import ensemble

    print("\n=== registered readings (n=6000 vs n=3000 reference) ===")
    for config in CONFIGS:
        e = ensemble(lambda s: results[config][s]["number"]["_balanced"],
                     SEEDS, label=f"number:{config}:n{N}")
        ref_mean, ref_ci = N3000_REFERENCE[config]
        margin = max(e.ci, ref_ci)
        verdict = ("MOVES (crowding implicated)"
                   if e.mean - ref_mean > margin else
                   ("DROPS" if ref_mean - e.mean > margin else
                    "flat (image structure implicated)"))
        print(f"{e}  vs n=3000 {ref_mean:.4f}+/-{ref_ci:.4f} -> {verdict}")


if __name__ == "__main__":
    sys.exit(main())
