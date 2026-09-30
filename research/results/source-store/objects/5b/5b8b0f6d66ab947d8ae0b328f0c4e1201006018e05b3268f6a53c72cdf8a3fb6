"""E6: form diversity vs the image-structure ceiling.

PRE-REGISTERED (task #135), after the manipulated-variable census and
BEFORE any recall measurement on the new corpus.

E5 (#134) localized the 0.71 ceiling to IMAGE STRUCTURE: ~10 distinct
plural forms build a weak PL image at any substrate size. E6 changes the
CORPUS, not a mechanism or a rate:

  * coverage-biased subject sampling (least-used-first) -- uniform-with-
    replacement under-covered the vocabulary vs natural long-tail text;
  * OBJECT number now varies at the same PLURAL_RATE -- objects were the
    last always-singular NP slot, a censused-constant axis carrying no
    information (the e909c61 doctrine applied to its own gap). Passives
    skip plural-object frames (the lexicon's aux covers is/was only).

MANIPULATED-VARIABLE CHECK, done before this run (the E4 lesson): the
census moved 10 -> 16 distinct attested plurals at n=1500, and a latent
ANALYZER defect surfaced on the way (number-ambiguous "fish" -- the test
licensed only plural agreement for a legally singular subject; fixed in
test_corpus_grammaticality, all 10 tests green).

CONFIGS: OFF / SCALED / BOTH(sqrt,4) on the DIVERSE corpus, n=3000,
seeds 42..51. Reference: the same configs on the uniform corpus,
measured in E3/E4 (0.530 / 0.650 / 0.710 +/- CIs).

REGISTERED BARS:
  D1 every cell attests >= 13 distinct PL forms (manipulation reached
     the training corpus; below that the diversity change failed to
     survive the pipeline and nothing else is interpretable).
  D2 SCALED or BOTH reaches mean number balanced >= 0.75 on the diverse
     corpus (the bar three mechanism experiments could not clear).
  D3 tense balanced within +/-0.05 of its uniform-corpus reference per
     config (diversity must not silently trade tense away).
  D4 guards exact (roles, C1).

DECISION RULE: D2 passes -> the ceiling WAS image structure, confirmed
causally, and the forcing-rate retirement A/B finally opens. D2 fails
with D1 passing -> diversity at this scale is insufficient; the next
axis is corpus SIZE (more frames per stage), which is a bigger realism
question and gets its own registration.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_new_document

SEEDS = list(range(42, 52))
CONFIGS = ["OFF", "SCALED", "BOTH"]
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("DF_N", "3000"))

UNIFORM_REFERENCE = {  # measured, E3/E4, same seeds/protocol, uniform corpus
    "OFF": (0.5300, 0.0384),
    "SCALED": (0.6500, 0.0584),
    "BOTH": (0.7100, 0.0554),
}
TENSE_REFERENCE = {
    "OFF": (0.6778, 0.0458),
    "SCALED": (0.6415, 0.0460),
    "BOTH": (0.6937, 0.0338),
}

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "diverse_forms_recall_results.json")

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
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        generation as generation_mod,
    )
    from _parallel import forked_parser

    # THE DIVERSE CORPUS IS THE ARM, not the default (the default stays at
    # the measured production corpus until this experiment's verdict --
    # patched here in the WORKER, since spawned processes import fresh
    # defaults). Every config trains on the same diverse corpus; the
    # uniform reference is the measured E3/E4 ensembles.
    generation_mod.SUBJECT_SAMPLING = "coverage"
    generation_mod.OBJECT_PLURAL_RATE = generation_mod.PLURAL_RATE

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

    # Tag SHARED with E3/E5: pre-stages have complexity < 3, so the frame
    # loop (the only code the diversity arm changes) never runs there --
    # the pre-checkpoints are byte-identical across corpus arms.
    parser = forked_parser(f"gm-n{N}", seed, build_and_train_pre,
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
          f"(PL={number['PL']['acc']} n={number['PL']['n']}) | "
          f"tense bal={tense['_balanced']} | roles {g_ok}/{g_total}",
          flush=True)
    return {"number": number, "tense": tense,
            "guards": {"roles_ok": g_ok, "roles_total": g_total},
            "n_pl_forms": number["PL"]["n"]}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell, [(c, s) for c in CONFIGS for s in SEEDS])
    results = defaultdict(dict)
    for (config, seed), res in cell_results.items():
        results[config][seed] = res

    write_new_document(Path(OUT_PATH), {
        c: {str(s): v for s, v in by.items()}
        for c, by in results.items()
    })

    from neural_assemblies.diagnostics import ensemble

    print("\n=== registered bars (diverse corpus vs uniform reference) ===")
    min_pl = min(results[c][s]["n_pl_forms"] for c in CONFIGS for s in SEEDS)
    print(f"D1 min distinct PL forms per cell: {min_pl} (bar >= 13)")
    for config in CONFIGS:
        e = ensemble(lambda s: results[config][s]["number"]["_balanced"],
                     SEEDS, label=f"number:{config}:diverse")
        ref_mean, ref_ci = UNIFORM_REFERENCE[config]
        print(f"{e}  vs uniform {ref_mean:.4f}+/-{ref_ci:.4f}")
        et = ensemble(lambda s: results[config][s]["tense"]["_balanced"],
                      SEEDS, label=f"tense:{config}:diverse")
        tref, _ = TENSE_REFERENCE[config]
        print(f"  {et}  vs uniform {tref:.4f} (D3 |delta| <= 0.05)")


if __name__ == "__main__":
    sys.exit(main())
