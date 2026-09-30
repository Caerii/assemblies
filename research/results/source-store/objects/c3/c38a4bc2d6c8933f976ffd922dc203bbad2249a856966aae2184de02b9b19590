"""What did the corpus variation BUY? Tense/number recall from the substrate.

PRE-REGISTERED before first run (task #129).

BACKGROUND. The realism overhaul (e909c61) made tense (~30% past) and subject
number (~30% plural) vary in the generated corpus, on the argument that a cue
which never varies carries no information. That argument says the OLD corpus
could not teach these features; it does not show the NEW corpus does. This
experiment closes the loop, and its census half is already decided by code
reading:

  * "tense" phase: LIVE in SENTENCES (schedule.py), but one-class until the
    overhaul -- train_tense saw only PRESENT.
  * "number" phase: DORMANT -- schedulable but listed by NO stage (the
    `phrases` dormant-selector shape), AND its teacher `detect_number` read
    "SG" for every corpus token (plural surfaces inherit the lemma's
    grounding verbatim; nothing ever set a PL feature). Both fixed this
    session: the phase is scheduled and the teacher resolves plurality from
    lexicon forms, mirroring `lookup_verb_form`.
  * Neither feature had a READOUT: detect_* is a Python teacher, train_*
    writes core->feature associations, and nothing read them back.
    `recall_tense`/`recall_number` (reconstruction-style argmax over
    stimulus images) are the new readout under measurement here.

WHAT IS FORM-LEVEL AND WHAT IS NOT. train_tense/train_number associate the
surface FORM's assembly with a feature stimulus. What is recallable is
therefore a lexical association ("chased" -> PAST, "dogs" -> PL), not
anything about the sentence; sentence-level tense (aux + participle) lives in
the Python detector the substrate never sees. The claim under test is the
substrate one: CAN the feature areas store and return the two-class contrast
the varied corpus now presents, across ~100+ forms sharing one area?

ARMS (paired by BRAIN seed; 5 seeds):
  DEFAULT      current pipeline: varied corpus, tense+number phases live.
  ABLATE       identical corpus (same rates, same seed), but the "tense" and
               "number" phases are removed from SENTENCES. Feature stimuli
               are then registered POST HOC so the readout can pose the same
               question to an untrained pathway. This is the honest null:
               a corpus-level null (PAST_RATE=0) empties the test set instead
               of nulling the mechanism, and an unregistered-stimulus null
               makes recall mechanically return None (images < 2). Memory:
               "beta=0 is not a null" -- the null must materialize the same
               objects.
  NO_VARIATION PAST_RATE=0, PLURAL_RATE=0: documents that the one-class
               corpus cannot even POSE the question (test sets empty).

TEST SETS (corpus-attested by construction: inflected surfaces only enter
stim_map via _register_surface_forms when they occur in a sentence), built by
scanning the RAW lexicon data (NOUNS/VERBS), not `lookup_lexicon_entry` --
the lexicon index keeps only the FIRST entry per surface (nouns before
verbs), so a homograph like "loves" resolves to the noun and its verbhood is
invisible. Found on the first instrumented run, not anticipated:
  * NOUN/VERB HOMOGRAPHS: "answers/hopes/loves/fears/surprises" are 3sg verb
    forms in the corpus AND noun-plural forms in the lexicon. Undecidable at
    form level; EXCLUDED from every class (and note the same collision
    biases `detect_number`'s teacher and hides these forms' verbhood from
    `lookup_verb_form`).
  * ZERO-DERIVATION PASTS: "put/let/cut/read" have past == lemma; the lemma
    registers unconditionally and trains as present. EXCLUDED (w == lemma),
    mirroring the PRESENT-class exclusion.
  * "fish" is its own plural; excluded by the same w != lemma rule.
  tense:  PAST  = attested forms == some verb's forms["past"], form != lemma,
                  not noun-ambiguous
          PRES  = attested forms == some verb's 3sg form, form != lemma,
                  not noun-ambiguous
  number: PL    = attested forms == some noun's forms["plural"], != lemma,
                  not verb-ambiguous
          SG    = the LEMMAS of exactly those nouns, not verb-ambiguous

METRICS per arm x seed: per-class accuracy, balanced accuracy, tie rate
(recall returned None -- a tie is a failure to read), mean margin, and
image_separation (overlap between the two label images; near 1.0 means the
feature area cannot answer and accuracy numbers are untrustworthy).

REGISTERED PREDICTIONS:
  P1 DEFAULT tense balanced accuracy >= 0.75 with image_separation < 0.5.
  P2 ABLATE is tie-dominated or near chance on the SAME items (paired):
     if ABLATE matches DEFAULT, the readout is measuring registration
     artifacts, not learned association -- the fake-perfect guard.
  P3 DEFAULT number balanced accuracy >= 0.75 (the phase this session woke).
  P4 NO_VARIATION: n_past == 0 and n_plural == 0 -- variation is what makes
     the question posable at all.

Failure of P1/P3 with healthy image_separation is a real negative: the
variation is decoration the substrate cannot read, and the note should say
so.

ADDENDUM, after the smoke run and BEFORE the scored run: P4 is already
REFUTED as written. At PAST_RATE=0 the corpus still attests past-form
surfaces because (a) PASSIVES inject regular participles ("is chased" --
ppart == past for regular verbs) at PASSIVE_EVERY regardless of PAST_RATE,
and (b) hand-authored curriculum sentences carry "was". The prediction's
spirit ("the generator's tense axis is off") holds; its letter ("n_past ==
0") did not survive contact with the OTHER variation sources. Scored-run
expectation for NO_VARIATION is therefore: small residual PAST class,
passive-participle-only.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from research.json_documents import write_new_document

SEEDS = [42, 43, 44, 45, 46]
ARMS = ("DEFAULT", "ABLATE", "NO_VARIATION")
STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES")
N = int(os.environ.get("VB_N", "3000"))

# Smoke mode: one seed, small n -- API-crash detection only, numbers not
# interpretable. The real run uses the defaults above.
if os.environ.get("VB_SMOKE") == "1":
    SEEDS = [42]
    N = int(os.environ.get("VB_N", "1500"))

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "what_variation_buys_results.json")


def build_parser(seed: int):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset

    return EmergentParser(n=N, k=30, seed=seed,
                          vocabulary=build_vocabulary_preset("core"),
                          fast_training=True)


def train(parser, arm: str):
    # Rates live in curriculum/generation.py (the corpus-generation
    # concern); stage phase lists live in trainer.py (orchestration).
    # Each arm patches the module that OWNS its axis.
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        trainer as trainer_mod,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        generation as generation_mod,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )

    saved_rates = (generation_mod.PAST_RATE, generation_mod.PLURAL_RATE)
    saved_phases = {s: list(trainer_mod._STAGE_CONFIG[s]["phases"])
                    for s in trainer_mod._STAGE_CONFIG}
    try:
        if arm == "NO_VARIATION":
            generation_mod.PAST_RATE = 0.0
            generation_mod.PLURAL_RATE = 0.0
        elif arm == "ABLATE":
            for s in trainer_mod._STAGE_CONFIG:
                trainer_mod._STAGE_CONFIG[s]["phases"] = [
                    p for p in saved_phases[s]
                    if p not in ("tense", "number")
                ]
        ct = CurriculumTrainer(parser)
        for stage in STAGES:
            ct.train_stage(stage)
    finally:
        generation_mod.PAST_RATE, generation_mod.PLURAL_RATE = saved_rates
        for s, ph in saved_phases.items():
            trainer_mod._STAGE_CONFIG[s]["phases"] = ph
    return parser


def ensure_stims(parser):
    """ABLATE arm: register feature stimuli so recall can pose the question
    to the untrained pathway (see ARMS in the module docstring)."""
    for name in ("tense_PRESENT", "tense_PAST", "number_SG", "number_PL"):
        if name not in parser.brain.stimuli:
            parser.brain.add_stimulus(name, parser.k)


# Test-set construction and scoring were PROMOTED to the package after this
# experiment ran (evaluation/morph_features.py holds the canonical copy and
# the documented exclusion classes); these names remain the experiment's
# protocol vocabulary.
from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets as test_sets  # noqa: E402
from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import score_recall as score  # noqa: E402


def main():
    results = defaultdict(dict)
    for arm in ARMS:
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            parser = build_parser(seed)
            train(parser, arm)
            if arm == "ABLATE":
                ensure_stims(parser)
            sets = test_sets(parser)
            tense = score(parser.recall_tense,
                          {k: sets[k] for k in ("PRESENT", "PAST")})
            number = score(parser.recall_number,
                           {k: sets[k] for k in ("SG", "PL")})
            results[arm][seed] = {"tense": tense, "number": number}
            print(f"[{arm} seed={seed}] "
                  f"tense bal={tense['_balanced']} "
                  f"(PAST n={tense['PAST']['n']} acc={tense['PAST']['acc']}, "
                  f"PRES n={tense['PRESENT']['n']} "
                  f"acc={tense['PRESENT']['acc']}, "
                  f"sep={tense['_image_separation']}) | "
                  f"number bal={number['_balanced']} "
                  f"(PL n={number['PL']['n']} acc={number['PL']['acc']}, "
                  f"SG n={number['SG']['n']} acc={number['SG']['acc']}, "
                  f"sep={number['_image_separation']})",
                  flush=True)

    write_new_document(Path(OUT_PATH), {
        a: {str(s): v for s, v in by.items()}
        for a, by in results.items()
    })
    print(f"\nwrote {OUT_PATH}")

    # Registered-prediction summary. Seed summaries go through
    # `diagnostics.ensemble` (mean +/- 95% CI, refuses < 3 seeds) -- the
    # in-file np.mean sites average over ITEMS within one run, which is a
    # different quantity and stays as-is.
    from neural_assemblies.diagnostics import ensemble

    def arm_ens(arm, feat):
        if any(results[arm][s][feat]["_balanced"] is None for s in SEEDS):
            return None
        return ensemble(
            lambda s: results[arm][s][feat]["_balanced"], SEEDS,
            label=f"{arm}:{feat}")

    print("\n=== registered predictions ===")
    print(f"P1 DEFAULT tense balanced (>=0.75?): "
          f"{arm_ens('DEFAULT', 'tense')}")
    print(f"P2 ABLATE tense balanced (chance/ties?): "
          f"{arm_ens('ABLATE', 'tense')}")
    print(f"P3 DEFAULT number balanced (>=0.75?): "
          f"{arm_ens('DEFAULT', 'number')}")
    nv = [results['NO_VARIATION'][s]['tense']['PAST']['n'] +
          results['NO_VARIATION'][s]['number']['PL']['n'] for s in SEEDS]
    print(f"P4 NO_VARIATION attested past+plural forms (0?): {nv}")


if __name__ == "__main__":
    sys.exit(main())
