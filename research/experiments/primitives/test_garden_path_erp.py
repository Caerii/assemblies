"""
Garden-Path ERP: Structural Ambiguity and Reanalysis

Tests whether the prediction+binding mechanism produces garden-path effects
when structural ambiguity forces reanalysis at a disambiguation point.

Phenomenon:
  "dog chases cat sees bird"  (garden-path, no complementizer)
  "dog that chases cat sees bird"  (unambiguous control)

  Without "that", the system initially parses "dog chases cat" as a complete
  SVO sentence. When "sees" arrives, it violates the prediction (expected PP
  or sentence-end) and requires structural reanalysis — "dog chases cat" was
  actually a reduced relative clause modifying "dog".

  The unambiguous version with "that" explicitly marks the relative clause,
  so the system expects a main verb after the embedded clause.

Architecture: Same 11-area RECURSIVE_VOCAB as center-embedding experiments.
  Training includes SVO, SVO+PP, and SRC sentences via RecursiveCFG.

Hypotheses:
  H1 (Prediction violation): N400 at the second verb is higher for the
      garden-path condition than the unambiguous control. The system
      predicted PP or sentence-end, not another verb.
  H2 (Reanalysis cost): P600 at the main patient position is higher for
      the garden-path condition. The binding must be established in a
      context where prior commitments conflict.
  H3 (Baseline intact): The object position (word 3) shows normal double
      dissociation, confirming the training worked.

Usage:
    uv run python research/experiments/primitives/test_garden_path_erp.py
    uv run python research/experiments/primitives/test_garden_path_erp.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
)
from research.experiments.lib.vocabulary import RECURSIVE_VOCAB
from research.experiments.lib.grammar import RecursiveCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400, measure_p600


@dataclass
class GardenPathConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    # Grammar
    pp_prob: float = 0.4
    rel_prob: float = 0.4
    max_pp_depth: int = 2
    # Training
    n_train_sentences: int = 80
    training_reps: int = 3
    # Test
    n_test_items: int = 5


def run_trial(
    cfg: GardenPathConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one garden-path trial.

    Trains on SVO/SVO+PP/SRC sentences, then compares:
      - Unambiguous: "dog that chases cat SEES bird"
      - Garden-path: "dog chases cat SEES bird"
    at the second verb position (disambiguation point).
    """
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Generate and train
    grammar = RecursiveCFG(
        pp_prob=cfg.pp_prob,
        rel_prob=cfg.rel_prob,
        max_pp_depth=cfg.max_pp_depth,
        vocab=vocab, rng=rng)

    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = grammar.generate_batch(n_train)

    n_rel = sum(1 for s in train_sents if s.get("has_rel"))
    n_pp = sum(1 for s in train_sents if s.get("has_pp"))

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    novels = list(vocab.novel_words.keys())

    # ── Test 1: Baseline — object position double dissociation ─────
    base_n400_gram, base_n400_cv = [], []
    base_p600_gram, base_p600_cv = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        base_n400_gram.append(measure_n400(predicted, lexicon[gram_obj]))
        base_n400_cv.append(measure_n400(predicted, lexicon[cv_obj]))
        base_p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        base_p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    # ── Test 2: Garden-path vs unambiguous at second verb ──────────
    gp_n400, unamb_n400 = [], []
    gp_p600_patient, unamb_p600_patient = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]
        main_patient = nouns[(i + 3) % len(nouns)]
        main_patient_cv = verbs[(i + 2) % len(verbs)]

        # --- Unambiguous: "agent that rel_verb rel_patient MAIN_VERB" ---
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        # Re-activate agent for main-clause prediction
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_unamb = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        unamb_n400.append(measure_n400(pred_unamb, lexicon[main_verb]))

        # Process main verb, then measure P600 at patient position
        activate_word(brain, main_verb, "VERB_CORE", 3)
        unamb_p600_patient.append(measure_p600(
            brain, main_patient, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

        # --- Garden-path: "agent rel_verb rel_patient MAIN_VERB" ---
        # (no "that" — system initially parses as SVO)
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        # After "rel_patient", system thinks SVO is complete.
        # Forward-project from NOUN_CORE — expects PP or nothing, not a verb.
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_gp = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        gp_n400.append(measure_n400(pred_gp, lexicon[main_verb]))

        # Process main verb in GP context, measure P600 at patient
        activate_word(brain, main_verb, "VERB_CORE", 3)
        gp_p600_patient.append(measure_p600(
            brain, main_patient, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

    # ── Test 3: N400 for different word types at GP position ───────
    # At the GP point, what does the system predict after apparent SVO?
    # Compare: expected prep vs unexpected verb vs expected noun (for PP)
    gp_n400_prep, gp_n400_verb, gp_n400_noun = [], [], []
    preps = vocab.words_for_category("PREP")
    locs = vocab.words_for_category("LOCATION")

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 1) % len(nouns)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_post_obj = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # What does the system expect after the object?
        gp_n400_prep.append(measure_n400(pred_post_obj,
                                         lexicon[preps[i % len(preps)]]))
        gp_n400_verb.append(measure_n400(pred_post_obj,
                                         lexicon[verbs[(i + 1) % len(verbs)]]))
        gp_n400_noun.append(measure_n400(pred_post_obj,
                                         lexicon[locs[i % len(locs)]]))

    brain.disable_plasticity = False

    return {
        "n_train": n_train, "n_rel": n_rel, "n_pp": n_pp,
        # Baseline
        "base_n400_gram": float(np.mean(base_n400_gram)),
        "base_n400_cv": float(np.mean(base_n400_cv)),
        "base_p600_gram": float(np.mean(base_p600_gram)),
        "base_p600_cv": float(np.mean(base_p600_cv)),
        # GP vs unambiguous: N400 at second verb
        "unamb_n400": float(np.mean(unamb_n400)),
        "gp_n400": float(np.mean(gp_n400)),
        # GP vs unambiguous: P600 at patient
        "unamb_p600_patient": float(np.mean(unamb_p600_patient)),
        "gp_p600_patient": float(np.mean(gp_p600_patient)),
        # Post-object predictions
        "gp_n400_prep": float(np.mean(gp_n400_prep)),
        "gp_n400_verb": float(np.mean(gp_n400_verb)),
        "gp_n400_noun": float(np.mean(gp_n400_noun)),
    }


class GardenPathERPExperiment(ExperimentBase):
    """Garden-path structural ambiguity experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="garden_path_erp",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[GardenPathConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or GardenPathConfig(
            **{k: v for k, v in kwargs.items()
               if k in GardenPathConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Garden-Path ERP: Structural Ambiguity + Reanalysis")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  pp_prob={cfg.pp_prob}, rel_prob={cfg.rel_prob}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = [
            "base_n400_gram", "base_n400_cv",
            "base_p600_gram", "base_p600_cv",
            "unamb_n400", "gp_n400",
            "unamb_p600_patient", "gp_p600_patient",
            "gp_n400_prep", "gp_n400_verb", "gp_n400_noun",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])

            if s == 0:
                self.log(f"    Training: {result['n_train']} sentences"
                         f" ({result['n_rel']} rel, {result['n_pp']} PP)")

        # ── Report: Baseline ──
        t_base_p600 = paired_ttest(vals["base_p600_cv"], vals["base_p600_gram"])
        self.log(f"\n  === Baseline Object Position ===")
        self.log(f"    N400 — gram: {np.mean(vals['base_n400_gram']):.3f}"
                 f"  cv: {np.mean(vals['base_n400_cv']):.3f}")
        self.log(f"    P600 — gram: {np.mean(vals['base_p600_gram']):.3f}"
                 f"  cv: {np.mean(vals['base_p600_cv']):.3f}"
                 f"  d={t_base_p600['d']:.2f}")

        # ── Report: Garden-path vs unambiguous ──
        t_n400_gp = paired_ttest(vals["gp_n400"], vals["unamb_n400"])
        t_p600_gp = paired_ttest(vals["gp_p600_patient"],
                                 vals["unamb_p600_patient"])

        self.log(f"\n  === Garden-Path vs Unambiguous ===")
        self.log(f"    N400 at second verb:")
        self.log(f"      Unambiguous: {np.mean(vals['unamb_n400']):.3f}"
                 f"  Garden-path: {np.mean(vals['gp_n400']):.3f}"
                 f"  d={t_n400_gp['d']:.2f}")
        self.log(f"    P600 at main patient:")
        self.log(f"      Unambiguous: {np.mean(vals['unamb_p600_patient']):.3f}"
                 f"  Garden-path: {np.mean(vals['gp_p600_patient']):.3f}"
                 f"  d={t_p600_gp['d']:.2f}")

        # ── Report: Post-object predictions ──
        self.log(f"\n  === Post-Object Predictions ===")
        self.log(f"    After apparent SVO, N400 for:")
        self.log(f"      Preposition: {np.mean(vals['gp_n400_prep']):.3f}"
                 f"  (trained continuation)")
        self.log(f"      Verb:        {np.mean(vals['gp_n400_verb']):.3f}"
                 f"  (garden-path trigger)")
        self.log(f"      Noun:        {np.mean(vals['gp_n400_noun']):.3f}"
                 f"  (unexpected category)")

        # Hypotheses
        gp_higher_n400 = np.mean(vals["gp_n400"]) > np.mean(vals["unamb_n400"])
        gp_higher_p600 = (np.mean(vals["gp_p600_patient"])
                          > np.mean(vals["unamb_p600_patient"]))
        baseline_ok = t_base_p600["d"] > 1.0

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (GP > Unamb N400): "
                 f"{'PASS' if gp_higher_n400 else 'FAIL'}"
                 f" (d={t_n400_gp['d']:.2f})")
        self.log(f"    H2 (GP > Unamb P600): "
                 f"{'PASS' if gp_higher_p600 else 'FAIL'}"
                 f" (d={t_p600_gp['d']:.2f})")
        self.log(f"    H3 (Baseline intact): "
                 f"{'PASS' if baseline_ok else 'FAIL'}"
                 f" (d={t_base_p600['d']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "baseline": {
                "n400_gram": summarize(vals["base_n400_gram"]),
                "n400_cv": summarize(vals["base_n400_cv"]),
                "p600_gram": summarize(vals["base_p600_gram"]),
                "p600_cv": summarize(vals["base_p600_cv"]),
                "p600_test": t_base_p600,
            },
            "garden_path": {
                "n400_unambiguous": summarize(vals["unamb_n400"]),
                "n400_garden_path": summarize(vals["gp_n400"]),
                "n400_test": t_n400_gp,
                "p600_unambiguous": summarize(vals["unamb_p600_patient"]),
                "p600_garden_path": summarize(vals["gp_p600_patient"]),
                "p600_test": t_p600_gp,
            },
            "post_object_predictions": {
                "n400_prep": summarize(vals["gp_n400_prep"]),
                "n400_verb": summarize(vals["gp_n400_verb"]),
                "n400_noun": summarize(vals["gp_n400_noun"]),
            },
            "hypotheses": {
                "H1_gp_higher_n400": gp_higher_n400,
                "H2_gp_higher_p600": gp_higher_p600,
                "H3_baseline_intact": baseline_ok,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "pp_prob": cfg.pp_prob, "rel_prob": cfg.rel_prob,
                "n_train_sentences": cfg.n_train_sentences,
                "training_reps": cfg.training_reps,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Garden-Path ERP Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = GardenPathERPExperiment(verbose=True)

    if args.quick:
        cfg = GardenPathConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2)
        n_seeds = args.seeds or 5
    else:
        cfg = GardenPathConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("GARDEN-PATH ERP SUMMARY")
    print("=" * 70)

    print("\nBaseline (object position):")
    b = m["baseline"]
    print(f"  N400 — gram: {b['n400_gram']['mean']:.3f}"
          f"  cv: {b['n400_cv']['mean']:.3f}")
    print(f"  P600 — gram: {b['p600_gram']['mean']:.3f}"
          f"  cv: {b['p600_cv']['mean']:.3f}"
          f"  d={b['p600_test']['d']:.2f}")

    print("\nGarden-path vs Unambiguous:")
    g = m["garden_path"]
    print(f"  N400 at 2nd verb — unamb: {g['n400_unambiguous']['mean']:.3f}"
          f"  GP: {g['n400_garden_path']['mean']:.3f}"
          f"  d={g['n400_test']['d']:.2f}")
    print(f"  P600 at patient  — unamb: {g['p600_unambiguous']['mean']:.3f}"
          f"  GP: {g['p600_garden_path']['mean']:.3f}"
          f"  d={g['p600_test']['d']:.2f}")

    print("\nPost-object predictions:")
    po = m["post_object_predictions"]
    print(f"  Prep: {po['n400_prep']['mean']:.3f}"
          f"  Verb: {po['n400_verb']['mean']:.3f}"
          f"  Noun: {po['n400_noun']['mean']:.3f}")

    h = m["hypotheses"]
    print(f"\nH1 GP>Unamb N400: {'PASS' if h['H1_gp_higher_n400'] else 'FAIL'}")
    print(f"H2 GP>Unamb P600: {'PASS' if h['H2_gp_higher_p600'] else 'FAIL'}")
    print(f"H3 Baseline OK:   {'PASS' if h['H3_baseline_intact'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
