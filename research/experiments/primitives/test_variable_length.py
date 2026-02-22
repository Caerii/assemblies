"""
Variable-Length Sentence Processing: SVO + Optional PP

Tests whether the prediction+binding mechanism generalizes from fixed 3-word
SVO sentences to variable-length input without any length-specific machinery.

The minimal extension: SVO ("dog chases cat") mixed with SVO+PP ("dog chases
cat in garden"). The prediction chain extends naturally — each word predicts
the next via the same co-projection mechanism. No length-specific logic needed.

Architecture (8 areas — vocabulary-driven, auto-configured):
  NOUN_CORE, VERB_CORE, PREP_CORE  -- lexical assemblies
  PREDICTION                        -- forward projection target
  ROLE_AGENT, ROLE_PATIENT, ROLE_PP_OBJ -- structural role slots

Hypotheses:
  H1 (Length invariance): N400/P600 at object position are unaffected by
      mixed-length training
  H2 (PP position ERP): N400/P600 at PP-object position show the same
      double dissociation as the object position
  H3 (Cross-position prediction): prediction chain works across all
      sentence positions
  H4 (PP binding): anchored instability separates trained vs untrained
      PP-object bindings

Usage:
    uv run python research/experiments/primitives/test_variable_length.py
    uv run python research/experiments/primitives/test_variable_length.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB, NOUNS, VERBS, PREPS
from research.experiments.lib.grammar import generate_mixed_sentences
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import (
    measure_erps_at_position,
    measure_n400,
    generate_test_triples,
    generate_pp_test_triples,
)


@dataclass
class VariableLengthConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    training_reps: int = 3
    n_train_sentences: int = 30
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    pp_fraction: float = 0.4


def run_trial(
    cfg: VariableLengthConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one variable-length ERP trial.

    Trains on mixed SVO / SVO+PP sentences, then tests N400 and P600
    at the object position (word 3) and PP-object position (word 5).
    """
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB

    # Create brain with vocabulary-driven area configuration
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Train on mixed-length CFG sentences
    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = generate_mixed_sentences(
        n_train, cfg.pp_fraction, rng=rng)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    # Build prediction lexicon (plasticity OFF)
    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    # -- Test: object position (word 3) ---
    obj_triples = generate_test_triples(rng, 5, vocab)

    obj_n400_gram, obj_n400_cv, obj_n400_novel = [], [], []
    obj_p600_gram, obj_p600_cv, obj_p600_novel = [], [], []

    for agent, verb_word, gram_obj, cv_obj, novel_obj in obj_triples:
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb_word, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        erps = measure_erps_at_position(
            brain, predicted, lexicon,
            gram_word=gram_obj, gram_core="NOUN_CORE",
            catviol_word=cv_obj, catviol_core="VERB_CORE",
            novel_word=novel_obj, novel_core="NOUN_CORE",
            role_area="ROLE_PATIENT", n_settling_rounds=cfg.n_settling_rounds)

        obj_n400_gram.append(erps["n400_gram"])
        obj_n400_cv.append(erps["n400_catviol"])
        obj_n400_novel.append(erps["n400_novel"])
        obj_p600_gram.append(erps["p600_gram"])
        obj_p600_cv.append(erps["p600_catviol"])
        obj_p600_novel.append(erps["p600_novel"])

    # -- Test: PP-object position (word 5) ---
    pp_triples = generate_pp_test_triples(rng, 5, vocab)

    pp_n400_gram, pp_n400_cv, pp_n400_novel = [], [], []
    pp_p600_gram, pp_p600_cv, pp_p600_novel = [], [], []

    for (agent, verb_word, patient, prep,
         gram_pp, cv_pp, novel_pp) in pp_triples:
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb_word, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)
        activate_word(brain, prep, "PREP_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"PREP_CORE": ["PREDICTION"]})
        predicted_pp = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        erps = measure_erps_at_position(
            brain, predicted_pp, lexicon,
            gram_word=gram_pp, gram_core="NOUN_CORE",
            catviol_word=cv_pp, catviol_core="VERB_CORE",
            novel_word=novel_pp, novel_core="NOUN_CORE",
            role_area="ROLE_PP_OBJ", n_settling_rounds=cfg.n_settling_rounds)

        pp_n400_gram.append(erps["n400_gram"])
        pp_n400_cv.append(erps["n400_catviol"])
        pp_n400_novel.append(erps["n400_novel"])
        pp_p600_gram.append(erps["p600_gram"])
        pp_p600_cv.append(erps["p600_catviol"])
        pp_p600_novel.append(erps["p600_novel"])

    # -- Test: preposition position (word 4) ---
    prep_n400_expected, prep_n400_unexpected = [], []

    for (agent, verb_word, patient, prep,
         gram_pp, cv_pp, novel_pp) in pp_triples:
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb_word, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)

        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        predicted_at_prep = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        prep_n400_expected.append(measure_n400(predicted_at_prep, lexicon[prep]))
        unexpected_noun = NOUNS[(NOUNS.index(patient) + 2) % len(NOUNS)]
        prep_n400_unexpected.append(
            measure_n400(predicted_at_prep, lexicon[unexpected_noun]))

    brain.disable_plasticity = False

    return {
        "obj_n400_gram_mean": float(np.mean(obj_n400_gram)),
        "obj_n400_catviol_mean": float(np.mean(obj_n400_cv)),
        "obj_n400_novel_mean": float(np.mean(obj_n400_novel)),
        "obj_p600_gram_mean": float(np.mean(obj_p600_gram)),
        "obj_p600_catviol_mean": float(np.mean(obj_p600_cv)),
        "obj_p600_novel_mean": float(np.mean(obj_p600_novel)),
        "pp_n400_gram_mean": float(np.mean(pp_n400_gram)),
        "pp_n400_catviol_mean": float(np.mean(pp_n400_cv)),
        "pp_n400_novel_mean": float(np.mean(pp_n400_novel)),
        "pp_p600_gram_mean": float(np.mean(pp_p600_gram)),
        "pp_p600_catviol_mean": float(np.mean(pp_p600_cv)),
        "pp_p600_novel_mean": float(np.mean(pp_p600_novel)),
        "prep_n400_expected_mean": float(np.mean(prep_n400_expected)),
        "prep_n400_unexpected_mean": float(np.mean(prep_n400_unexpected)),
    }


class VariableLengthExperiment(ExperimentBase):
    """Variable-length sentence processing: SVO + optional PP."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="variable_length",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[VariableLengthConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or VariableLengthConfig(
            **{k: v for k, v in kwargs.items()
               if k in VariableLengthConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Variable-Length Sentence Processing: SVO + PP")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  pp_fraction={cfg.pp_fraction}, "
                 f"n_train_sentences={cfg.n_train_sentences}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-seed means
        keys = [
            "obj_n400_gram", "obj_n400_catviol", "obj_n400_novel",
            "obj_p600_gram", "obj_p600_catviol", "obj_p600_novel",
            "pp_n400_gram", "pp_n400_catviol", "pp_n400_novel",
            "pp_p600_gram", "pp_p600_catviol", "pp_p600_novel",
            "prep_n400_expected", "prep_n400_unexpected",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            for k in keys:
                vals[k].append(result[f"{k}_mean"])

            if s == 0:
                self.log(f"    Obj  — N400: g={result['obj_n400_gram_mean']:.3f}"
                         f"  cv={result['obj_n400_catviol_mean']:.3f}"
                         f"  n={result['obj_n400_novel_mean']:.3f}")
                self.log(f"    Obj  — P600: g={result['obj_p600_gram_mean']:.3f}"
                         f"  cv={result['obj_p600_catviol_mean']:.3f}"
                         f"  n={result['obj_p600_novel_mean']:.3f}")
                self.log(f"    PP   — N400: g={result['pp_n400_gram_mean']:.3f}"
                         f"  cv={result['pp_n400_catviol_mean']:.3f}"
                         f"  n={result['pp_n400_novel_mean']:.3f}")
                self.log(f"    PP   — P600: g={result['pp_p600_gram_mean']:.3f}"
                         f"  cv={result['pp_p600_catviol_mean']:.3f}"
                         f"  n={result['pp_p600_novel_mean']:.3f}")

        # Statistical tests
        t = {}
        t["obj_n400_cv"] = paired_ttest(vals["obj_n400_catviol"], vals["obj_n400_gram"])
        t["obj_n400_novel"] = paired_ttest(vals["obj_n400_novel"], vals["obj_n400_gram"])
        t["obj_p600_cv"] = paired_ttest(vals["obj_p600_catviol"], vals["obj_p600_gram"])
        t["obj_p600_novel"] = paired_ttest(vals["obj_p600_novel"], vals["obj_p600_gram"])
        t["pp_n400_cv"] = paired_ttest(vals["pp_n400_catviol"], vals["pp_n400_gram"])
        t["pp_n400_novel"] = paired_ttest(vals["pp_n400_novel"], vals["pp_n400_gram"])
        t["pp_p600_cv"] = paired_ttest(vals["pp_p600_catviol"], vals["pp_p600_gram"])
        t["pp_p600_novel"] = paired_ttest(vals["pp_p600_novel"], vals["pp_p600_gram"])
        t["prep"] = paired_ttest(vals["prep_n400_unexpected"], vals["prep_n400_expected"])

        self.log(f"\n  === Object Position (Word 3) ===")
        self.log(f"    N400 — g: {np.mean(vals['obj_n400_gram']):.4f}"
                 f"  cv: {np.mean(vals['obj_n400_catviol']):.4f}"
                 f"  n: {np.mean(vals['obj_n400_novel']):.4f}")
        self.log(f"      CatViol>Gram: d={t['obj_n400_cv']['d']:.2f}")
        self.log(f"    P600 — g: {np.mean(vals['obj_p600_gram']):.4f}"
                 f"  cv: {np.mean(vals['obj_p600_catviol']):.4f}"
                 f"  n: {np.mean(vals['obj_p600_novel']):.4f}")
        self.log(f"      CatViol>Gram: d={t['obj_p600_cv']['d']:.2f},"
                 f"  Novel~Gram: d={t['obj_p600_novel']['d']:.2f}")

        self.log(f"\n  === PP-Object Position (Word 5) ===")
        self.log(f"    N400 — g: {np.mean(vals['pp_n400_gram']):.4f}"
                 f"  cv: {np.mean(vals['pp_n400_catviol']):.4f}"
                 f"  n: {np.mean(vals['pp_n400_novel']):.4f}")
        self.log(f"      CatViol>Gram: d={t['pp_n400_cv']['d']:.2f}")
        self.log(f"    P600 — g: {np.mean(vals['pp_p600_gram']):.4f}"
                 f"  cv: {np.mean(vals['pp_p600_catviol']):.4f}"
                 f"  n: {np.mean(vals['pp_p600_novel']):.4f}")
        self.log(f"      CatViol>Gram: d={t['pp_p600_cv']['d']:.2f},"
                 f"  Novel~Gram: d={t['pp_p600_novel']['d']:.2f}")

        self.log(f"\n  === Preposition Position (Word 4) ===")
        self.log(f"    N400 — expected: {np.mean(vals['prep_n400_expected']):.4f}"
                 f"  unexpected: {np.mean(vals['prep_n400_unexpected']):.4f}")
        self.log(f"    d={t['prep']['d']:.2f}, p={t['prep']['p']:.4f}")

        # Double dissociation
        dd_n400_novel = float(np.mean(vals["obj_n400_novel"]) - np.mean(vals["obj_n400_gram"]))
        dd_p600_novel = float(np.mean(vals["obj_p600_novel"]) - np.mean(vals["obj_p600_gram"]))
        dd_n400_cv = float(np.mean(vals["obj_n400_catviol"]) - np.mean(vals["obj_n400_gram"]))
        dd_p600_cv = float(np.mean(vals["obj_p600_catviol"]) - np.mean(vals["obj_p600_gram"]))

        self.log(f"\n  === Double Dissociation ===")
        self.log(f"    Novel:   N400={dd_n400_novel:+.3f}  P600={dd_p600_novel:+.3f}")
        self.log(f"    CatViol: N400={dd_n400_cv:+.3f}  P600={dd_p600_cv:+.3f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "object_position": {
                "n400": {c: summarize(vals[f"obj_n400_{c}"]) for c in ["gram", "catviol", "novel"]},
                "p600": {c: summarize(vals[f"obj_p600_{c}"]) for c in ["gram", "catviol", "novel"]},
                "tests": {k: v for k, v in t.items() if k.startswith("obj_")},
                "double_dissociation": {
                    "novel_n400": dd_n400_novel, "novel_p600": dd_p600_novel,
                    "catviol_n400": dd_n400_cv, "catviol_p600": dd_p600_cv,
                },
            },
            "pp_object_position": {
                "n400": {c: summarize(vals[f"pp_n400_{c}"]) for c in ["gram", "catviol", "novel"]},
                "p600": {c: summarize(vals[f"pp_p600_{c}"]) for c in ["gram", "catviol", "novel"]},
                "tests": {k: v for k, v in t.items() if k.startswith("pp_")},
            },
            "prep_position": {
                "n400_expected": summarize(vals["prep_n400_expected"]),
                "n400_unexpected": summarize(vals["prep_n400_unexpected"]),
                "test": t["prep"],
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max, "pp_fraction": cfg.pp_fraction,
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
        description="Variable-Length Sentence Processing Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = VariableLengthExperiment(verbose=True)

    if args.quick:
        cfg = VariableLengthConfig(
            n=5000, k=50, training_reps=3, n_train_sentences=20)
        n_seeds = args.seeds or 5
    else:
        cfg = VariableLengthConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("VARIABLE-LENGTH SUMMARY")
    print("=" * 70)

    for pos, label in [("object_position", "Object (word 3)"),
                       ("pp_object_position", "PP-Object (word 5)")]:
        p = m[pos]
        print(f"\n{label}:")
        print(f"  N400 — g: {p['n400']['gram']['mean']:.3f}"
              f"  cv: {p['n400']['catviol']['mean']:.3f}"
              f"  n: {p['n400']['novel']['mean']:.3f}")
        print(f"  P600 — g: {p['p600']['gram']['mean']:.3f}"
              f"  cv: {p['p600']['catviol']['mean']:.3f}"
              f"  n: {p['p600']['novel']['mean']:.3f}")

    pr = m["prep_position"]
    print(f"\nPrep (word 4): expected={pr['n400_expected']['mean']:.3f}"
          f"  unexpected={pr['n400_unexpected']['mean']:.3f}"
          f"  d={pr['test']['d']:.2f}")

    if "double_dissociation" in m["object_position"]:
        dd = m["object_position"]["double_dissociation"]
        print(f"\nDouble dissociation:")
        print(f"  Novel:   N400={dd['novel_n400']:+.3f}  P600={dd['novel_p600']:+.3f}")
        print(f"  CatViol: N400={dd['catviol_n400']:+.3f}  P600={dd['catviol_p600']:+.3f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
