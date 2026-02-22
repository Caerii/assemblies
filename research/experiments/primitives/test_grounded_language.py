"""
Grounded Language: Sensory Features Affect Prediction and Enable Cross-Modal Activation

Tests whether sensory grounding (co-projecting PHON + sensory features into
core areas) creates meaningful semantic structure that affects prediction (N400)
and enables cross-modal activation.

Three tests:
  Test 1: Semantic overlap — words sharing sensory features (dog/cat → SENSE_ANIMAL)
          have higher assembly overlap than unrelated words (dog/garden)
  Test 2: Grounded prediction — after training, same-group nouns produce lower N400
          than different-group nouns when predicted by context
  Test 3: Cross-modal activation — sensory features alone (no phonological input)
          partially evoke word-like assemblies in core areas

Hypotheses:
  H1: Grounded overlap — within-group > between-group overlap (d > 0.5)
  H2: Semantic N400 — same-group N400 < different-group N400 (d > 0.5)
  H3: Cross-modal — sensory-only activation overlaps with word assembly > chance

Usage:
    uv run python research/experiments/primitives/test_grounded_language.py
    uv run python research/experiments/primitives/test_grounded_language.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
    measure_overlap,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400
from research.experiments.lib.grounding import (
    SENSORY_FEATURES,
    SEMANTIC_GROUPS,
    create_grounded_brain,
    activate_sensory_only,
)


@dataclass
class GroundedLanguageConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    grounding_rounds: int = 10
    # Training
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    n_train_sentences: int = 100
    training_reps: int = 3
    pp_prob: float = 0.4
    # Test
    n_test_items: int = 5


def run_trial(
    cfg: GroundedLanguageConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one grounded language trial."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB
    n_train = cfg.n_train_sentences * cfg.training_reps

    # Create grounded brain
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_grounded_brain(
        bcfg, vocab, seed, cfg.grounding_rounds)

    # ── Test 1: Semantic overlap from shared grounding ─────────────
    # Measure assembly overlap for within-group vs between-group pairs
    within_overlaps = []
    between_overlaps = []

    # Read out word assemblies from core areas
    word_assemblies = {}
    for word in vocab.all_words:
        core = vocab.core_area_for(word)
        activate_word(brain, word, core, 3)
        word_assemblies[word] = np.array(
            brain.areas[core].winners, dtype=np.uint32)

    # Within-group: words sharing sensory features
    for group_name, group_words in SEMANTIC_GROUPS.items():
        available = [w for w in group_words if w in word_assemblies]
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                w1, w2 = available[i], available[j]
                # Only compare within same core area
                if vocab.core_area_for(w1) == vocab.core_area_for(w2):
                    ov = measure_overlap(
                        word_assemblies[w1], word_assemblies[w2])
                    within_overlaps.append(ov)

    # Between-group: words from different semantic groups in same core area
    all_nouns = vocab.words_for_category("NOUN")
    animal_words = set(SEMANTIC_GROUPS.get("ANIMAL", []))
    person_words = set(SEMANTIC_GROUPS.get("PERSON", []))
    for w1 in animal_words:
        for w2 in person_words:
            if (w1 in word_assemblies and w2 in word_assemblies
                    and vocab.core_area_for(w1) == vocab.core_area_for(w2)):
                ov = measure_overlap(
                    word_assemblies[w1], word_assemblies[w2])
                between_overlaps.append(ov)

    # Also compare across locations (PLACE group) vs nouns
    place_words = set(SEMANTIC_GROUPS.get("PLACE", []))
    for w1 in animal_words:
        for w2 in place_words:
            if (w1 in word_assemblies and w2 in word_assemblies
                    and vocab.core_area_for(w1) == vocab.core_area_for(w2)):
                ov = measure_overlap(
                    word_assemblies[w1], word_assemblies[w2])
                between_overlaps.append(ov)

    # ── Train prediction bridges ───────────────────────────────────
    grammar = SimpleCFG(pp_prob=cfg.pp_prob, vocab=vocab, rng=rng)
    train_sents = grammar.generate_batch(n_train)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    # ── Test 2: Grounded prediction — semantic N400 ────────────────
    # After "dog chases", predict object:
    #   same-group noun (cat — also ANIMAL) should have lower N400
    #   than different-group noun (garden — PLACE)
    verbs = vocab.words_for_category("VERB")
    nouns = vocab.words_for_category("NOUN")
    locations = vocab.words_for_category("LOCATION")

    same_group_n400 = []
    diff_group_n400 = []

    ni = cfg.n_test_items
    animals = [w for w in nouns if w in animal_words]
    non_animals = [w for w in nouns if w not in animal_words]
    if not non_animals:
        non_animals = locations[:3]

    for i in range(ni):
        agent = animals[i % len(animals)]
        verb = verbs[i % len(verbs)]

        # Context: agent + verb -> predict object
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # Same-group: another animal
        same_obj = animals[(i + 1) % len(animals)]
        same_group_n400.append(measure_n400(predicted, lexicon[same_obj]))

        # Different-group: non-animal noun or location
        diff_obj = non_animals[i % len(non_animals)]
        diff_group_n400.append(measure_n400(predicted, lexicon[diff_obj]))

    # ── Test 3: Cross-modal activation ─────────────────────────────
    # Activate SENSE_ANIMAL alone -> project to NOUN_CORE
    # Does it partially activate animal-word assemblies?
    cross_modal_overlaps = []
    cross_modal_baselines = []

    for word in animals:
        features = SENSORY_FEATURES.get(word, [])
        if not features:
            continue

        # Sensory-only activation
        sensory_assembly = activate_sensory_only(
            brain, features, "NOUN_CORE", rounds=3)

        # Compare with the word's phonological assembly
        word_assembly = word_assemblies[word]
        cross_modal_overlaps.append(
            measure_overlap(sensory_assembly, word_assembly))

        # Baseline: overlap with an unrelated word
        baseline_word = non_animals[0] if non_animals else locations[0]
        baseline_assembly = word_assemblies.get(baseline_word)
        if baseline_assembly is not None:
            cross_modal_baselines.append(
                measure_overlap(sensory_assembly, baseline_assembly))

    brain.disable_plasticity = False

    return {
        "within_overlaps": within_overlaps,
        "between_overlaps": between_overlaps,
        "same_group_n400": same_group_n400,
        "diff_group_n400": diff_group_n400,
        "cross_modal_overlaps": cross_modal_overlaps,
        "cross_modal_baselines": cross_modal_baselines,
    }


class GroundedLanguageExperiment(ExperimentBase):
    """Grounded language experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="grounded_language",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[GroundedLanguageConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or GroundedLanguageConfig(
            **{k: v for k, v in kwargs.items()
               if k in GroundedLanguageConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Grounded Language: Sensory Features in Prediction")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  grounding_rounds={cfg.grounding_rounds}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        all_within = []
        all_between = []
        all_same_n400 = []
        all_diff_n400 = []
        all_cross_modal = []
        all_cross_baseline = []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            all_within.append(float(np.mean(trial["within_overlaps"]))
                              if trial["within_overlaps"] else 0.0)
            all_between.append(float(np.mean(trial["between_overlaps"]))
                               if trial["between_overlaps"] else 0.0)
            all_same_n400.append(float(np.mean(trial["same_group_n400"]))
                                 if trial["same_group_n400"] else 0.0)
            all_diff_n400.append(float(np.mean(trial["diff_group_n400"]))
                                 if trial["diff_group_n400"] else 0.0)
            all_cross_modal.append(float(np.mean(trial["cross_modal_overlaps"]))
                                   if trial["cross_modal_overlaps"] else 0.0)
            all_cross_baseline.append(float(np.mean(trial["cross_modal_baselines"]))
                                      if trial["cross_modal_baselines"] else 0.0)

        # Compute effect sizes
        overlap_test = paired_ttest(all_within, all_between)
        n400_test = paired_ttest(all_diff_n400, all_same_n400)  # diff > same expected
        cross_modal_test = paired_ttest(all_cross_modal, all_cross_baseline)

        # Report
        self.log(f"\n  === Test 1: Semantic Overlap ===")
        self.log(f"    Within-group:  {np.mean(all_within):.4f} "
                 f"+/- {np.std(all_within)/max(np.sqrt(n_seeds),1):.4f}")
        self.log(f"    Between-group: {np.mean(all_between):.4f} "
                 f"+/- {np.std(all_between)/max(np.sqrt(n_seeds),1):.4f}")
        self.log(f"    d = {overlap_test['d']:.2f}")

        self.log(f"\n  === Test 2: Semantic N400 ===")
        self.log(f"    Same-group N400:  {np.mean(all_same_n400):.4f}")
        self.log(f"    Diff-group N400:  {np.mean(all_diff_n400):.4f}")
        self.log(f"    d = {n400_test['d']:.2f}")

        self.log(f"\n  === Test 3: Cross-Modal Activation ===")
        self.log(f"    Sensory->word overlap:  {np.mean(all_cross_modal):.4f}")
        self.log(f"    Sensory->unrelated:     {np.mean(all_cross_baseline):.4f}")
        self.log(f"    d = {cross_modal_test['d']:.2f}")

        # Hypotheses
        h1 = overlap_test["d"] > 0.5
        h2 = n400_test["d"] > 0.5
        h3 = cross_modal_test["d"] > 0.5

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Overlap d > 0.5):       "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" (d={overlap_test['d']:.2f})")
        self.log(f"    H2 (Semantic N400 d > 0.5): "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" (d={n400_test['d']:.2f})")
        self.log(f"    H3 (Cross-modal d > 0.5):   "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (d={cross_modal_test['d']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "semantic_overlap": {
                "within_group": summarize(all_within),
                "between_group": summarize(all_between),
                **overlap_test,
            },
            "semantic_n400": {
                "same_group": summarize(all_same_n400),
                "diff_group": summarize(all_diff_n400),
                **n400_test,
            },
            "cross_modal": {
                "sensory_word": summarize(all_cross_modal),
                "sensory_baseline": summarize(all_cross_baseline),
                **cross_modal_test,
            },
            "hypotheses": {
                "H1_grounded_overlap": h1,
                "H2_semantic_n400": h2,
                "H3_cross_modal": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "grounding_rounds": cfg.grounding_rounds,
                "n_train_sentences": cfg.n_train_sentences,
                "training_reps": cfg.training_reps,
                "pp_prob": cfg.pp_prob,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Grounded Language Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = GroundedLanguageExperiment(verbose=True)

    if args.quick:
        cfg = GroundedLanguageConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2,
            n_test_items=4)
        n_seeds = args.seeds or 3
    else:
        cfg = GroundedLanguageConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("GROUNDED LANGUAGE SUMMARY")
    print("=" * 70)
    print(f"\nH1 Grounded overlap:  {'PASS' if h['H1_grounded_overlap'] else 'FAIL'}")
    print(f"H2 Semantic N400:     {'PASS' if h['H2_semantic_n400'] else 'FAIL'}")
    print(f"H3 Cross-modal:       {'PASS' if h['H3_cross_modal'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
