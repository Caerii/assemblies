"""
Free-Form Learner: Learning Language From Scratch (2-CORE Baseline)

Tests whether the FreeFormLearner can acquire language from raw sentence
exposure without any pre-registered words or category labels. Uses 2 CORE
areas with refractory_period=1 to discover a binary noun/verb split from
SVO word order alone.

Setup:
  - Create empty-brain FreeFormLearner (2 CORE areas, 6 STRUCT areas)
  - Feed sentences from SimpleCFG one at a time (words only, no labels)
  - Measure learning curves at checkpoints

Hypotheses:
  H1: Word assemblies stable after formation (overlap > 0.8)
  H2: Cumulative N400 separates grammatical from scrambled (d > 1.0)
  H3: Binding consistency > 70% (same word -> same STRUCT area)
  H4: Prediction improves with exposure (N400 decreases over time)
  H5: Generalization to novel word combinations

Definitive results (10 seeds, n=10000, 300 train sentences, 30 test each):
  H1: PASS  stability = 0.823 +/- 0.008
  H2: FAIL  N400 d = 0.23 +/- 0.09  (positive but below threshold)
  H3: PASS  binding = 99.9% +/- 0.1%
  H4: FAIL  5/10 seeds improved (borderline)
  H5: FAIL  novel d = 0.33 +/- 0.05

  Category discovery: 8/8 correct (NOUN+LOC vs VERB+PREP) across ALL seeds
  N400 limitation: binary prediction (A->B, B->A) cannot distinguish
    grammatical from scrambled because scrambling preserves alternation.
  See test_freeform_det.py for the 3-category extension that solves this.

Usage:
    uv run python research/experiments/primitives/test_freeform_learner.py
    uv run python research/experiments/primitives/test_freeform_learner.py --quick
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
from research.experiments.lib.freeform import FreeFormConfig, FreeFormLearner


@dataclass
class FreeFormExperimentConfig:
    # Learner params
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    n_core_areas: int = 2
    core_refractory_period: int = 1
    n_struct_areas: int = 6
    struct_refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    lexicon_refresh_interval: int = 50
    use_p600_feedback: bool = False
    # Experiment params
    n_train_sentences: int = 300
    checkpoint_interval: int = 50
    n_test_grammatical: int = 30
    n_test_scrambled: int = 30


def make_learner_config(cfg: FreeFormExperimentConfig) -> FreeFormConfig:
    """Extract FreeFormConfig from experiment config."""
    return FreeFormConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds,
        n_core_areas=cfg.n_core_areas,
        core_refractory_period=cfg.core_refractory_period,
        n_struct_areas=cfg.n_struct_areas,
        struct_refractory_period=cfg.struct_refractory_period,
        inhibition_strength=cfg.inhibition_strength,
        stabilize_rounds=cfg.stabilize_rounds,
        train_rounds_per_pair=cfg.train_rounds_per_pair,
        binding_rounds=cfg.binding_rounds,
        lexicon_refresh_interval=cfg.lexicon_refresh_interval,
        use_p600_feedback=cfg.use_p600_feedback,
    )


def generate_scrambled(words: List[str], rng: np.random.Generator) -> List[str]:
    """Scramble word order to create ungrammatical sequence."""
    scrambled = list(words)
    rng.shuffle(scrambled)
    # Ensure it's actually different
    if scrambled == list(words):
        scrambled[0], scrambled[-1] = scrambled[-1], scrambled[0]
    return scrambled


def generate_novel_combinations(
    vocab, rng: np.random.Generator, n_items: int,
) -> List[List[str]]:
    """Generate sentences with novel word combinations (SVO + SVO-PP)."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    preps = vocab.words_for_category("PREP")
    locs = vocab.words_for_category("LOCATION")
    combos = []
    for i in range(n_items):
        agent = nouns[rng.integers(len(nouns))]
        verb = verbs[rng.integers(len(verbs))]
        patient = nouns[rng.integers(len(nouns))]
        while patient == agent:
            patient = nouns[rng.integers(len(nouns))]
        sent = [agent, verb, patient]
        # ~30% chance of PP, matching training distribution
        if rng.random() < 0.3:
            prep = preps[rng.integers(len(preps))]
            loc = locs[rng.integers(len(locs))]
            sent.extend([prep, loc])
        combos.append(sent)
    return combos


def run_trial(cfg: FreeFormExperimentConfig, seed: int) -> Dict[str, Any]:
    """Run one trial of free-form learning."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB

    # Create learner (empty brain)
    learner_cfg = make_learner_config(cfg)
    learner = FreeFormLearner(learner_cfg, seed)

    # Generate curriculum (only words, no labels)
    grammar = SimpleCFG(pp_prob=0.3, vocab=vocab, rng=rng)
    curriculum = grammar.generate_batch(cfg.n_train_sentences)

    # Generate test sets (pp_prob=0.3 matches training distribution;
    # longer sentences give more prediction positions for stronger N400)
    test_grammar = SimpleCFG(pp_prob=0.3, vocab=vocab,
                             rng=np.random.default_rng(seed + 5000))
    test_grammatical = [
        s["words"] for s in test_grammar.generate_batch(cfg.n_test_grammatical)
    ]
    test_scrambled = [
        generate_scrambled(g, rng) for g in test_grammatical
    ]

    # Learning loop with checkpoints
    checkpoint_data = []

    for i, sent in enumerate(curriculum):
        learner.process_sentence(sent["words"])

        if (i + 1) % cfg.checkpoint_interval == 0:
            # Force lexicon refresh for measurements
            learner._refresh_lexicon()

            # Measure assembly stability (sample 5 known words)
            stab_words = list(learner.vocab.known_words)[:5]
            stabilities = [
                learner.measure_assembly_stability(w) for w in stab_words
            ]

            # Measure N400 grammatical vs scrambled
            gram_n400s = [
                learner.measure_sentence_acceptability(g)
                for g in test_grammatical
            ]
            scram_n400s = [
                learner.measure_sentence_acceptability(s)
                for s in test_scrambled
            ]

            # Binding consistency (sample 3 words)
            bind_words = list(learner.vocab.known_words)[:3]
            consistencies = [
                learner.measure_binding_consistency(w, n_trials=5)
                for w in bind_words
            ]

            cp = {
                "sentence_idx": i + 1,
                "n_words": len(learner.vocab.known_words),
                "mean_stability": float(np.mean(stabilities)) if stabilities else 0.0,
                "gram_n400_mean": float(np.mean(gram_n400s)),
                "scram_n400_mean": float(np.mean(scram_n400s)),
                "n400_separation": float(np.mean(scram_n400s)) - float(np.mean(gram_n400s)),
                "mean_binding_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
            }
            checkpoint_data.append(cp)

    # Final measurements
    learner._refresh_lexicon()

    # H1: Assembly stability for all known words
    all_stabilities = [
        learner.measure_assembly_stability(w) for w in learner.vocab.known_words
    ]

    # H2: Final N400 separation
    final_gram = [learner.measure_sentence_acceptability(g)
                  for g in test_grammatical]
    final_scram = [learner.measure_sentence_acceptability(s)
                   for s in test_scrambled]
    n400_test = paired_ttest(final_scram, final_gram)

    # H3: Binding consistency for all known words
    all_consistency = [
        learner.measure_binding_consistency(w, n_trials=5)
        for w in learner.vocab.known_words
    ]

    # H4: Prediction improvement (compare first and last checkpoint)
    if len(checkpoint_data) >= 2:
        first_sep = checkpoint_data[0]["n400_separation"]
        last_sep = checkpoint_data[-1]["n400_separation"]
        prediction_improved = last_sep > first_sep
    else:
        prediction_improved = False
        first_sep = last_sep = 0.0

    # H5: Novel combinations
    novel_sents = generate_novel_combinations(vocab, rng, cfg.n_test_grammatical)
    novel_n400s = [learner.measure_sentence_acceptability(s) for s in novel_sents]
    novel_scram = [
        learner.measure_sentence_acceptability(generate_scrambled(s, rng))
        for s in novel_sents
    ]
    novel_test = paired_ttest(novel_scram, novel_n400s)

    return {
        "checkpoints": checkpoint_data,
        "all_stabilities": all_stabilities,
        "mean_stability": float(np.mean(all_stabilities)),
        "n400_gram_mean": float(np.mean(final_gram)),
        "n400_scram_mean": float(np.mean(final_scram)),
        "n400_d": n400_test["d"],
        "mean_binding_consistency": float(np.mean(all_consistency)),
        "prediction_improved": prediction_improved,
        "first_separation": first_sep,
        "last_separation": last_sep,
        "novel_d": novel_test["d"],
        "n_words_learned": len(learner.vocab.known_words),
        "stats": learner.get_stats(),
    }


class FreeFormLearnerExperiment(ExperimentBase):
    """Free-form learner experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="freeform_learner",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 5,
        config: Optional[FreeFormExperimentConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or FreeFormExperimentConfig()

        self.log("=" * 70)
        self.log("Free-Form Learner: Learning Language From Scratch")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train={cfg.n_train_sentences}, "
                 f"checkpoint_interval={cfg.checkpoint_interval}")
        self.log(f"  n_core_areas={cfg.n_core_areas}, "
                 f"core_refractory={cfg.core_refractory_period}")
        self.log(f"  n_struct_areas={cfg.n_struct_areas}, "
                 f"struct_refractory={cfg.struct_refractory_period}")
        self.log(f"  p600_feedback={cfg.use_p600_feedback}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Accumulators
        stabilities = []
        n400_ds = []
        consistencies = []
        improvements = []
        novel_ds = []
        n_words = []

        for s in range(n_seeds):
            self.log(f"\n  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            stabilities.append(trial["mean_stability"])
            n400_ds.append(trial["n400_d"])
            consistencies.append(trial["mean_binding_consistency"])
            improvements.append(trial["prediction_improved"])
            novel_ds.append(trial["novel_d"])
            n_words.append(trial["n_words_learned"])

            # Log per-seed summary
            self.log(f"    Words: {trial['n_words_learned']}, "
                     f"Stab: {trial['mean_stability']:.3f}, "
                     f"N400 d: {trial['n400_d']:.2f}, "
                     f"Bind: {trial['mean_binding_consistency']:.2f}, "
                     f"Novel d: {trial['novel_d']:.2f}")

            # Log word -> CORE area assignments
            stats = trial["stats"]
            if "word_assignments" in stats:
                self.log(f"    CORE assignments: {stats['word_assignments']}")
            if "words_per_core_area" in stats:
                self.log(f"    Words per area: {stats['words_per_core_area']}")

            # Log learning curve
            for cp in trial["checkpoints"]:
                self.log(f"      @{cp['sentence_idx']:>4d}: "
                         f"n400_sep={cp['n400_separation']:+.3f}, "
                         f"stab={cp['mean_stability']:.3f}, "
                         f"bind={cp['mean_binding_consistency']:.2f}")

        # Aggregate
        stab_summary = summarize(stabilities)
        n400_d_summary = summarize(n400_ds)
        consist_summary = summarize(consistencies)
        novel_d_summary = summarize(novel_ds)

        # Hypotheses
        h1 = stab_summary["mean"] > 0.8
        h2 = n400_d_summary["mean"] > 1.0
        h3 = consist_summary["mean"] > 0.7
        h4 = sum(improvements) > len(improvements) / 2
        h5 = novel_d_summary["mean"] > 0.5

        self.log(f"\n  === Results (across {n_seeds} seeds) ===")
        self.log(f"    Assembly stability:     {stab_summary['mean']:.3f} "
                 f"+/- {stab_summary.get('sem', 0):.3f}")
        self.log(f"    N400 gram vs scram d:   {n400_d_summary['mean']:.2f} "
                 f"+/- {n400_d_summary.get('sem', 0):.2f}")
        self.log(f"    Binding consistency:     {consist_summary['mean']:.3f} "
                 f"+/- {consist_summary.get('sem', 0):.3f}")
        self.log(f"    Prediction improved:     "
                 f"{sum(improvements)}/{len(improvements)} seeds")
        self.log(f"    Novel combination d:     {novel_d_summary['mean']:.2f} "
                 f"+/- {novel_d_summary.get('sem', 0):.2f}")
        self.log(f"    Words learned:           "
                 f"{float(np.mean(n_words)):.0f}")

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Stability > 0.8):    "
                 f"{'PASS' if h1 else 'FAIL'} ({stab_summary['mean']:.3f})")
        self.log(f"    H2 (N400 sep d > 1.0):   "
                 f"{'PASS' if h2 else 'FAIL'} ({n400_d_summary['mean']:.2f})")
        self.log(f"    H3 (Binding > 70%):      "
                 f"{'PASS' if h3 else 'FAIL'} ({consist_summary['mean']:.1%})")
        self.log(f"    H4 (Prediction improves): "
                 f"{'PASS' if h4 else 'FAIL'} "
                 f"({sum(improvements)}/{len(improvements)})")
        self.log(f"    H5 (Novel d > 0.5):      "
                 f"{'PASS' if h5 else 'FAIL'} ({novel_d_summary['mean']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "assembly_stability": stab_summary,
            "n400_separation_d": n400_d_summary,
            "binding_consistency": consist_summary,
            "prediction_improved_count": sum(improvements),
            "novel_combination_d": novel_d_summary,
            "n_words_learned": float(np.mean(n_words)),
            "hypotheses": {
                "H1_assembly_stable": h1,
                "H2_n400_separation": h2,
                "H3_binding_consistent": h3,
                "H4_prediction_improves": h4,
                "H5_novel_generalization": h5,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_train_sentences": cfg.n_train_sentences,
                "n_core_areas": cfg.n_core_areas,
                "core_refractory_period": cfg.core_refractory_period,
                "n_struct_areas": cfg.n_struct_areas,
                "struct_refractory_period": cfg.struct_refractory_period,
                "use_p600_feedback": cfg.use_p600_feedback,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Free-Form Learner Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = FreeFormLearnerExperiment(verbose=True)

    if args.quick:
        cfg = FreeFormExperimentConfig(
            n=5000, k=50,
            n_train_sentences=100,
            checkpoint_interval=25,
            n_test_grammatical=15,
            n_test_scrambled=15,
            lexicon_refresh_interval=25,
        )
        n_seeds = args.seeds or 3
    else:
        cfg = FreeFormExperimentConfig()
        n_seeds = args.seeds or 5

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("FREE-FORM LEARNER SUMMARY")
    print("=" * 70)
    print(f"\nH1 Assembly stability > 0.8:  {'PASS' if h['H1_assembly_stable'] else 'FAIL'}")
    print(f"H2 N400 separation d > 1.0:   {'PASS' if h['H2_n400_separation'] else 'FAIL'}")
    print(f"H3 Binding consistency > 70%: {'PASS' if h['H3_binding_consistent'] else 'FAIL'}")
    print(f"H4 Prediction improves:       {'PASS' if h['H4_prediction_improves'] else 'FAIL'}")
    print(f"H5 Novel generalization:      {'PASS' if h['H5_novel_generalization'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
