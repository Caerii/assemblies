"""
N400 from Prediction Error

Tests whether prediction error -- the mismatch between a forward-projected
assembly and the actual incoming word's assembly in the PREDICTION area --
produces graded N400-like signals across sentence conditions.

This composes the forward prediction primitive (Hebbian bridges via
co-projection) with sentence-level processing to derive N400 as a natural
consequence of prediction failure, replacing ad-hoc energy-based proxies.

Architecture:
  PHON_<word>    — stimulus for each word
  NOUN_CORE      — noun assemblies self-organize here
  VERB_CORE      — verb assemblies self-organize here
  PREDICTION     — forward projection target

Protocol:
  1. Build word assemblies in NOUN_CORE and VERB_CORE
  2. Train prediction bridges via SVO co-projection (noun->verb, verb->noun)
  3. Build reference lexicon in PREDICTION (plasticity OFF)
  4. Test sentences word-by-word, measuring prediction error at critical
     (object) position:
       Grammatical:  "dog chases cat"   — trained noun in predicted noun slot
       CatViol:      "dog chases likes" — verb in predicted noun slot
       Novel:        "dog chases table" — untrained noun in predicted noun slot

Hypotheses:
  H1: prediction_error(catviol) > prediction_error(gram)
  H2: prediction_error(novel) between gram and catviol
  H3: At verb position, grammatical verbs show low prediction error

Usage:
    uv run python research/experiments/primitives/test_prediction_n400.py
    uv run python research/experiments/primitives/test_prediction_n400.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
    paired_ttest,
)
from src.core.brain import Brain


NOUNS = ["dog", "cat", "bird", "boy", "girl"]
VERBS = ["chases", "sees", "eats", "finds", "hits"]
NOVEL_NOUNS = ["table", "chair"]


@dataclass
class N400Config:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    training_reps: int = 3
    n_train_sentences: int = 20
    lexicon_readout_rounds: int = 5


def generate_svo_sentences(
    n_sentences: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str, str]]:
    """Generate random SVO triples (agent, verb, patient) from trained vocab."""
    sentences = []
    for _ in range(n_sentences):
        agent = rng.choice(NOUNS)
        patient = rng.choice([n for n in NOUNS if n != agent])
        verb = rng.choice(VERBS)
        sentences.append((agent, verb, patient))
    return sentences


def _activate_word(brain: Brain, stim_name: str, area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection."""
    brain.inhibit_areas([area])
    for _ in range(rounds):
        brain.project({stim_name: [area]}, {area: [area]})


def generate_test_triples(
    rng: np.random.Generator,
    n_triples: int = 5,
) -> List[Tuple[str, str, str, str, str]]:
    """Generate matched test triples: (agent, verb, gram_obj, catviol_obj, novel_obj).

    Each triple shares context (agent + verb) and varies only the critical word.
    """
    triples = []
    nouns = list(NOUNS)
    verbs = list(VERBS)
    novels = list(NOVEL_NOUNS)

    for i in range(n_triples):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        # Grammatical object: a trained noun different from agent
        remaining = [n for n in nouns if n != agent]
        gram_obj = remaining[i % len(remaining)]
        # Category violation: a verb in the object slot
        catviol_obj = verbs[(i + 1) % len(verbs)]
        # Novel object: an untrained noun
        novel_obj = novels[i % len(novels)]
        triples.append((agent, verb, gram_obj, catviol_obj, novel_obj))

    return triples


def run_trial(
    cfg: N400Config,
    seed: int,
) -> Dict[str, Any]:
    """Run one N400 prediction error trial.

    Returns prediction error metrics at the critical (object) position
    for grammatical, category violation, and novel conditions.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Create areas
    brain.add_area("NOUN_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("VERB_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("PREDICTION", cfg.n, cfg.k, cfg.beta)

    # Register stimuli (trained + novel)
    for noun in NOUNS + NOVEL_NOUNS:
        brain.add_stimulus(f"PHON_{noun}", cfg.k)
    for verb in VERBS:
        brain.add_stimulus(f"PHON_{verb}", cfg.k)

    # -- Build word assemblies in core areas -----------------------
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)

    # Build novel noun assemblies (same process, separate from trained)
    for noun in NOVEL_NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    # -- Prediction training (SVO co-projection) -------------------
    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = generate_svo_sentences(n_train, rng)

    for agent, verb_word, patient in train_sents:
        # Activate agent in NOUN_CORE
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)

        # Co-project: NOUN_CORE + verb stimulus -> PREDICTION
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.train_rounds_per_pair):
            brain.project(
                {f"PHON_{verb_word}": ["PREDICTION"]},
                {"NOUN_CORE": ["PREDICTION"]},
            )

        # Activate verb in VERB_CORE
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)

        # Co-project: VERB_CORE + object stimulus -> PREDICTION
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.train_rounds_per_pair):
            brain.project(
                {f"PHON_{patient}": ["PREDICTION"]},
                {"VERB_CORE": ["PREDICTION"]},
            )

    # -- Build prediction lexicon (plasticity OFF) -----------------
    brain.disable_plasticity = True
    lexicon = {}
    for word in NOUNS + VERBS + NOVEL_NOUNS:
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.lexicon_readout_rounds):
            brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
    brain.disable_plasticity = False

    # -- Test sentences --------------------------------------------
    brain.disable_plasticity = True

    test_triples = generate_test_triples(rng, n_triples=5)

    # Word-specific N400: 1 - overlap(predicted, word's PREDICTION entry)
    gram_errors = []
    catviol_errors = []
    novel_errors = []
    verb_pos_errors = []

    # Category-match N400: 1 - max_overlap(predicted, same_category_refs)
    # This captures category-level prediction (noun vs verb), not item-level.
    # For novel nouns, category match uses trained noun refs.
    gram_cat_errors = []
    catviol_cat_errors = []
    novel_cat_errors = []

    noun_refs = [lexicon[n] for n in NOUNS]
    verb_refs = [lexicon[v] for v in VERBS]

    for agent, verb_word, gram_obj, catviol_obj, novel_obj in test_triples:
        # --- Position 1: subject noun (activate context) ---
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)

        # --- Position 2: verb ---
        # Forward project: NOUN_CORE -> PREDICTION (verb expectation)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        predicted_at_verb = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # N400 at verb position: compare prediction with verb's lexicon entry
        verb_error = 1.0 - measure_overlap(predicted_at_verb, lexicon[verb_word])
        verb_pos_errors.append(verb_error)

        # Activate verb in VERB_CORE
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)

        # Forward project: VERB_CORE -> PREDICTION (object expectation)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted_at_obj = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # --- Position 3: critical word (object slot) ---
        # Word-specific N400
        gram_error = 1.0 - measure_overlap(predicted_at_obj, lexicon[gram_obj])
        catviol_error = 1.0 - measure_overlap(predicted_at_obj, lexicon[catviol_obj])
        novel_error = 1.0 - measure_overlap(predicted_at_obj, lexicon[novel_obj])

        gram_errors.append(gram_error)
        catviol_errors.append(catviol_error)
        novel_errors.append(novel_error)

        # Category-match N400: max overlap with same-category trained refs
        # gram_obj is a noun -> compare with noun refs
        gram_cat_match = max(measure_overlap(predicted_at_obj, r) for r in noun_refs)
        gram_cat_errors.append(1.0 - gram_cat_match)
        # catviol_obj is a verb -> compare with verb refs
        catviol_cat_match = max(measure_overlap(predicted_at_obj, r) for r in verb_refs)
        catviol_cat_errors.append(1.0 - catviol_cat_match)
        # novel_obj is a noun -> compare with trained noun refs
        novel_cat_match = max(measure_overlap(predicted_at_obj, r) for r in noun_refs)
        novel_cat_errors.append(1.0 - novel_cat_match)

    brain.disable_plasticity = False

    return {
        "gram_errors": gram_errors,
        "catviol_errors": catviol_errors,
        "novel_errors": novel_errors,
        "verb_pos_errors": verb_pos_errors,
        "gram_error_mean": float(np.mean(gram_errors)),
        "catviol_error_mean": float(np.mean(catviol_errors)),
        "novel_error_mean": float(np.mean(novel_errors)),
        "verb_pos_error_mean": float(np.mean(verb_pos_errors)),
        # Category-match metrics
        "gram_cat_error_mean": float(np.mean(gram_cat_errors)),
        "catviol_cat_error_mean": float(np.mean(catviol_cat_errors)),
        "novel_cat_error_mean": float(np.mean(novel_cat_errors)),
    }


class PredictionN400Experiment(ExperimentBase):
    """Test N400-like prediction error signals at critical sentence positions."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="prediction_n400",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[N400Config] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or N400Config(
            **{k: v for k, v in kwargs.items()
               if k in N400Config.__dataclass_fields__})

        self.log("=" * 70)
        self.log("N400 from Prediction Error")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  train_rounds_per_pair={cfg.train_rounds_per_pair}")
        self.log(f"  training_reps={cfg.training_reps}, "
                 f"n_train_sentences={cfg.n_train_sentences}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-seed means (word-specific and category-match)
        gram_vals = []
        catviol_vals = []
        novel_vals = []
        verb_pos_vals = []
        gram_cat_vals = []
        catviol_cat_vals = []
        novel_cat_vals = []

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            gram_vals.append(result["gram_error_mean"])
            catviol_vals.append(result["catviol_error_mean"])
            novel_vals.append(result["novel_error_mean"])
            verb_pos_vals.append(result["verb_pos_error_mean"])
            gram_cat_vals.append(result["gram_cat_error_mean"])
            catviol_cat_vals.append(result["catviol_cat_error_mean"])
            novel_cat_vals.append(result["novel_cat_error_mean"])

            if s == 0:
                self.log(f"    Word-specific:  gram={result['gram_error_mean']:.4f}  "
                         f"catviol={result['catviol_error_mean']:.4f}  "
                         f"novel={result['novel_error_mean']:.4f}")
                self.log(f"    Category-match: gram={result['gram_cat_error_mean']:.4f}  "
                         f"catviol={result['catviol_cat_error_mean']:.4f}  "
                         f"novel={result['novel_cat_error_mean']:.4f}")
                self.log(f"    Verb pos error: {result['verb_pos_error_mean']:.4f}")

        # H1: catviol > gram (word-specific)
        h1_test = paired_ttest(catviol_vals, gram_vals)
        # H2: novel between gram and catviol (word-specific)
        h2_novel_vs_gram = paired_ttest(novel_vals, gram_vals)
        h2_catviol_vs_novel = paired_ttest(catviol_vals, novel_vals)

        # Category-match tests
        h1_cat_test = paired_ttest(catviol_cat_vals, gram_cat_vals)
        h2_cat_novel_vs_gram = paired_ttest(novel_cat_vals, gram_cat_vals)

        self.log(f"\n  === Word-specific N400 ===")
        self.log(f"  H1 -- CatViol > Gram:")
        self.log(f"    Gram:    {np.mean(gram_vals):.4f} "
                 f"+/- {np.std(gram_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol: {np.mean(catviol_vals):.4f} "
                 f"+/- {np.std(catviol_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    d={h1_test['d']:.2f}, p={h1_test['p']:.4f}")

        self.log(f"\n  H2 -- Novel vs others:")
        self.log(f"    Novel:   {np.mean(novel_vals):.4f} "
                 f"+/- {np.std(novel_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Novel>Gram:    d={h2_novel_vs_gram['d']:.2f}, "
                 f"p={h2_novel_vs_gram['p']:.4f}")
        self.log(f"    CatViol>Novel: d={h2_catviol_vs_novel['d']:.2f}, "
                 f"p={h2_catviol_vs_novel['p']:.4f}")

        self.log(f"\n  === Category-match N400 ===")
        self.log(f"    Gram:    {np.mean(gram_cat_vals):.4f} "
                 f"+/- {np.std(gram_cat_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol: {np.mean(catviol_cat_vals):.4f} "
                 f"+/- {np.std(catviol_cat_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Novel:   {np.mean(novel_cat_vals):.4f} "
                 f"+/- {np.std(novel_cat_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol>Gram: d={h1_cat_test['d']:.2f}, "
                 f"p={h1_cat_test['p']:.4f}")
        self.log(f"    Novel>Gram:   d={h2_cat_novel_vs_gram['d']:.2f}, "
                 f"p={h2_cat_novel_vs_gram['p']:.4f}")

        self.log(f"\n  H3 -- Verb position (control):")
        self.log(f"    Verb pos error: {np.mean(verb_pos_vals):.4f} "
                 f"+/- {np.std(verb_pos_vals)/np.sqrt(n_seeds):.4f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "n400_gram": summarize(gram_vals),
            "n400_catviol": summarize(catviol_vals),
            "n400_novel": summarize(novel_vals),
            "n400_verb_pos": summarize(verb_pos_vals),
            "h1_catviol_vs_gram": h1_test,
            "h2_novel_vs_gram": h2_novel_vs_gram,
            "h2_catviol_vs_novel": h2_catviol_vs_novel,
            "n400_cat_gram": summarize(gram_cat_vals),
            "n400_cat_catviol": summarize(catviol_cat_vals),
            "n400_cat_novel": summarize(novel_cat_vals),
            "h1_cat_catviol_vs_gram": h1_cat_test,
            "h2_cat_novel_vs_gram": h2_cat_novel_vs_gram,
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "lexicon_rounds": cfg.lexicon_rounds,
                "train_rounds_per_pair": cfg.train_rounds_per_pair,
                "training_reps": cfg.training_reps,
                "n_train_sentences": cfg.n_train_sentences,
                "lexicon_readout_rounds": cfg.lexicon_readout_rounds,
                "n_seeds": n_seeds,
                "n_nouns": len(NOUNS),
                "n_verbs": len(VERBS),
                "n_novel": len(NOVEL_NOUNS),
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="N400 Prediction Error Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = PredictionN400Experiment(verbose=True)

    if args.quick:
        cfg = N400Config(
            n=5000, k=50, training_reps=3, n_train_sentences=15)
        n_seeds = args.seeds or 5
    else:
        cfg = N400Config()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("N400 PREDICTION ERROR SUMMARY")
    print("=" * 70)

    m = result.metrics
    print(f"\nWord-specific N400 (prediction error at object position):")
    print(f"  Grammatical: {m['n400_gram']['mean']:.4f} "
          f"+/- {m['n400_gram']['sem']:.4f}")
    print(f"  CatViol:     {m['n400_catviol']['mean']:.4f} "
          f"+/- {m['n400_catviol']['sem']:.4f}")
    print(f"  Novel:       {m['n400_novel']['mean']:.4f} "
          f"+/- {m['n400_novel']['sem']:.4f}")

    print(f"\n  H1 CatViol > Gram: "
          f"d={m['h1_catviol_vs_gram']['d']:.2f}, "
          f"p={m['h1_catviol_vs_gram']['p']:.4f}")
    print(f"  H2 Novel > Gram:   "
          f"d={m['h2_novel_vs_gram']['d']:.2f}, "
          f"p={m['h2_novel_vs_gram']['p']:.4f}")
    print(f"     CatViol > Novel: "
          f"d={m['h2_catviol_vs_novel']['d']:.2f}, "
          f"p={m['h2_catviol_vs_novel']['p']:.4f}")

    print(f"\nCategory-match N400 (max overlap with same-category refs):")
    print(f"  Grammatical: {m['n400_cat_gram']['mean']:.4f} "
          f"+/- {m['n400_cat_gram']['sem']:.4f}")
    print(f"  CatViol:     {m['n400_cat_catviol']['mean']:.4f} "
          f"+/- {m['n400_cat_catviol']['sem']:.4f}")
    print(f"  Novel:       {m['n400_cat_novel']['mean']:.4f} "
          f"+/- {m['n400_cat_novel']['sem']:.4f}")

    print(f"\n  CatViol > Gram: "
          f"d={m['h1_cat_catviol_vs_gram']['d']:.2f}, "
          f"p={m['h1_cat_catviol_vs_gram']['p']:.4f}")
    print(f"  Novel > Gram:   "
          f"d={m['h2_cat_novel_vs_gram']['d']:.2f}, "
          f"p={m['h2_cat_novel_vs_gram']['p']:.4f}")

    print(f"\nVerb position (control): {m['n400_verb_pos']['mean']:.4f} "
          f"+/- {m['n400_verb_pos']['sem']:.4f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
