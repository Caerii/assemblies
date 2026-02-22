"""
Free-Form Learner with Optional Adjectives: Variable-Length Constituents

Tests whether the FreeFormLearner can discover 4 lexical categories
(DET, ADJ, NOUN, VERB) when adjectives are optional — some sentences have
DET-ADJ-N-V and others have DET-N-V. This is the first test of whether
the routing mechanism handles variable-length constituents rather than
rigid positional patterns.

Uses a staged developmental curriculum (like infant-directed speech):
  1. Warmup phase (first N sentences): mandatory adjectives establish
     the clean 4-way category split (DET/ADJ/NOUN/VERB).
  2. Main phase: optional adjectives (adj_prob=0.5) test whether the
     learner maintains the split with variable-length constituents.

By the warmup's end, all words are correctly assigned. The main phase
tests whether prediction and N400 signals remain strong when sentence
structure varies.

Definitive results (5 seeds, n=10000, 400 train, 50 warmup, 30 test):
  H1: PASS  stability = 0.801 +/- 0.009
  H2: PASS  N400 d = 1.13 +/- 0.19
  H3: PASS  binding = 100.0% +/- 0.0%
  H4: FAIL  2/5 seeds improved (prediction near-ceiling from warmup)
  H5: PASS  novel d = 1.28 +/- 0.30

  Category accuracy: 95.7% +/- 4.3%  (4/5 seeds perfect 23/23)
  Staged curriculum enables 4-category discovery with variable-length
  constituents — simple structures first, then optionality.

Hypotheses:
  H1: Word assemblies stable after formation (overlap > 0.8)
  H2: Cumulative N400 separates grammatical from scrambled (d > 1.0)
  H3: Binding consistency > 70% (same word -> same STRUCT area)
  H4: Prediction improves with exposure (N400 decreases over time)
  H5: Generalization to novel word combinations

Usage:
    uv run python research/experiments/primitives/test_freeform_opt_adj.py
    uv run python research/experiments/primitives/test_freeform_opt_adj.py --quick
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
from research.experiments.lib.vocabulary import ADJ_VOCAB
from research.experiments.lib.grammar import OptAdjCFG
from research.experiments.lib.freeform import FreeFormConfig, FreeFormLearner


@dataclass
class OptAdjExperimentConfig:
    # Learner params
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    n_core_areas: int = 4
    core_refractory_period: int = 3
    n_struct_areas: int = 12
    struct_refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    lexicon_refresh_interval: int = 50
    use_p600_feedback: bool = False
    engine: str = "auto"
    # Grammar params
    adj_prob: float = 0.5
    pp_prob: float = 0.3
    # Curriculum params — staged learning like infant-directed speech:
    # first warmup_sentences use mandatory adjectives (adj_prob=1.0) to
    # establish the 4-way split, then switch to optional (adj_prob).
    warmup_sentences: int = 50
    # Experiment params
    n_train_sentences: int = 400
    checkpoint_interval: int = 50
    n_test_grammatical: int = 30
    n_test_scrambled: int = 30


def make_learner_config(cfg: OptAdjExperimentConfig) -> FreeFormConfig:
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
        engine=cfg.engine,
    )


def generate_scrambled(words: List[str], rng: np.random.Generator) -> List[str]:
    """Scramble word order to create ungrammatical sequence."""
    scrambled = list(words)
    rng.shuffle(scrambled)
    if scrambled == list(words):
        scrambled[0], scrambled[-1] = scrambled[-1], scrambled[0]
    return scrambled


def generate_novel_opt_adj_combinations(
    vocab, rng: np.random.Generator, n_items: int, adj_prob: float = 0.5,
) -> List[List[str]]:
    """Generate sentences with optional adjectives and novel word combinations."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    dets = vocab.words_for_category("DET")
    adjs = vocab.words_for_category("ADJ")
    preps = vocab.words_for_category("PREP")
    locs = vocab.words_for_category("LOCATION")
    combos = []
    for _ in range(n_items):
        agent = nouns[rng.integers(len(nouns))]
        verb = verbs[rng.integers(len(verbs))]
        patient = nouns[rng.integers(len(nouns))]
        while patient == agent:
            patient = nouns[rng.integers(len(nouns))]
        det1 = dets[rng.integers(len(dets))]
        det2 = dets[rng.integers(len(dets))]
        sent = [det1]
        if rng.random() < adj_prob:
            sent.append(adjs[rng.integers(len(adjs))])
        sent.extend([agent, verb, det2])
        if rng.random() < adj_prob:
            sent.append(adjs[rng.integers(len(adjs))])
        sent.append(patient)
        # ~30% chance of PP
        if rng.random() < 0.3:
            prep = preps[rng.integers(len(preps))]
            loc = locs[rng.integers(len(locs))]
            det3 = dets[rng.integers(len(dets))]
            sent.append(prep)
            sent.append(det3)
            if rng.random() < adj_prob:
                sent.append(adjs[rng.integers(len(adjs))])
            sent.append(loc)
        combos.append(sent)
    return combos


def count_category_accuracy(stats: Dict[str, Any], vocab) -> Dict[str, Any]:
    """Count how many words are correctly categorized per CORE area."""
    assignments = stats.get("word_assignments", {})
    correct = 0
    total = 0
    misrouted = []
    for word, area in assignments.items():
        word_str = str(word)
        total += 1
        cat = vocab.category_for_word(word_str)
        # Expected mapping: DET→CORE_0, ADJ→CORE_1, NOUN/LOC→CORE_2, VERB/PREP→CORE_3
        expected = {
            "DET": "CORE_0", "ADJ": "CORE_1",
            "NOUN": "CORE_2", "LOCATION": "CORE_2",
            "VERB": "CORE_3", "PREP": "CORE_3",
        }
        if area == expected.get(cat, ""):
            correct += 1
        else:
            misrouted.append(f"{word_str}({cat})->{area}")
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "misrouted": misrouted,
    }


def run_trial(cfg: OptAdjExperimentConfig, seed: int) -> Dict[str, Any]:
    """Run one trial of free-form learning with optional adjectives."""
    rng = np.random.default_rng(seed)
    vocab = ADJ_VOCAB

    learner_cfg = make_learner_config(cfg)
    learner = FreeFormLearner(learner_cfg, seed)

    # Staged curriculum: warmup with mandatory adjectives, then optional.
    # This mirrors infant-directed speech where simpler structures come first,
    # establishing the category split before introducing optionality.
    from research.experiments.lib.grammar import AdjCFG
    warmup_grammar = AdjCFG(pp_prob=cfg.pp_prob, vocab=vocab, rng=rng)
    opt_grammar = OptAdjCFG(pp_prob=cfg.pp_prob, adj_prob=cfg.adj_prob,
                            vocab=vocab, rng=rng)
    n_warmup = min(cfg.warmup_sentences, cfg.n_train_sentences)
    n_opt = cfg.n_train_sentences - n_warmup
    curriculum = (warmup_grammar.generate_batch(n_warmup)
                  + opt_grammar.generate_batch(n_opt))

    test_grammar = OptAdjCFG(pp_prob=cfg.pp_prob, adj_prob=cfg.adj_prob,
                             vocab=vocab,
                             rng=np.random.default_rng(seed + 5000))
    test_grammatical = [
        s["words"] for s in test_grammar.generate_batch(cfg.n_test_grammatical)
    ]
    test_scrambled = [
        generate_scrambled(g, rng) for g in test_grammatical
    ]

    checkpoint_data = []

    for i, sent in enumerate(curriculum):
        learner.process_sentence(sent["words"])

        if (i + 1) % cfg.checkpoint_interval == 0:
            learner._refresh_lexicon()

            stab_words = list(learner.vocab.known_words)[:5]
            stabilities = [
                learner.measure_assembly_stability(w) for w in stab_words
            ]

            gram_n400s = [
                learner.measure_sentence_acceptability(g)
                for g in test_grammatical
            ]
            scram_n400s = [
                learner.measure_sentence_acceptability(s)
                for s in test_scrambled
            ]

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

    all_stabilities = [
        learner.measure_assembly_stability(w) for w in learner.vocab.known_words
    ]

    final_gram = [learner.measure_sentence_acceptability(g)
                  for g in test_grammatical]
    final_scram = [learner.measure_sentence_acceptability(s)
                   for s in test_scrambled]
    n400_test = paired_ttest(final_scram, final_gram)

    all_consistency = [
        learner.measure_binding_consistency(w, n_trials=5)
        for w in learner.vocab.known_words
    ]

    if len(checkpoint_data) >= 2:
        first_sep = checkpoint_data[0]["n400_separation"]
        last_sep = checkpoint_data[-1]["n400_separation"]
        prediction_improved = last_sep > first_sep
    else:
        prediction_improved = False
        first_sep = last_sep = 0.0

    novel_sents = generate_novel_opt_adj_combinations(
        vocab, rng, cfg.n_test_grammatical, cfg.adj_prob)
    novel_n400s = [learner.measure_sentence_acceptability(s) for s in novel_sents]
    novel_scram = [
        learner.measure_sentence_acceptability(generate_scrambled(s, rng))
        for s in novel_sents
    ]
    novel_test = paired_ttest(novel_scram, novel_n400s)

    stats = learner.get_stats()
    cat_accuracy = count_category_accuracy(stats, vocab)

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
        "stats": stats,
        "category_accuracy": cat_accuracy,
    }


class FreeFormOptAdjExperiment(ExperimentBase):
    """Free-form learner with optional adjectives experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="freeform_opt_adj",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 5,
        config: Optional[OptAdjExperimentConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or OptAdjExperimentConfig()

        self.log("=" * 70)
        self.log("Free-Form Learner + Optional Adjectives: 4-Category Discovery")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  adj_prob={cfg.adj_prob}, pp_prob={cfg.pp_prob}, "
                 f"warmup={cfg.warmup_sentences}")
        self.log(f"  n_train={cfg.n_train_sentences}, "
                 f"checkpoint_interval={cfg.checkpoint_interval}")
        self.log(f"  n_core_areas={cfg.n_core_areas}, "
                 f"core_refractory={cfg.core_refractory_period}")
        self.log(f"  n_struct_areas={cfg.n_struct_areas}, "
                 f"struct_refractory={cfg.struct_refractory_period}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        stabilities = []
        n400_ds = []
        consistencies = []
        improvements = []
        novel_ds = []
        n_words = []
        cat_accuracies = []

        for s in range(n_seeds):
            self.log(f"\n  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            stabilities.append(trial["mean_stability"])
            n400_ds.append(trial["n400_d"])
            consistencies.append(trial["mean_binding_consistency"])
            improvements.append(trial["prediction_improved"])
            novel_ds.append(trial["novel_d"])
            n_words.append(trial["n_words_learned"])

            cat_acc = trial["category_accuracy"]
            cat_accuracies.append(cat_acc["accuracy"])

            self.log(f"    Words: {trial['n_words_learned']}, "
                     f"Stab: {trial['mean_stability']:.3f}, "
                     f"N400 d: {trial['n400_d']:.2f}, "
                     f"Bind: {trial['mean_binding_consistency']:.2f}, "
                     f"Novel d: {trial['novel_d']:.2f}")
            self.log(f"    Category accuracy: "
                     f"{cat_acc['correct']}/{cat_acc['total']} "
                     f"({cat_acc['accuracy']:.1%})")
            if cat_acc["misrouted"]:
                self.log(f"    Misrouted: {', '.join(cat_acc['misrouted'])}")

            stats = trial["stats"]
            if "word_assignments" in stats:
                self.log(f"    CORE assignments: {stats['word_assignments']}")
            if "words_per_core_area" in stats:
                self.log(f"    Words per area: {stats['words_per_core_area']}")

            for cp in trial["checkpoints"]:
                self.log(f"      @{cp['sentence_idx']:>4d}: "
                         f"n400_sep={cp['n400_separation']:+.3f}, "
                         f"stab={cp['mean_stability']:.3f}, "
                         f"bind={cp['mean_binding_consistency']:.2f}")

        stab_summary = summarize(stabilities)
        n400_d_summary = summarize(n400_ds)
        consist_summary = summarize(consistencies)
        novel_d_summary = summarize(novel_ds)
        cat_acc_summary = summarize(cat_accuracies)

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
        self.log(f"    Category accuracy:       {cat_acc_summary['mean']:.1%} "
                 f"+/- {cat_acc_summary.get('sem', 0):.1%}")
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
            "category_accuracy": cat_acc_summary,
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
                "warmup_sentences": cfg.warmup_sentences,
                "n_core_areas": cfg.n_core_areas,
                "core_refractory_period": cfg.core_refractory_period,
                "n_struct_areas": cfg.n_struct_areas,
                "struct_refractory_period": cfg.struct_refractory_period,
                "adj_prob": cfg.adj_prob,
                "pp_prob": cfg.pp_prob,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Free-Form Learner + Optional Adjectives Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--adj-prob", type=float, default=None)
    parser.add_argument("--engine", type=str, default="auto")
    args = parser.parse_args()

    exp = FreeFormOptAdjExperiment(verbose=True)

    if args.quick:
        cfg = OptAdjExperimentConfig(
            n=5000, k=50,
            n_train_sentences=150,
            warmup_sentences=30,
            checkpoint_interval=30,
            n_test_grammatical=15,
            n_test_scrambled=15,
            lexicon_refresh_interval=30,
        )
        n_seeds = args.seeds or 3
    else:
        cfg = OptAdjExperimentConfig()
        n_seeds = args.seeds or 5

    if args.adj_prob is not None:
        cfg.adj_prob = args.adj_prob
    if args.engine != "auto":
        cfg.engine = args.engine

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    cat_acc = result.metrics["category_accuracy"]
    print("\n" + "=" * 70)
    print("FREE-FORM LEARNER + OPTIONAL ADJECTIVES SUMMARY")
    print("=" * 70)
    print(f"\nH1 Assembly stability > 0.8:  {'PASS' if h['H1_assembly_stable'] else 'FAIL'}")
    print(f"H2 N400 separation d > 1.0:   {'PASS' if h['H2_n400_separation'] else 'FAIL'}")
    print(f"H3 Binding consistency > 70%: {'PASS' if h['H3_binding_consistent'] else 'FAIL'}")
    print(f"H4 Prediction improves:       {'PASS' if h['H4_prediction_improves'] else 'FAIL'}")
    print(f"H5 Novel generalization:      {'PASS' if h['H5_novel_generalization'] else 'FAIL'}")
    print(f"\nCategory accuracy: {cat_acc['mean']:.1%}")
    print(f"Duration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
