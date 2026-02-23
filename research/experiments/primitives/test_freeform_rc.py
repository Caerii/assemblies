"""
Free-Form Learner with Right-Branching Relative Clauses: 5-Category Discovery

Tests whether the FreeFormLearner can discover 5 lexical categories
(DET, ADJ, NOUN, VERB, COMP) when right-branching relative clauses are
present. This is the first test of clause boundary detection and recursive
structure in the free-form learner.

Right-branching RC example:
  "the big dog chases the small cat that eats the old bird"
  DET  ADJ N   V      DET  ADJ N   COMP V    DET  ADJ N

Uses a 3-phase staged developmental curriculum:
  1. Phase 1 (warmup): mandatory adjectives, no RC.
     Establishes the 4-way split (DET/ADJ/NOUN/VERB) in CORE_0-3.
  2. Phase 2 (RC introduction): mandatory adjectives + mandatory RC.
     Introduces "that" which claims the unused CORE_4.
  3. Phase 3 (consolidation): mixed sentences (50% RC, 50% no-RC).
     Tests whether the 5-way split is maintained with variable structure.

Key mechanism: the bootstrap cycle wraps at min(n_core_areas, refractory+1),
so with 5 areas and refractory=3, the 4-position cycle leaves CORE_4 empty.
The unused-area preference then assigns "that" to CORE_4 when it first appears.

Hypotheses:
  H1: Word assemblies stable after formation (overlap > 0.8)
  H2: Cumulative N400 separates grammatical from scrambled (d > 1.0)
  H3: Binding consistency > 70% (same word -> same STRUCT area)
  H4: Prediction improves with exposure (N400 decreases over time)
  H5: Generalization to novel word combinations
  H6: 5-category discovery accuracy > 90%

Usage:
    uv run python research/experiments/primitives/test_freeform_rc.py
    uv run python research/experiments/primitives/test_freeform_rc.py --quick
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
from research.experiments.lib.vocabulary import RC_VOCAB
from research.experiments.lib.grammar import AdjCFG, RCCFG
from research.experiments.lib.freeform import FreeFormConfig, FreeFormLearner


@dataclass
class RCExperimentConfig:
    # Learner params
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    n_core_areas: int = 5
    core_refractory_period: int = 3
    n_struct_areas: int = 15
    struct_refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    lexicon_refresh_interval: int = 50
    use_p600_feedback: bool = False
    engine: str = "auto"
    # Grammar params
    rc_prob: float = 0.5
    pp_prob: float = 0.0
    # Curriculum params
    warmup_adj_sentences: int = 20
    warmup_rc_sentences: int = 30
    # Experiment params
    n_train_sentences: int = 400
    checkpoint_interval: int = 50
    n_test_grammatical: int = 30
    n_test_scrambled: int = 30


def make_learner_config(cfg: RCExperimentConfig) -> FreeFormConfig:
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


def count_category_accuracy(stats: Dict[str, Any], vocab) -> Dict[str, Any]:
    """Count how many words are correctly categorized per CORE area."""
    assignments = stats.get("word_assignments", {})
    correct = 0
    total = 0
    misrouted = []
    expected = {
        "DET": "CORE_0", "ADJ": "CORE_1",
        "NOUN": "CORE_2", "LOCATION": "CORE_2",
        "VERB": "CORE_3", "PREP": "CORE_3",
        "COMP": "CORE_4",
    }
    for word, area in assignments.items():
        word_str = str(word)
        total += 1
        cat = vocab.category_for_word(word_str)
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


def run_trial(cfg: RCExperimentConfig, seed: int) -> Dict[str, Any]:
    """Run one trial of free-form learning with right-branching RCs."""
    rng = np.random.default_rng(seed)
    vocab = RC_VOCAB

    # Create learner
    lcfg = make_learner_config(cfg)
    learner = FreeFormLearner(lcfg, seed=seed)

    # --- Curriculum ---
    # Phase 1: mandatory adjectives, no RC (establishes 4 categories)
    warmup_adj_grammar = AdjCFG(pp_prob=cfg.pp_prob, vocab=vocab, rng=rng)
    phase1 = warmup_adj_grammar.generate_batch(cfg.warmup_adj_sentences)

    # Phase 2: mandatory adjectives + mandatory RC (introduces COMP)
    warmup_rc_grammar = RCCFG(
        pp_prob=cfg.pp_prob, rc_prob=1.0, vocab=vocab, rng=rng,
    )
    phase2 = warmup_rc_grammar.generate_batch(cfg.warmup_rc_sentences)

    # Phase 3: mixed (some RC, some not)
    main_grammar = RCCFG(
        pp_prob=cfg.pp_prob, rc_prob=cfg.rc_prob, vocab=vocab, rng=rng,
    )
    n_main = cfg.n_train_sentences - cfg.warmup_adj_sentences - cfg.warmup_rc_sentences
    phase3 = main_grammar.generate_batch(max(0, n_main))

    curriculum = phase1 + phase2 + phase3

    # --- Training with checkpoints ---
    checkpoints = []
    for i, sent in enumerate(curriculum, 1):
        words = [str(w) for w in sent["words"]]
        learner.process_sentence(words)

        if i % cfg.checkpoint_interval == 0:
            learner._refresh_lexicon()
            # Test at checkpoint
            test_rng = np.random.default_rng(seed + 10000 + i)
            test_grammar = RCCFG(
                pp_prob=cfg.pp_prob, rc_prob=cfg.rc_prob, vocab=vocab,
                rng=test_rng,
            )
            gram_scores = []
            scram_scores = []
            for _ in range(10):
                test_sent = test_grammar.generate()
                test_words = [str(w) for w in test_sent["words"]]
                g = learner.measure_sentence_acceptability(test_words)
                gram_scores.append(g)
                s = learner.measure_sentence_acceptability(
                    generate_scrambled(test_words, test_rng))
                scram_scores.append(s)
            stab_samples = []
            for w in list(learner.vocab.known_words)[:10]:
                stab_samples.append(learner.measure_assembly_stability(w))
            bind_samples = []
            for w in list(learner.vocab.known_words)[:10]:
                bind_samples.append(
                    learner.measure_binding_consistency(w, n_trials=5))
            n400_sep = np.mean(scram_scores) - np.mean(gram_scores)
            checkpoints.append({
                "sentence": i,
                "n400_sep": float(n400_sep),
                "stability": float(np.mean(stab_samples)),
                "binding": float(np.mean(bind_samples)),
            })

    # --- Final evaluation ---
    learner._refresh_lexicon()
    stats = learner.get_stats()
    cat_result = count_category_accuracy(stats, vocab)

    # N400 gram vs scrambled
    eval_rng = np.random.default_rng(seed + 99999)
    eval_grammar = RCCFG(
        pp_prob=cfg.pp_prob, rc_prob=cfg.rc_prob, vocab=vocab, rng=eval_rng,
    )
    gram_scores = []
    scram_scores = []
    for _ in range(cfg.n_test_grammatical):
        test_sent = eval_grammar.generate()
        test_words = [str(w) for w in test_sent["words"]]
        g = learner.measure_sentence_acceptability(test_words)
        gram_scores.append(g)
        s = learner.measure_sentence_acceptability(
            generate_scrambled(test_words, eval_rng))
        scram_scores.append(s)

    gram_arr = np.array(gram_scores)
    scram_arr = np.array(scram_scores)
    pooled_std = np.sqrt((np.var(gram_arr) + np.var(scram_arr)) / 2)
    n400_d = float((np.mean(scram_arr) - np.mean(gram_arr)) / pooled_std) \
        if pooled_std > 0 else 0.0

    # Stability
    stab_values = []
    for w in learner.vocab.known_words:
        stab_values.append(learner.measure_assembly_stability(w))
    stab_mean = float(np.mean(stab_values))

    # Binding
    bind_values = []
    for w in learner.vocab.known_words:
        bind_values.append(learner.measure_binding_consistency(w, n_trials=10))
    bind_mean = float(np.mean(bind_values))

    # Novel combinations
    novel_scores = []
    novel_grammar = RCCFG(
        pp_prob=cfg.pp_prob, rc_prob=cfg.rc_prob, vocab=vocab,
        rng=np.random.default_rng(seed + 77777),
    )
    for _ in range(cfg.n_test_grammatical):
        ns = novel_grammar.generate()
        nw = [str(w) for w in ns["words"]]
        novel_scores.append(learner.measure_sentence_acceptability(nw))
    novel_arr = np.array(novel_scores)
    novel_pooled = np.sqrt((np.var(novel_arr) + np.var(scram_arr)) / 2)
    novel_d = float((np.mean(scram_arr) - np.mean(novel_arr)) / novel_pooled) \
        if novel_pooled > 0 else 0.0

    # Prediction improvement
    if len(checkpoints) >= 2:
        early = checkpoints[0]["n400_sep"]
        late = checkpoints[-1]["n400_sep"]
        prediction_improved = late > early
    else:
        prediction_improved = False

    return {
        "seed": seed,
        "n_words": stats["n_words"],
        "stability": stab_mean,
        "n400_d": n400_d,
        "binding": bind_mean,
        "prediction_improved": prediction_improved,
        "novel_d": novel_d,
        "category_accuracy": cat_result["accuracy"],
        "category_correct": cat_result["correct"],
        "category_total": cat_result["total"],
        "category_misrouted": cat_result["misrouted"],
        "checkpoints": checkpoints,
        "word_assignments": stats["word_assignments"],
        "words_per_core_area": stats["words_per_core_area"],
    }


class FreeFormRCExperiment(ExperimentBase):
    """Free-form learner with right-branching relative clauses."""

    def __init__(self, cfg: RCExperimentConfig, n_seeds: int = 5,
                 quick: bool = False, tag: str = "freeform_rc"):
        super().__init__(tag)
        self.cfg = cfg
        self.n_seeds = n_seeds
        self.quick = quick
        self.tag = tag

    def run(self) -> ExperimentResult:
        self._start_timer()
        cfg = self.cfg
        tag = f"[{self.tag}]"

        print(f"{tag} " + "=" * 70)
        print(f"{tag} Free-Form Learner + Right-Branching RC: 5-Category Discovery")
        print(f"{tag}   n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        print(f"{tag}   rc_prob={cfg.rc_prob}, pp_prob={cfg.pp_prob}")
        print(f"{tag}   warmup_adj={cfg.warmup_adj_sentences}, "
              f"warmup_rc={cfg.warmup_rc_sentences}")
        print(f"{tag}   n_train={cfg.n_train_sentences}, "
              f"checkpoint_interval={cfg.checkpoint_interval}")
        print(f"{tag}   n_core_areas={cfg.n_core_areas}, "
              f"core_refractory={cfg.core_refractory_period}")
        print(f"{tag}   n_struct_areas={cfg.n_struct_areas}, "
              f"struct_refractory={cfg.struct_refractory_period}")
        print(f"{tag}   n_seeds={self.n_seeds}")
        print(f"{tag} " + "=" * 70)
        print()

        all_results = []
        for i, seed in enumerate(range(1, self.n_seeds + 1)):
            print(f"  Seed {i+1}/{self.n_seeds} ...")
            result = run_trial(cfg, seed)
            all_results.append(result)

            print(f"{tag}     Words: {result['n_words']}, "
                  f"Stab: {result['stability']:.3f}, "
                  f"N400 d: {result['n400_d']:.2f}, "
                  f"Bind: {result['binding']:.2f}, "
                  f"Novel d: {result['novel_d']:.2f}")
            ca = result["category_accuracy"]
            cc = result["category_correct"]
            ct = result["category_total"]
            print(f"{tag}     Category accuracy: "
                  f"{cc}/{ct} ({100*ca:.1f}%)")
            if result["category_misrouted"]:
                print(f"{tag}     Misrouted: "
                      f"{', '.join(result['category_misrouted'])}")
            print(f"{tag}     CORE assignments: "
                  f"{result['word_assignments']}")
            print(f"{tag}     Words per area: "
                  f"{result['words_per_core_area']}")
            for cp in result["checkpoints"]:
                print(f"{tag}       @ {cp['sentence']:3d}: "
                      f"n400_sep={cp['n400_sep']:+.3f}, "
                      f"stab={cp['stability']:.3f}, "
                      f"bind={cp['binding']:.2f}")
            print()

        # Aggregate
        stabs = [r["stability"] for r in all_results]
        n400s = [r["n400_d"] for r in all_results]
        binds = [r["binding"] for r in all_results]
        novels = [r["novel_d"] for r in all_results]
        cats = [r["category_accuracy"] for r in all_results]
        n_improved = sum(1 for r in all_results if r["prediction_improved"])
        n_words = all_results[0]["n_words"]

        stab_s = summarize(stabs)
        n400_s = summarize(n400s)
        bind_s = summarize(binds)
        novel_s = summarize(novels)
        cat_s = summarize(cats)
        stab_m, stab_se = stab_s["mean"], stab_s.get("sem", 0)
        n400_m, n400_se = n400_s["mean"], n400_s.get("sem", 0)
        bind_m, bind_se = bind_s["mean"], bind_s.get("sem", 0)
        novel_m, novel_se = novel_s["mean"], novel_s.get("sem", 0)
        cat_m, cat_se = cat_s["mean"], cat_s.get("sem", 0)

        print(f"  === Results (across {self.n_seeds} seeds) ===")
        print(f"{tag}     Assembly stability:     "
              f"{stab_m:.3f} +/- {stab_se:.3f}")
        print(f"{tag}     N400 gram vs scram d:   "
              f"{n400_m:.2f} +/- {n400_se:.2f}")
        print(f"{tag}     Binding consistency:     "
              f"{bind_m:.3f} +/- {bind_se:.3f}")
        print(f"{tag}     Prediction improved:     "
              f"{n_improved}/{self.n_seeds} seeds")
        print(f"{tag}     Novel combination d:     "
              f"{novel_m:.2f} +/- {novel_se:.2f}")
        print(f"{tag}     Category accuracy:       "
              f"{100*cat_m:.1f}% +/- {100*cat_se:.1f}%")
        print(f"{tag}     Words learned:           {n_words}")

        h1 = stab_m > 0.8
        h2 = n400_m > 1.0
        h3 = bind_m > 0.7
        h4 = n_improved > self.n_seeds // 2
        h5 = novel_m > 0.5
        h6 = cat_m > 0.9

        print()
        print(f"  === Hypotheses ===")
        print(f"{tag}     H1 (Stability > 0.8):    "
              f"{'PASS' if h1 else 'FAIL'} ({stab_m:.3f})")
        print(f"{tag}     H2 (N400 sep d > 1.0):   "
              f"{'PASS' if h2 else 'FAIL'} ({n400_m:.2f})")
        print(f"{tag}     H3 (Binding > 70%):      "
              f"{'PASS' if h3 else 'FAIL'} ({100*bind_m:.1f}%)")
        print(f"{tag}     H4 (Prediction improves): "
              f"{'PASS' if h4 else 'FAIL'} ({n_improved}/{self.n_seeds})")
        print(f"{tag}     H5 (Novel d > 0.5):      "
              f"{'PASS' if h5 else 'FAIL'} ({novel_m:.2f})")
        print(f"{tag}     H6 (Category acc > 90%):  "
              f"{'PASS' if h6 else 'FAIL'} ({100*cat_m:.1f}%)")

        metrics = {
            "stability_mean": stab_m,
            "stability_se": stab_se,
            "n400_d_mean": n400_m,
            "n400_d_se": n400_se,
            "binding_mean": bind_m,
            "binding_se": bind_se,
            "novel_d_mean": novel_m,
            "novel_d_se": novel_se,
            "prediction_improved_count": n_improved,
            "category_accuracy_mean": cat_m,
            "category_accuracy_se": cat_se,
            "n_words": n_words,
            "H1_stability": h1,
            "H2_n400_separation": h2,
            "H3_binding": h3,
            "H4_prediction_improves": h4,
            "H5_novel_generalization": h5,
            "H6_category_accuracy": h6,
        }
        duration = self._stop_timer()

        return ExperimentResult(
            experiment_name=self.tag,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_core_areas": cfg.n_core_areas,
                "core_refractory_period": cfg.core_refractory_period,
                "n_struct_areas": cfg.n_struct_areas,
                "n_train_sentences": cfg.n_train_sentences,
                "warmup_adj_sentences": cfg.warmup_adj_sentences,
                "warmup_rc_sentences": cfg.warmup_rc_sentences,
                "rc_prob": cfg.rc_prob,
                "pp_prob": cfg.pp_prob,
                "n_seeds": self.n_seeds,
                "quick": self.quick,
            },
            metrics=metrics,
            raw_data={"trials": all_results},
            duration_seconds=duration,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Free-form learner with right-branching RCs")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with smaller params")
    parser.add_argument("--engine", type=str, default="auto",
                        help="Engine: auto, numpy_sparse, cuda_implicit")
    args = parser.parse_args()

    if args.quick:
        cfg = RCExperimentConfig(
            n=5000, k=50,
            n_train_sentences=200,
            warmup_adj_sentences=20,
            warmup_rc_sentences=30,
            checkpoint_interval=50,
            engine=args.engine,
        )
        n_seeds = 3
    else:
        cfg = RCExperimentConfig(engine=args.engine)
        n_seeds = 5

    exp = FreeFormRCExperiment(
        cfg, n_seeds=n_seeds, quick=args.quick, tag="freeform_rc",
    )
    result = exp.run()
    suffix = "_quick" if args.quick else ""
    path = exp.save_result(result, suffix)

    print()
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"[freeform_rc] Results saved to {path}")

    # Summary
    print()
    print("=" * 70)
    print("FREE-FORM LEARNER + RIGHT-BRANCHING RC SUMMARY")
    print("=" * 70)
    print()
    m = result.metrics
    hyps = ["H1_stability", "H2_n400_separation", "H3_binding",
            "H4_prediction_improves", "H5_novel_generalization",
            "H6_category_accuracy"]
    for h in hyps:
        label = h.replace("_", " ").upper()
        print(f"  {label}: {'PASS' if m[h] else 'FAIL'}")
    print()
    print(f"  Category accuracy: {100*m['category_accuracy_mean']:.1f}%")
    print(f"  Duration: {result.duration_seconds:.1f}s")
