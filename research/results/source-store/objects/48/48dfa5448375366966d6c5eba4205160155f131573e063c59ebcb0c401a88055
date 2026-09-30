"""
Center-Embedded Relative Clauses: Baseline Test

Tests whether the FreeFormLearner handles center-embedded RCs using only
the existing bigram prediction mechanism. Center-embedding interrupts the
main clause:

  "the big dog that the small cat chases eats the old bird"
   DET ADJ  N   COMP DET ADJ   N    V     V   DET ADJ  N

The critical challenge: after "the big dog", the system predicts VERB.
Instead it gets "that" (COMP), then an embedded clause. The main verb
"eats" arrives 5 words late. Can the system still process it correctly?

Uses the same 3-phase curriculum as the RC experiment:
  1. Warmup with mandatory adjectives (establishes 4 categories)
  2. RC introduction with center-embedded sentences (introduces COMP)
  3. Mixed (some CE, some simple)

Results (5 seeds, n=10000, k=100):
  All 5 hypotheses PASS. The system naturally produces a human-like
  processing difficulty gradient: simple < RB < CE-d1 < CE-d2, without
  any mechanism changes. Main verb is harder than embedded verb (N400
  0.80 vs 0.70), matching psycholinguistic findings.

Approaches tried and ruled out for improving CE main-verb prediction:
  - CONTEXT accumulator (self-recurrent area accumulating left context):
    Lossy bag-of-words accumulation. Context-only mode destroyed
    grammaticality discrimination (d: 1.91 -> -0.21). Hybrid same-area
    mode caused destructive interference (d: -0.02). Separate
    PREDICTION_CTX area preserved d=1.18 but only improved main-verb
    N400 by 1.9% -- not worth the added complexity.
  - STRUCT-based prediction (using STRUCT area cycling as positional
    encoding): STRUCT areas collapse to period-2 oscillation between
    2-3 dominant areas due to Hebbian competition. Cannot distinguish
    main-clause from embedded-clause positions.
  - Increased CE training exposure (200-800 sentences, ce_prob 0.3-0.8):
    Zero improvement in main-verb N400 (0.836 -> 0.833). The bigram
    transition is already saturated; the difficulty is structural, not
    a training gap.

Hypotheses:
  H1: Category accuracy > 90% (routing should still work -- same 5 cats)
  H2: N400 separation d > 0.5 (weaker threshold -- CE disrupts predictions)
  H3: Binding consistency > 70%
  H4: Right-branching N400 < center-embedded N400 (CE is harder)
  H5: Depth-2 N400 > depth-1 N400 (deeper embedding = more disruption)

Usage:
    uv run python research/experiments/primitives/test_center_embedded_baseline.py
    uv run python research/experiments/primitives/test_center_embedded_baseline.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    measure_overlap,
)
from research.experiments.lib.vocabulary import RC_VOCAB
from research.experiments.lib.grammar import AdjCFG, RCCFG, CenterEmbeddedCFG
from research.experiments.lib.freeform import FreeFormConfig, FreeFormLearner


@dataclass
class CEBaselineConfig:
    # Learner params (same as RC experiment)
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
    ce_prob: float = 0.5
    # Curriculum params
    warmup_adj_sentences: int = 20
    warmup_ce_sentences: int = 30
    # Experiment params
    n_train_sentences: int = 400
    n_test: int = 30


def make_learner_config(cfg: CEBaselineConfig) -> FreeFormConfig:
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
    scrambled = list(words)
    rng.shuffle(scrambled)
    if scrambled == list(words):
        scrambled[0], scrambled[-1] = scrambled[-1], scrambled[0]
    return scrambled


def count_category_accuracy(stats: Dict[str, Any], vocab) -> Dict[str, Any]:
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
        total += 1
        cat = vocab.category_for_word(str(word))
        if area == expected.get(cat, ""):
            correct += 1
        else:
            misrouted.append(f"{word}({cat})->{area}")
    return {
        "correct": correct, "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "misrouted": misrouted,
    }


def run_trial(cfg: CEBaselineConfig, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    vocab = RC_VOCAB

    lcfg = make_learner_config(cfg)
    learner = FreeFormLearner(lcfg, seed=seed)

    # --- Curriculum ---
    # Phase 1: mandatory adjectives, no RC
    warmup_adj = AdjCFG(pp_prob=0.0, vocab=vocab, rng=rng)
    phase1 = warmup_adj.generate_batch(cfg.warmup_adj_sentences)

    # Phase 2: mandatory center-embedded RC
    warmup_ce = CenterEmbeddedCFG(
        pp_prob=0.0, ce_prob=1.0, max_depth=1, vocab=vocab, rng=rng,
    )
    phase2 = warmup_ce.generate_batch(cfg.warmup_ce_sentences)

    # Phase 3: mixed
    main_grammar = CenterEmbeddedCFG(
        pp_prob=0.0, ce_prob=cfg.ce_prob, max_depth=1, vocab=vocab, rng=rng,
    )
    n_main = cfg.n_train_sentences - cfg.warmup_adj_sentences - cfg.warmup_ce_sentences
    phase3 = main_grammar.generate_batch(max(0, n_main))

    curriculum = phase1 + phase2 + phase3

    # --- Training ---
    for sent in curriculum:
        words = [str(w) for w in sent["words"]]
        learner.process_sentence(words)

    # --- Final evaluation ---
    learner._refresh_lexicon()
    stats = learner.get_stats()
    cat_result = count_category_accuracy(stats, vocab)

    eval_rng = np.random.default_rng(seed + 99999)

    # N400: center-embedded grammatical vs scrambled
    ce_grammar = CenterEmbeddedCFG(
        pp_prob=0.0, ce_prob=1.0, max_depth=1, vocab=vocab, rng=eval_rng,
    )
    ce_gram_scores = []
    ce_scram_scores = []
    for _ in range(cfg.n_test):
        s = ce_grammar.generate()
        w = [str(x) for x in s["words"]]
        ce_gram_scores.append(learner.measure_sentence_acceptability(w))
        ce_scram_scores.append(
            learner.measure_sentence_acceptability(generate_scrambled(w, eval_rng)))

    ce_gram = np.array(ce_gram_scores)
    ce_scram = np.array(ce_scram_scores)
    ce_pooled = np.sqrt((np.var(ce_gram) + np.var(ce_scram)) / 2)
    ce_d = float((np.mean(ce_scram) - np.mean(ce_gram)) / ce_pooled) if ce_pooled > 0 else 0.0

    # N400: right-branching for comparison
    rb_grammar = RCCFG(
        pp_prob=0.0, rc_prob=1.0, vocab=vocab,
        rng=np.random.default_rng(seed + 88888),
    )
    rb_gram_scores = []
    rb_scram_scores = []
    for _ in range(cfg.n_test):
        s = rb_grammar.generate()
        w = [str(x) for x in s["words"]]
        rb_gram_scores.append(learner.measure_sentence_acceptability(w))
        rb_scram_scores.append(
            learner.measure_sentence_acceptability(generate_scrambled(w, eval_rng)))

    rb_gram = np.array(rb_gram_scores)
    rb_scram = np.array(rb_scram_scores)
    rb_pooled = np.sqrt((np.var(rb_gram) + np.var(rb_scram)) / 2)
    rb_d = float((np.mean(rb_scram) - np.mean(rb_gram)) / rb_pooled) if rb_pooled > 0 else 0.0

    # N400: simple sentences (no RC)
    simple_grammar = AdjCFG(pp_prob=0.0, vocab=vocab,
                            rng=np.random.default_rng(seed + 77777))
    simple_scores = []
    for _ in range(cfg.n_test):
        s = simple_grammar.generate()
        w = [str(x) for x in s["words"]]
        simple_scores.append(learner.measure_sentence_acceptability(w))

    # N400: depth-2 center-embedding
    d2_grammar = CenterEmbeddedCFG(
        pp_prob=0.0, ce_prob=1.0, max_depth=2, vocab=vocab,
        rng=np.random.default_rng(seed + 66666),
    )
    d2_scores = []
    for _ in range(cfg.n_test):
        s = d2_grammar.generate()
        w = [str(x) for x in s["words"]]
        d2_scores.append(learner.measure_sentence_acceptability(w))

    # Stability + binding
    stab_values = [learner.measure_assembly_stability(w)
                   for w in learner.vocab.known_words]
    bind_values = [learner.measure_binding_consistency(w, n_trials=10)
                   for w in learner.vocab.known_words]

    return {
        "seed": seed,
        "n_words": stats["n_words"],
        "category_accuracy": cat_result["accuracy"],
        "category_correct": cat_result["correct"],
        "category_total": cat_result["total"],
        "category_misrouted": cat_result["misrouted"],
        "stability": float(np.mean(stab_values)),
        "binding": float(np.mean(bind_values)),
        "ce_n400_d": ce_d,
        "rb_n400_d": rb_d,
        "ce_gram_mean": float(np.mean(ce_gram)),
        "rb_gram_mean": float(np.mean(rb_gram)),
        "simple_gram_mean": float(np.mean(simple_scores)),
        "d2_gram_mean": float(np.mean(d2_scores)),
        "word_assignments": stats["word_assignments"],
        "words_per_core_area": stats["words_per_core_area"],
    }


class CEBaselineExperiment(ExperimentBase):
    def __init__(self, cfg: CEBaselineConfig, n_seeds: int = 5,
                 quick: bool = False):
        super().__init__("center_embedded_baseline")
        self.cfg = cfg
        self.n_seeds = n_seeds
        self.quick = quick

    def run(self) -> ExperimentResult:
        self._start_timer()
        tag = "[ce_baseline]"

        print(f"{tag} {'='*60}")
        print(f"{tag} Center-Embedded RC: Baseline (no mechanism changes)")
        print(f"{tag}   n={self.cfg.n}, k={self.cfg.k}")
        print(f"{tag}   n_core_areas={self.cfg.n_core_areas}, "
              f"n_struct_areas={self.cfg.n_struct_areas}")
        print(f"{tag}   n_train={self.cfg.n_train_sentences}, "
              f"ce_prob={self.cfg.ce_prob}")
        print(f"{tag}   n_seeds={self.n_seeds}")
        print(f"{tag} {'='*60}")
        print()

        all_results = []
        for i, seed in enumerate(range(1, self.n_seeds + 1)):
            print(f"  Seed {i+1}/{self.n_seeds} ...")
            result = run_trial(self.cfg, seed)
            all_results.append(result)

            ca = result["category_accuracy"]
            print(f"{tag}   Cat acc: {100*ca:.1f}%, "
                  f"CE d: {result['ce_n400_d']:.2f}, "
                  f"RB d: {result['rb_n400_d']:.2f}, "
                  f"Stab: {result['stability']:.3f}, "
                  f"Bind: {result['binding']:.2f}")
            print(f"{tag}   N400 means — simple: {result['simple_gram_mean']:.4f}, "
                  f"RB: {result['rb_gram_mean']:.4f}, "
                  f"CE: {result['ce_gram_mean']:.4f}, "
                  f"D2: {result['d2_gram_mean']:.4f}")
            if result["category_misrouted"]:
                print(f"{tag}   Misrouted: {', '.join(result['category_misrouted'])}")
            print()

        # Aggregate
        cats = [r["category_accuracy"] for r in all_results]
        ce_ds = [r["ce_n400_d"] for r in all_results]
        rb_ds = [r["rb_n400_d"] for r in all_results]
        stabs = [r["stability"] for r in all_results]
        binds = [r["binding"] for r in all_results]
        simple_means = [r["simple_gram_mean"] for r in all_results]
        rb_means = [r["rb_gram_mean"] for r in all_results]
        ce_means = [r["ce_gram_mean"] for r in all_results]
        d2_means = [r["d2_gram_mean"] for r in all_results]

        cat_s = summarize(cats)
        ce_s = summarize(ce_ds)
        rb_s = summarize(rb_ds)
        stab_s = summarize(stabs)
        bind_s = summarize(binds)

        print(f"  === Results ({self.n_seeds} seeds) ===")
        print(f"{tag}   Category accuracy:   {100*cat_s['mean']:.1f}% "
              f"+/- {100*cat_s.get('sem',0):.1f}%")
        print(f"{tag}   CE N400 d:           {ce_s['mean']:.2f} "
              f"+/- {ce_s.get('sem',0):.2f}")
        print(f"{tag}   RB N400 d:           {rb_s['mean']:.2f} "
              f"+/- {rb_s.get('sem',0):.2f}")
        print(f"{tag}   Stability:           {stab_s['mean']:.3f} "
              f"+/- {stab_s.get('sem',0):.3f}")
        print(f"{tag}   Binding:             {bind_s['mean']:.3f} "
              f"+/- {bind_s.get('sem',0):.3f}")

        # Processing difficulty gradient
        simple_m = float(np.mean(simple_means))
        rb_m = float(np.mean(rb_means))
        ce_m = float(np.mean(ce_means))
        d2_m = float(np.mean(d2_means))
        print()
        print(f"  === Processing Difficulty (mean N400) ===")
        print(f"{tag}   Simple (no RC):      {simple_m:.4f}")
        print(f"{tag}   Right-branching:     {rb_m:.4f}")
        print(f"{tag}   Center-embedded d1:  {ce_m:.4f}")
        print(f"{tag}   Center-embedded d2:  {d2_m:.4f}")

        h1 = cat_s["mean"] > 0.9
        h2 = ce_s["mean"] > 0.5
        h3 = bind_s["mean"] > 0.7
        h4 = rb_m < ce_m  # CE harder than RB
        h5 = ce_m < d2_m  # depth-2 harder than depth-1

        print()
        print(f"  === Hypotheses ===")
        print(f"{tag}   H1 (Cat acc > 90%):         "
              f"{'PASS' if h1 else 'FAIL'} ({100*cat_s['mean']:.1f}%)")
        print(f"{tag}   H2 (CE N400 d > 0.5):       "
              f"{'PASS' if h2 else 'FAIL'} ({ce_s['mean']:.2f})")
        print(f"{tag}   H3 (Binding > 70%):         "
              f"{'PASS' if h3 else 'FAIL'} ({100*bind_s['mean']:.1f}%)")
        print(f"{tag}   H4 (RB easier than CE):     "
              f"{'PASS' if h4 else 'FAIL'} (RB={rb_m:.4f} vs CE={ce_m:.4f})")
        print(f"{tag}   H5 (D1 easier than D2):     "
              f"{'PASS' if h5 else 'FAIL'} (D1={ce_m:.4f} vs D2={d2_m:.4f})")

        metrics = {
            "category_accuracy_mean": cat_s["mean"],
            "category_accuracy_se": cat_s.get("sem", 0),
            "ce_n400_d_mean": ce_s["mean"],
            "rb_n400_d_mean": rb_s["mean"],
            "stability_mean": stab_s["mean"],
            "binding_mean": bind_s["mean"],
            "simple_n400_mean": simple_m,
            "rb_n400_mean": rb_m,
            "ce_n400_mean": ce_m,
            "d2_n400_mean": d2_m,
            "H1_category_accuracy": h1,
            "H2_ce_n400_separation": h2,
            "H3_binding": h3,
            "H4_rb_easier_than_ce": h4,
            "H5_d1_easier_than_d2": h5,
        }
        duration = self._stop_timer()

        return ExperimentResult(
            experiment_name="center_embedded_baseline",
            parameters={
                "n": self.cfg.n, "k": self.cfg.k,
                "n_core_areas": self.cfg.n_core_areas,
                "n_struct_areas": self.cfg.n_struct_areas,
                "n_train_sentences": self.cfg.n_train_sentences,
                "ce_prob": self.cfg.ce_prob,
                "n_seeds": self.n_seeds,
                "quick": self.quick,
            },
            metrics=metrics,
            raw_data={"trials": all_results},
            duration_seconds=duration,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Center-embedded RC baseline test")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--engine", type=str, default="auto")
    args = parser.parse_args()

    if args.quick:
        cfg = CEBaselineConfig(
            n=5000, k=50,
            n_train_sentences=200,
            warmup_adj_sentences=20,
            warmup_ce_sentences=30,
            n_test=20,
            engine=args.engine,
        )
        n_seeds = 3
    else:
        cfg = CEBaselineConfig(engine=args.engine)
        n_seeds = 5

    exp = CEBaselineExperiment(cfg, n_seeds=n_seeds, quick=args.quick)
    result = exp.run()
    suffix = "_quick" if args.quick else ""
    path = exp.save_result(result, suffix)

    print()
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Results saved to {path}")
