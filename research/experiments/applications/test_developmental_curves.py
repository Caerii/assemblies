"""
Developmental Acquisition Curves Experiment (Tier 3)

Tests whether assembly calculus acquisition curves match child language
development patterns. Compares the emergent parser's per-category learning
trajectories to known psycholinguistic facts about first-language acquisition.

Scientific Question:
    Do assembly calculus acquisition curves match child language
    development patterns?

Hypotheses:
    H1: Nouns are acquired before verbs (higher accuracy at fewer
        training examples).
    H2: Content words (NOUN, VERB, ADJ) are acquired before function
        words (DET, PREP).
    H3: The vocabulary spurt occurs at a predictable training threshold
        (steepest accuracy gain between consecutive increments).
    H4: Classification accuracy follows a sigmoid growth curve
        (R-squared of sigmoid fit > 0.9).

Protocol:
    1. Create parser with full vocabulary from build_vocabulary().
    2. Train incrementally: expose to 1, 2, 5, 10, 20, 50, 100 sentences.
    3. After each increment, measure per-category classification accuracy.
    4. Plot learning curves: accuracy vs training exposure for each POS.
    5. Compare acquisition order to AoA data in lexicon.
    6. Fit sigmoid to each category's learning curve.

Statistical methodology:
    - n_seeds=5 independent random seeds per condition.
    - Sigmoid curve fitting via scipy.optimize.curve_fit.
    - Mean +/- SEM, Cohen's d effect sizes.
    - Spearman rank correlation for acquisition order vs AoA.

References:
    - Mitropolsky & Papadimitriou (2025). "Simulated Language Acquisition."
    - Papadimitriou et al., PNAS 117(25):14464-14472, 2020.
    - Bates et al. (1994). "Developmental and stylistic variation in the
      composition of early vocabulary." J. Child Language, 21(1), 85-123.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats as scipy_stats

from research.experiments.base import ExperimentBase, ExperimentResult, summarize, ttest_vs_null
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence, generate_training_sentences
from src.assembly_calculus.emergent.vocabulary_builder import build_vocabulary


@dataclass
class DevelopmentalConfig:
    """Configuration for developmental curves experiment."""
    n: int = 10000
    k: int = 100
    n_seeds: int = 5
    rounds: int = 10
    p: float = 0.05
    beta: float = 0.1
    training_increments: Tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)


# -- POS category groupings -------------------------------------------------

CONTENT_CATEGORIES = {"NOUN", "VERB", "ADJ"}
FUNCTION_CATEGORIES = {"DET", "PREP"}
ALL_TEST_CATEGORIES = {"NOUN", "VERB", "ADJ", "ADV", "DET", "PREP", "PRON"}

# Approximate mean AoA per category (from lexicon data, in years)
# Used for H1/H2 acquisition order comparison
CATEGORY_MEAN_AOA = {
    "NOUN": 2.0,
    "VERB": 2.5,
    "ADJ": 3.0,
    "ADV": 3.5,
    "PRON": 2.5,
    "DET": 2.0,
    "PREP": 3.0,
}


def _sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Sigmoid function: L / (1 + exp(-k*(x - x0))) + b."""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def _fit_sigmoid(x_data: List[float], y_data: List[float]) -> Dict[str, Any]:
    """Fit a sigmoid curve to learning data.

    Returns fit parameters, R-squared, and whether the fit succeeded.
    """
    from scipy.optimize import curve_fit

    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    if len(x) < 4 or np.std(y) < 1e-6:
        return {"success": False, "r_squared": 0.0, "params": {}}

    try:
        # Initial guesses: L=max(y), k=0.1, x0=median(x), b=min(y)
        p0 = [max(y) - min(y), 0.1, float(np.median(x)), min(y)]
        bounds = ([0, 0, 0, -1], [2.0, 10.0, max(x) * 2, 1.0])
        popt, _ = curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds,
                            maxfev=5000)
        y_pred = _sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
        return {
            "success": True,
            "r_squared": float(r_squared),
            "params": {
                "L": float(popt[0]),
                "k": float(popt[1]),
                "x0": float(popt[2]),
                "b": float(popt[3]),
            },
        }
    except (RuntimeError, ValueError):
        return {"success": False, "r_squared": 0.0, "params": {}}


def _build_category_vocab_map(
    vocab: Dict[str, GroundingContext],
) -> Dict[str, List[str]]:
    """Map each POS category to its vocabulary words.

    Uses dominant_modality to infer category.
    """
    modality_to_cat = {
        "visual": "NOUN",
        "motor": "VERB",
        "properties": "ADJ",
        "spatial": "PREP",
        "social": "PRON",
        "temporal": "ADV",
        "none": "DET",
    }
    cat_words: Dict[str, List[str]] = {c: [] for c in ALL_TEST_CATEGORIES}
    for word, ctx in vocab.items():
        cat = modality_to_cat.get(ctx.dominant_modality)
        if cat and cat in cat_words:
            cat_words[cat].append(word)
    return cat_words


def _evaluate_per_category(
    parser: EmergentParser,
    cat_words: Dict[str, List[str]],
) -> Dict[str, float]:
    """Evaluate classification accuracy per POS category."""
    results = {}
    for cat, words in cat_words.items():
        if not words:
            results[cat] = 0.0
            continue
        correct = 0
        for word in words:
            grounding = parser.word_grounding.get(word)
            predicted, _ = parser.classify_word(word, grounding=grounding)
            if predicted == cat:
                correct += 1
        results[cat] = correct / len(words)
    return results


def _find_spurt_threshold(
    increments: List[int],
    accuracies: List[float],
) -> Dict[str, Any]:
    """Find the training increment with the steepest accuracy gain.

    The 'vocabulary spurt' is the increment where accuracy growth rate
    (delta_accuracy / delta_sentences) is maximized.
    """
    if len(increments) < 2:
        return {"threshold": 0, "max_gain": 0.0}

    max_gain = -1.0
    threshold_idx = 0
    for i in range(1, len(increments)):
        delta_acc = accuracies[i] - accuracies[i - 1]
        delta_sent = increments[i] - increments[i - 1]
        gain = delta_acc / max(delta_sent, 1)
        if gain > max_gain:
            max_gain = gain
            threshold_idx = i

    return {
        "threshold_sentences": increments[threshold_idx],
        "max_gain_rate": float(max_gain),
        "threshold_index": threshold_idx,
    }


# -- Experiment class -------------------------------------------------------


class DevelopmentalCurvesExperiment(ExperimentBase):
    """Test whether parser acquisition curves match child language patterns."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="developmental_curves",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "applications"),
            verbose=verbose,
        )

    def run(self, n_seeds: int = 5, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = DevelopmentalConfig(n_seeds=n_seeds, **{
            k: v for k, v in kwargs.items()
            if k in ("n", "k", "rounds", "p", "beta", "training_increments")
        })

        self.log("=" * 60)
        self.log("Developmental Acquisition Curves Experiment")
        self.log(f"  n={cfg.n}, k={cfg.k}, n_seeds={cfg.n_seeds}")
        self.log(f"  increments={cfg.training_increments}")
        self.log("=" * 60)

        seeds = [self.seed + i for i in range(cfg.n_seeds)]
        metrics: Dict[str, Any] = {}

        # Build full vocabulary
        vocab = build_vocabulary()
        cat_words = _build_category_vocab_map(vocab)
        self.log(f"\nVocabulary: {len(vocab)} words")
        for cat, words in sorted(cat_words.items()):
            self.log(f"  {cat}: {len(words)} words")

        # Generate a large pool of training sentences
        all_sentences = generate_training_sentences(vocab, n_sentences=200, seed=42)
        self.log(f"Training pool: {len(all_sentences)} sentences")

        # ==============================================================
        # Main loop: incremental training and per-category evaluation
        # ==============================================================
        self.log("\nIncremental training:")

        # Per-category, per-increment, per-seed accuracies
        # Shape conceptually: {category: {increment: [accuracy_per_seed]}}
        learning_curves: Dict[str, Dict[int, List[float]]] = {
            cat: {inc: [] for inc in cfg.training_increments}
            for cat in ALL_TEST_CATEGORIES
        }

        for s in seeds:
            self.log(f"\n  Seed {s}:")
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=s, rounds=cfg.rounds, vocabulary=vocab,
            )

            for inc in cfg.training_increments:
                # Take the first `inc` sentences from the pool
                subset = all_sentences[:inc]

                # Train lexicon and roles on the subset
                parser.train_lexicon()
                parser.train_roles(subset)

                # Evaluate per-category classification
                per_cat = _evaluate_per_category(parser, cat_words)
                for cat in ALL_TEST_CATEGORIES:
                    learning_curves[cat][inc].append(per_cat.get(cat, 0.0))

                overall = np.mean([per_cat.get(c, 0.0)
                                   for c in ALL_TEST_CATEGORIES])
                self.log(f"    inc={inc:3d}: overall={overall:.3f}  "
                         f"N={per_cat.get('NOUN', 0):.2f}  "
                         f"V={per_cat.get('VERB', 0):.2f}  "
                         f"D={per_cat.get('DET', 0):.2f}")

        # ==============================================================
        # Compute summary statistics across seeds
        # ==============================================================
        curve_summaries: Dict[str, List[Dict[str, Any]]] = {}
        for cat in ALL_TEST_CATEGORIES:
            cat_summary = []
            for inc in cfg.training_increments:
                values = learning_curves[cat][inc]
                cat_summary.append({
                    "increment": inc,
                    "stats": summarize(values),
                })
            curve_summaries[cat] = cat_summary

        metrics["learning_curves"] = {
            cat: [{"increment": entry["increment"],
                   "mean": entry["stats"]["mean"],
                   "sem": entry["stats"]["sem"]}
                  for entry in entries]
            for cat, entries in curve_summaries.items()
        }

        # ==============================================================
        # H1: Nouns acquired before verbs
        # ==============================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Nouns acquired before verbs")

        # Compare accuracy at each increment: nouns vs verbs
        h1_noun_advantage = []
        for inc in cfg.training_increments:
            noun_vals = learning_curves["NOUN"][inc]
            verb_vals = learning_curves["VERB"][inc]
            # Mean advantage per seed at this increment
            advantages = [n - v for n, v in zip(noun_vals, verb_vals)]
            h1_noun_advantage.append({
                "increment": inc,
                "advantage": summarize(advantages),
            })

        # Overall: average noun accuracy vs verb accuracy across all increments
        all_noun_means = [np.mean(learning_curves["NOUN"][inc])
                          for inc in cfg.training_increments]
        all_verb_means = [np.mean(learning_curves["VERB"][inc])
                          for inc in cfg.training_increments]
        overall_advantages = [n - v for n, v in zip(all_noun_means, all_verb_means)]
        h1_overall = summarize(overall_advantages)

        metrics["h1_noun_before_verb"] = {
            "per_increment": h1_noun_advantage,
            "overall_advantage": h1_overall,
            "noun_leads_at_all_increments": all(a >= 0 for a in overall_advantages),
        }
        self.log(f"  Noun advantage (mean across increments): "
                 f"{h1_overall['mean']:.3f} +/- {h1_overall['sem']:.3f}")
        self.log(f"  Noun leads at all increments: "
                 f"{metrics['h1_noun_before_verb']['noun_leads_at_all_increments']}")

        # ==============================================================
        # H2: Content words acquired before function words
        # ==============================================================
        self.log("\nH2: Content words before function words")

        h2_advantages = []
        for inc in cfg.training_increments:
            content_vals = []
            function_vals = []
            for s_idx in range(cfg.n_seeds):
                content_acc = np.mean([
                    learning_curves[c][inc][s_idx]
                    for c in CONTENT_CATEGORIES
                    if c in learning_curves
                ])
                function_acc = np.mean([
                    learning_curves[c][inc][s_idx]
                    for c in FUNCTION_CATEGORIES
                    if c in learning_curves
                ])
                content_vals.append(content_acc)
                function_vals.append(function_acc)
            diff = [c - f for c, f in zip(content_vals, function_vals)]
            h2_advantages.append({
                "increment": inc,
                "content_mean": summarize(content_vals),
                "function_mean": summarize(function_vals),
                "advantage": summarize(diff),
            })

        # Overall content vs function advantage
        all_content_means = [entry["content_mean"]["mean"] for entry in h2_advantages]
        all_function_means = [entry["function_mean"]["mean"] for entry in h2_advantages]
        overall_cf_diff = [c - f for c, f in zip(all_content_means, all_function_means)]
        h2_overall = summarize(overall_cf_diff)

        metrics["h2_content_before_function"] = {
            "per_increment": h2_advantages,
            "overall_advantage": h2_overall,
            "content_leads_majority": sum(1 for d in overall_cf_diff if d > 0) > len(overall_cf_diff) / 2,
        }
        self.log(f"  Content-function advantage (mean): "
                 f"{h2_overall['mean']:.3f} +/- {h2_overall['sem']:.3f}")
        self.log(f"  Content leads at majority of increments: "
                 f"{metrics['h2_content_before_function']['content_leads_majority']}")

        # ==============================================================
        # H3: Vocabulary spurt at a predictable threshold
        # ==============================================================
        self.log("\nH3: Vocabulary spurt threshold")

        # Use overall accuracy (mean across categories and seeds)
        overall_by_increment = []
        for inc in cfg.training_increments:
            all_cats_vals = []
            for cat in ALL_TEST_CATEGORIES:
                all_cats_vals.extend(learning_curves[cat][inc])
            overall_by_increment.append(float(np.mean(all_cats_vals)))

        spurt = _find_spurt_threshold(
            list(cfg.training_increments), overall_by_increment)
        metrics["h3_vocabulary_spurt"] = spurt
        self.log(f"  Spurt at {spurt['threshold_sentences']} sentences "
                 f"(gain rate={spurt['max_gain_rate']:.4f})")

        # Per-category spurt thresholds
        per_cat_spurts = {}
        for cat in ALL_TEST_CATEGORIES:
            cat_means = [np.mean(learning_curves[cat][inc])
                         for inc in cfg.training_increments]
            cat_spurt = _find_spurt_threshold(
                list(cfg.training_increments), cat_means)
            per_cat_spurts[cat] = cat_spurt
            self.log(f"    {cat}: spurt at {cat_spurt['threshold_sentences']} sentences")
        metrics["h3_per_category_spurts"] = per_cat_spurts

        # ==============================================================
        # H4: Sigmoid fit to learning curves
        # ==============================================================
        self.log("\nH4: Sigmoid fit to learning curves")

        h4_fits = {}
        for cat in ALL_TEST_CATEGORIES:
            cat_means = [np.mean(learning_curves[cat][inc])
                         for inc in cfg.training_increments]
            x_data = [float(inc) for inc in cfg.training_increments]
            fit = _fit_sigmoid(x_data, cat_means)
            h4_fits[cat] = fit
            status = ("R^2={:.3f}".format(fit["r_squared"])
                      if fit["success"] else "FAILED")
            self.log(f"  {cat}: {status}")

        all_r2 = [f["r_squared"] for f in h4_fits.values() if f["success"]]
        mean_r2 = float(np.mean(all_r2)) if all_r2 else 0.0
        metrics["h4_sigmoid_fit"] = {
            "per_category": h4_fits,
            "mean_r_squared": mean_r2,
            "all_above_0_9": all(r > 0.9 for r in all_r2) if all_r2 else False,
            "n_successful_fits": len(all_r2),
        }
        self.log(f"  Mean R^2: {mean_r2:.3f}")
        self.log(f"  All > 0.9: {metrics['h4_sigmoid_fit']['all_above_0_9']}")

        # ==============================================================
        # Acquisition order vs AoA rank correlation
        # ==============================================================
        self.log("\nAcquisition order vs AoA (Spearman rank correlation)")

        # Rank categories by mean accuracy at the smallest increment
        first_inc = cfg.training_increments[0]
        acq_order = {}
        for cat in ALL_TEST_CATEGORIES:
            if cat in CATEGORY_MEAN_AOA:
                acq_order[cat] = np.mean(learning_curves[cat][first_inc])

        if len(acq_order) >= 3:
            cats_sorted = sorted(acq_order.keys())
            model_ranks = [acq_order[c] for c in cats_sorted]
            aoa_ranks = [CATEGORY_MEAN_AOA[c] for c in cats_sorted]
            # Higher accuracy = earlier acquisition; lower AoA = earlier acquisition
            # Expect negative correlation: high accuracy <-> low AoA
            rho, p_val = scipy_stats.spearmanr(model_ranks, aoa_ranks)
            metrics["acquisition_order_correlation"] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
                "expected_negative": rho < 0,
            }
            self.log(f"  Spearman rho={rho:.3f}, p={p_val:.3f}")
        else:
            metrics["acquisition_order_correlation"] = {
                "spearman_rho": 0.0, "p_value": 1.0,
                "significant": False, "expected_negative": False,
            }
            self.log("  Insufficient categories for correlation")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "n_seeds": cfg.n_seeds,
                "rounds": cfg.rounds, "p": cfg.p, "beta": cfg.beta,
                "training_increments": list(cfg.training_increments),
                "vocab_size": len(vocab),
            },
            metrics=metrics,
            raw_data={
                "learning_curves": {
                    cat: {str(inc): learning_curves[cat][inc]
                          for inc in cfg.training_increments}
                    for cat in ALL_TEST_CATEGORIES
                },
            },
            duration_seconds=duration,
        )


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Developmental Acquisition Curves Experiment")
    ap.add_argument("--quick", action="store_true",
                    help="Quick run with fewer seeds and smaller network")
    args = ap.parse_args()

    exp = DevelopmentalCurvesExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=2, n=5000, k=50,
                         training_increments=(1, 5, 20, 50))
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # -- Summary --
    print("\n" + "=" * 70)
    print("DEVELOPMENTAL ACQUISITION CURVES SUMMARY")
    print("=" * 70)

    m = result.metrics

    print("\nLearning curves (mean accuracy):")
    curves = m["learning_curves"]
    cats = sorted(curves.keys())
    increments = [entry["increment"] for entry in curves[cats[0]]]
    header = f"{'inc':>6s} " + " ".join(f"{c:>6s}" for c in cats)
    print(f"  {header}")
    for i, inc in enumerate(increments):
        vals = [curves[c][i]["mean"] for c in cats]
        row = f"{inc:6d} " + " ".join(f"{v:6.3f}" for v in vals)
        print(f"  {row}")

    h1 = m["h1_noun_before_verb"]
    print(f"\nH1 -- Noun > Verb advantage: "
          f"{h1['overall_advantage']['mean']:.3f} "
          f"(leads at all: {h1['noun_leads_at_all_increments']})")

    h2 = m["h2_content_before_function"]
    print(f"H2 -- Content > Function advantage: "
          f"{h2['overall_advantage']['mean']:.3f} "
          f"(majority: {h2['content_leads_majority']})")

    h3 = m["h3_vocabulary_spurt"]
    print(f"H3 -- Spurt at {h3['threshold_sentences']} sentences "
          f"(rate={h3['max_gain_rate']:.4f})")

    h4 = m["h4_sigmoid_fit"]
    print(f"H4 -- Sigmoid mean R^2: {h4['mean_r_squared']:.3f} "
          f"(all>0.9: {h4['all_above_0_9']})")

    if "acquisition_order_correlation" in m:
        corr = m["acquisition_order_correlation"]
        print(f"\nAoA correlation: rho={corr['spearman_rho']:.3f}, "
              f"p={corr['p_value']:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
