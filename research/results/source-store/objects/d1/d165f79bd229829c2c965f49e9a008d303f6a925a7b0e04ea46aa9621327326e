"""
P600 Signal Informativeness

Tests whether the P600 signal is informative about what needs learning.
If P600 measures structural integration difficulty, then items with high P600
after partial training should improve more from additional training.

Setup:
  - Train brain on first half of training data
  - Measure P600 for all test items, split into high-P600 and low-P600 groups
  - Train on second half of data
  - Re-measure P600 for both groups
  - Verify: high-P600 items show larger decrease

Hypotheses:
  H1: High-P600 items show larger decrease after training (delta > low delta)
  H2: Initial P600 correlates with improvement (r > 0.3)
  H3: After full training, P600 variance narrows

Usage:
    uv run python research/experiments/primitives/test_p600_informativeness.py
    uv run python research/experiments/primitives/test_p600_informativeness.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_p600


@dataclass
class P600InformativenessConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    # Training
    n_train_first_half: int = 100
    n_train_second_half: int = 100
    prediction_rounds: int = 5
    binding_rounds: int = 10
    # Test
    n_settling_rounds: int = 10


def generate_test_items(vocab, rng) -> List[Tuple[str, str, str]]:
    """Generate (word, core_area, role_area) triples for P600 measurement.

    Tests all nouns at patient and agent positions.
    """
    nouns = vocab.words_for_category("NOUN")
    items = []
    for noun in nouns:
        items.append((noun, "NOUN_CORE", "ROLE_PATIENT"))
        items.append((noun, "NOUN_CORE", "ROLE_AGENT"))
    return items


def run_trial(cfg: P600InformativenessConfig, seed: int) -> Dict[str, Any]:
    """Run one trial of P600 informativeness."""
    vocab = DEFAULT_VOCAB
    rng = np.random.default_rng(seed)

    # Create brain
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Generate all training data
    grammar = SimpleCFG(pp_prob=0.4, vocab=vocab, rng=rng)
    all_sents = grammar.generate_batch(
        cfg.n_train_first_half + cfg.n_train_second_half)
    first_half = all_sents[:cfg.n_train_first_half]
    second_half = all_sents[cfg.n_train_first_half:]

    # Phase 1: Train on first half
    for sent in first_half:
        train_sentence(brain, sent, vocab,
                       cfg.prediction_rounds, cfg.binding_rounds)

    # Measure P600 for all test items after partial training
    test_items = generate_test_items(vocab, rng)

    brain.disable_plasticity = True
    p600_partial = []
    for word, core, role in test_items:
        p = measure_p600(brain, word, core, role, cfg.n_settling_rounds)
        p600_partial.append(p)
    brain.disable_plasticity = False

    # Phase 2: Train on second half
    for sent in second_half:
        train_sentence(brain, sent, vocab,
                       cfg.prediction_rounds, cfg.binding_rounds)

    # Measure P600 for all test items after full training
    brain.disable_plasticity = True
    p600_full = []
    for word, core, role in test_items:
        p = measure_p600(brain, word, core, role, cfg.n_settling_rounds)
        p600_full.append(p)
    brain.disable_plasticity = False

    return {
        "p600_partial": p600_partial,
        "p600_full": p600_full,
        "test_items": [(w, c, r) for w, c, r in test_items],
    }


class P600InformativenessExperiment(ExperimentBase):
    """P600 signal informativeness experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="p600_informativeness",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[P600InformativenessConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or P600InformativenessConfig(
            **{k: v for k, v in kwargs.items()
               if k in P600InformativenessConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("P600 Signal Informativeness")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train: {cfg.n_train_first_half} + "
                 f"{cfg.n_train_second_half}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Accumulators across seeds
        high_deltas = []
        low_deltas = []
        correlations = []
        partial_vars = []
        full_vars = []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            partial = np.array(trial["p600_partial"])
            full = np.array(trial["p600_full"])
            delta = partial - full  # positive = improvement

            # Split into high/low by median of partial P600
            median_p600 = np.median(partial)
            high_mask = partial >= median_p600
            low_mask = partial < median_p600

            if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                high_deltas.append(float(np.mean(delta[high_mask])))
                low_deltas.append(float(np.mean(delta[low_mask])))

            # Correlation between initial P600 and improvement
            if len(partial) > 2 and np.std(partial) > 0 and np.std(delta) > 0:
                r = float(np.corrcoef(partial, delta)[0, 1])
                correlations.append(r)

            partial_vars.append(float(np.var(partial)))
            full_vars.append(float(np.var(full)))

        # Aggregate
        high_delta_summary = summarize(high_deltas) if high_deltas else {"mean": 0.0}
        low_delta_summary = summarize(low_deltas) if low_deltas else {"mean": 0.0}
        corr_summary = summarize(correlations) if correlations else {"mean": 0.0}

        # Hypotheses
        h1 = (high_delta_summary["mean"] > low_delta_summary["mean"]
               if high_deltas and low_deltas else False)
        h2 = corr_summary["mean"] > 0.3 if correlations else False
        h3 = (float(np.mean(full_vars)) < float(np.mean(partial_vars))
               if partial_vars and full_vars else False)

        self.log(f"\n  === Results ===")
        self.log(f"    High-P600 group delta: {high_delta_summary['mean']:.3f}")
        self.log(f"    Low-P600 group delta:  {low_delta_summary['mean']:.3f}")
        self.log(f"    Mean correlation:       {corr_summary['mean']:.3f}")
        self.log(f"    Partial P600 variance:  {float(np.mean(partial_vars)):.4f}")
        self.log(f"    Full P600 variance:     {float(np.mean(full_vars)):.4f}")

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (High delta > low delta):  "
                 f"{'PASS' if h1 else 'FAIL'} "
                 f"({high_delta_summary['mean']:.3f} vs "
                 f"{low_delta_summary['mean']:.3f})")
        self.log(f"    H2 (Correlation > 0.3):       "
                 f"{'PASS' if h2 else 'FAIL'} "
                 f"(r={corr_summary['mean']:.3f})")
        self.log(f"    H3 (Variance narrows):        "
                 f"{'PASS' if h3 else 'FAIL'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "high_p600_delta": high_delta_summary,
            "low_p600_delta": low_delta_summary,
            "correlation": corr_summary,
            "partial_variance_mean": float(np.mean(partial_vars)) if partial_vars else 0.0,
            "full_variance_mean": float(np.mean(full_vars)) if full_vars else 0.0,
            "hypotheses": {
                "H1_high_larger_decrease": h1,
                "H2_correlation_above_03": h2,
                "H3_variance_narrows": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_train_first_half": cfg.n_train_first_half,
                "n_train_second_half": cfg.n_train_second_half,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="P600 Informativeness Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = P600InformativenessExperiment(verbose=True)

    if args.quick:
        cfg = P600InformativenessConfig(
            n=5000, k=50,
            n_train_first_half=50,
            n_train_second_half=50)
        n_seeds = args.seeds or 3
    else:
        cfg = P600InformativenessConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("P600 INFORMATIVENESS SUMMARY")
    print("=" * 70)
    print(f"\nH1 High delta > low delta: {'PASS' if h['H1_high_larger_decrease'] else 'FAIL'}")
    print(f"H2 Correlation > 0.3:      {'PASS' if h['H2_correlation_above_03'] else 'FAIL'}")
    print(f"H3 Variance narrows:       {'PASS' if h['H3_variance_narrows'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
