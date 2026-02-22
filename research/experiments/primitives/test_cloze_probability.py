"""
Cloze Probability: Graded Prediction Strength and N400

Tests whether N400 scales continuously with training frequency,
analogous to how human N400 scales with cloze probability.

Design:
  Control verb-object pair frequency during training:
    HIGH:   pair appears 20x (e.g., "chases" -> "cat")
    MEDIUM: pair appears 5x  (e.g., "chases" -> "bird")
    LOW:    pair appears 1x  (e.g., "chases" -> "girl")
    ZERO:   pair never appears (e.g., "chases" -> "boy")

  After training, measure N400 at object position for each frequency level.
  The prediction is: N400_high < N400_med < N400_low < N400_zero.

  This establishes the quantitative relationship between training exposure
  and prediction error — essential for fitting human N400 data where cloze
  probability is the primary predictor of N400 amplitude.

Architecture: DEFAULT_VOCAB (7 areas). No CFG — direct sentence construction
  with controlled frequencies to isolate the frequency-N400 relationship.

Hypotheses:
  H1: Monotonic N400 gradient (high < med < low < zero)
  H2: Strong negative correlation between log(freq) and N400 (rho < -0.8)
  H3: Zero vs high N400 difference has d > 1.0

Usage:
    uv run python research/experiments/primitives/test_cloze_probability.py
    uv run python research/experiments/primitives/test_cloze_probability.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400


@dataclass
class ClozeConfig:
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
    # Frequency levels
    high_freq: int = 20
    med_freq: int = 5
    low_freq: int = 1


def build_frequency_corpus(
    cfg: ClozeConfig,
    vocab,
    rng: np.random.Generator,
) -> Tuple[List[Dict], Dict]:
    """Build training corpus with controlled verb-object pair frequencies.

    Assigns each verb a set of target nouns at different frequency levels.
    Returns sentence dicts and the frequency map for reference.
    """
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")

    # Assign frequency levels for the first 3 verbs
    # Each verb gets: high=nouns[0], med=nouns[1], low=nouns[2], zero=nouns[3]
    freq_map = {}  # (verb, noun) -> count
    sentences = []

    for vi, verb in enumerate(verbs[:3]):
        # Rotate noun assignment per verb to avoid confounds
        n_rotated = nouns[vi:] + nouns[:vi]
        assignments = [
            (n_rotated[0], cfg.high_freq),
            (n_rotated[1], cfg.med_freq),
            (n_rotated[2], cfg.low_freq),
            # n_rotated[3] is zero (not trained with this verb)
        ]

        for target_noun, count in assignments:
            freq_map[(verb, target_noun)] = count
            for _ in range(count):
                # Pick a random agent different from target
                agents = [n for n in nouns if n != target_noun]
                agent = rng.choice(agents)
                sentences.append({
                    "words": [agent, verb, target_noun],
                    "roles": ["AGENT", "VERB", "PATIENT"],
                    "categories": ["NOUN", "VERB", "NOUN"],
                    "has_pp": False,
                })

    # Shuffle training order
    rng.shuffle(sentences)
    return sentences, freq_map


def run_trial(
    cfg: ClozeConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one cloze probability trial."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Build controlled corpus
    sentences, freq_map = build_frequency_corpus(cfg, vocab, rng)

    # Train
    for sent in sentences:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")

    # Measure N400 at each frequency level
    n400_high, n400_med, n400_low, n400_zero = [], [], [], []

    for vi, verb in enumerate(verbs[:3]):
        n_rotated = nouns[vi:] + nouns[:vi]
        targets = {
            "high": n_rotated[0],
            "med": n_rotated[1],
            "low": n_rotated[2],
            "zero": n_rotated[3] if len(n_rotated) > 3 else nouns[-1],
        }

        # Pick a random agent for context
        agent = rng.choice([n for n in nouns
                            if n not in targets.values()])

        # Activate context
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        n400_high.append(measure_n400(predicted, lexicon[targets["high"]]))
        n400_med.append(measure_n400(predicted, lexicon[targets["med"]]))
        n400_low.append(measure_n400(predicted, lexicon[targets["low"]]))
        n400_zero.append(measure_n400(predicted, lexicon[targets["zero"]]))

    # Compute correlation
    freqs = ([cfg.high_freq] * len(n400_high)
             + [cfg.med_freq] * len(n400_med)
             + [cfg.low_freq] * len(n400_low)
             + [0] * len(n400_zero))
    n400s = n400_high + n400_med + n400_low + n400_zero
    log_freqs = [np.log(f + 1) for f in freqs]

    if len(set(n400s)) > 1 and len(set(log_freqs)) > 1:
        rho, rho_p = stats.spearmanr(log_freqs, n400s)
    else:
        rho, rho_p = 0.0, 1.0

    brain.disable_plasticity = False

    return {
        "n400_high": float(np.mean(n400_high)),
        "n400_med": float(np.mean(n400_med)),
        "n400_low": float(np.mean(n400_low)),
        "n400_zero": float(np.mean(n400_zero)),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "n_train_sentences": len(sentences),
    }


class ClozeProbabilityExperiment(ExperimentBase):
    """Cloze probability / graded prediction experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="cloze_probability",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[ClozeConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or ClozeConfig(
            **{k: v for k, v in kwargs.items()
               if k in ClozeConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Cloze Probability: Graded Prediction and N400")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  Frequencies: high={cfg.high_freq},"
                 f" med={cfg.med_freq}, low={cfg.low_freq}, zero=0")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = ["n400_high", "n400_med", "n400_low", "n400_zero",
                "spearman_rho", "spearman_p"]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])
            if s == 0:
                self.log(f"    Training corpus: {result['n_train_sentences']} sentences")

        # Report
        self.log(f"\n  === N400 by Frequency Level ===")
        for level in ["high", "med", "low", "zero"]:
            freq = {"high": cfg.high_freq, "med": cfg.med_freq,
                    "low": cfg.low_freq, "zero": 0}[level]
            self.log(f"    {level:5s} (freq={freq:2d}): "
                     f"{np.mean(vals[f'n400_{level}']):.3f}"
                     f" +/- {np.std(vals[f'n400_{level}']):.3f}")

        t_zero_high = paired_ttest(vals["n400_zero"], vals["n400_high"])
        t_low_high = paired_ttest(vals["n400_low"], vals["n400_high"])
        t_med_high = paired_ttest(vals["n400_med"], vals["n400_high"])

        self.log(f"\n  === Pairwise Comparisons (vs high) ===")
        self.log(f"    med  vs high: d={t_med_high['d']:.2f}")
        self.log(f"    low  vs high: d={t_low_high['d']:.2f}")
        self.log(f"    zero vs high: d={t_zero_high['d']:.2f}")

        mean_rho = np.mean(vals["spearman_rho"])
        self.log(f"\n  === Correlation ===")
        self.log(f"    Spearman rho (log freq vs N400): {mean_rho:.3f}")

        # Check monotonicity
        means = [np.mean(vals[f"n400_{l}"]) for l in ["high", "med", "low", "zero"]]
        is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Monotonic gradient): "
                 f"{'PASS' if is_monotonic else 'FAIL'}"
                 f" ({' < '.join(f'{m:.3f}' for m in means)})")
        self.log(f"    H2 (Spearman rho < -0.8): "
                 f"{'PASS' if mean_rho < -0.8 else 'FAIL'}"
                 f" (rho={mean_rho:.3f})")
        self.log(f"    H3 (Zero vs high d > 1.0): "
                 f"{'PASS' if t_zero_high['d'] > 1.0 else 'FAIL'}"
                 f" (d={t_zero_high['d']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "n400_by_frequency": {
                level: summarize(vals[f"n400_{level}"])
                for level in ["high", "med", "low", "zero"]
            },
            "pairwise_tests": {
                "med_vs_high": t_med_high,
                "low_vs_high": t_low_high,
                "zero_vs_high": t_zero_high,
            },
            "correlation": {
                "spearman_rho": summarize(vals["spearman_rho"]),
                "spearman_p": summarize(vals["spearman_p"]),
            },
            "hypotheses": {
                "H1_monotonic": is_monotonic,
                "H2_spearman_strong": mean_rho < -0.8,
                "H3_zero_high_large": t_zero_high["d"] > 1.0,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "high_freq": cfg.high_freq,
                "med_freq": cfg.med_freq,
                "low_freq": cfg.low_freq,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cloze Probability Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = ClozeProbabilityExperiment(verbose=True)

    if args.quick:
        cfg = ClozeConfig(
            n=5000, k=50,
            high_freq=10, med_freq=3, low_freq=1)
        n_seeds = args.seeds or 5
    else:
        cfg = ClozeConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("CLOZE PROBABILITY SUMMARY")
    print("=" * 70)

    print("\nN400 by frequency level:")
    for level in ["high", "med", "low", "zero"]:
        s = m["n400_by_frequency"][level]
        print(f"  {level:5s}: {s['mean']:.3f} +/- {s['sem']:.3f}")

    print("\nPairwise (vs high):")
    for label, key in [("med", "med_vs_high"), ("low", "low_vs_high"),
                        ("zero", "zero_vs_high")]:
        t = m["pairwise_tests"][key]
        print(f"  {label:5s}: d={t['d']:.2f}")

    print(f"\nSpearman rho: {m['correlation']['spearman_rho']['mean']:.3f}")

    h = m["hypotheses"]
    print(f"\nH1 Monotonic:     {'PASS' if h['H1_monotonic'] else 'FAIL'}")
    print(f"H2 Rho < -0.8:   {'PASS' if h['H2_spearman_strong'] else 'FAIL'}")
    print(f"H3 Zero-high d>1: {'PASS' if h['H3_zero_high_large'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
