"""
Vocabulary and Structure Scaling

Tests how the system's performance scales with vocabulary size.
Creates multiple vocabulary sizes (5, 10, 20, 40 nouns with proportional
verbs) and measures assembly discriminability, prediction accuracy (N400),
and binding accuracy (P600) at each scale.

Hypotheses:
  H1: Assembly pairwise overlap stays < 0.5 up to 20 nouns (n=10000, k=100)
  H2: Prediction (N400 d) is robust across scales (d > 2.0)
  H3: There's a critical scale where binding (P600 d) breaks down
  H4: Training efficiency scales sublinearly (generalization from categories)

Usage:
    uv run python research/experiments/primitives/test_vocabulary_scaling.py
    uv run python research/experiments/primitives/test_vocabulary_scaling.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from itertools import combinations

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
    measure_overlap,
)
from research.experiments.lib.vocabulary import Vocabulary, CategoryDef
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import (
    measure_n400,
    measure_p600,
    forward_predict_from_context,
)


# Word pools for generating scaled vocabularies
NOUN_POOL = [
    "dog", "cat", "bird", "boy", "girl",
    "fish", "bear", "deer", "fox", "cow",
    "ant", "bee", "rat", "hen", "pig",
    "bat", "elk", "yak", "ram", "owl",
    "ape", "jay", "eel", "cod", "gnu",
    "fly", "bug", "pup", "cub", "kit",
    "doe", "ewe", "dam", "cob", "pen",
    "tom", "vix", "hob", "nit", "dab",
]

VERB_POOL = [
    "chases", "sees", "eats", "finds", "hits",
    "kicks", "bites", "pokes", "grabs", "pulls",
    "lifts", "drops", "holds", "throws", "picks",
    "bumps", "taps", "rubs", "hugs", "pats",
]


@dataclass
class VocabularyScalingConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    # Training
    sents_per_noun: int = 20  # sentences per noun in vocabulary
    prediction_rounds: int = 5
    binding_rounds: int = 10
    # Test
    n_test_items: int = 5
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    # Scales
    noun_counts: List[int] = None

    def __post_init__(self):
        if self.noun_counts is None:
            self.noun_counts = [5, 10, 20, 40]


def make_vocabulary(n_nouns: int, n_verbs: int) -> Vocabulary:
    """Create a vocabulary with specified noun and verb counts."""
    nouns = NOUN_POOL[:n_nouns]
    verbs = VERB_POOL[:n_verbs]

    return Vocabulary(
        categories={
            "NOUN": CategoryDef(
                words=nouns,
                core_area="NOUN_CORE",
                role_areas={
                    "AGENT": "ROLE_AGENT",
                    "PATIENT": "ROLE_PATIENT",
                },
            ),
            "VERB": CategoryDef(
                words=verbs,
                core_area="VERB_CORE",
                role_areas={},
            ),
        },
    )


def measure_pairwise_overlap(
    brain, words: List[str], area: str, activate_rounds: int = 3,
) -> float:
    """Measure mean pairwise assembly overlap between words in an area."""
    assemblies = {}
    brain.disable_plasticity = True
    for word in words:
        activate_word(brain, word, area, activate_rounds)
        assemblies[word] = np.array(brain.areas[area].winners, dtype=np.uint32)
    brain.disable_plasticity = False

    overlaps = []
    for w1, w2 in combinations(words, 2):
        overlaps.append(measure_overlap(assemblies[w1], assemblies[w2]))

    return float(np.mean(overlaps)) if overlaps else 0.0


def run_scale_trial(
    cfg: VocabularyScalingConfig,
    n_nouns: int,
    seed: int,
) -> Dict[str, Any]:
    """Run one trial at a given vocabulary scale."""
    n_verbs = max(3, n_nouns // 2)
    vocab = make_vocabulary(n_nouns, n_verbs)

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Train with proportional sentence count
    rng = np.random.default_rng(seed)
    n_train = cfg.sents_per_noun * n_nouns
    grammar = SimpleCFG(pp_prob=0.0, vocab=vocab, rng=rng)
    train_sents = grammar.generate_batch(n_train)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.prediction_rounds, cfg.binding_rounds)

    # Measure assembly discriminability
    nouns = vocab.words_for_category("NOUN")
    pairwise_overlap = measure_pairwise_overlap(brain, nouns, "NOUN_CORE")

    # Measure prediction accuracy (N400)
    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    verbs = vocab.words_for_category("VERB")
    ni = min(cfg.n_test_items, n_nouns - 1)

    n400_gram = []
    n400_cv = []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        predicted = forward_predict_from_context(
            brain, [agent, verb], vocab)
        n400_gram.append(measure_n400(predicted, lexicon[gram_obj]))
        n400_cv.append(measure_n400(predicted, lexicon[cv_obj]))

    # Measure binding accuracy (P600)
    p600_gram = []
    p600_cv = []
    for i in range(ni):
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]
        p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

    brain.disable_plasticity = False

    # Effect sizes
    n400_d = paired_ttest(n400_cv, n400_gram)["d"] if len(n400_gram) > 1 else 0.0
    p600_d = paired_ttest(p600_cv, p600_gram)["d"] if len(p600_gram) > 1 else 0.0

    return {
        "n_nouns": n_nouns,
        "n_verbs": n_verbs,
        "n_train": n_train,
        "pairwise_overlap": pairwise_overlap,
        "n400_d": n400_d,
        "p600_d": p600_d,
        "n400_gram_mean": float(np.mean(n400_gram)) if n400_gram else 0.0,
        "n400_cv_mean": float(np.mean(n400_cv)) if n400_cv else 0.0,
        "p600_gram_mean": float(np.mean(p600_gram)) if p600_gram else 0.0,
        "p600_cv_mean": float(np.mean(p600_cv)) if p600_cv else 0.0,
    }


class VocabularyScalingExperiment(ExperimentBase):
    """Vocabulary scaling experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="vocabulary_scaling",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 5,
        config: Optional[VocabularyScalingConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or VocabularyScalingConfig(
            **{k: v for k, v in kwargs.items()
               if k in VocabularyScalingConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Vocabulary and Structure Scaling")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  noun_counts: {cfg.noun_counts}")
        self.log(f"  sents_per_noun: {cfg.sents_per_noun}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Accumulate per scale
        scale_results = {nc: [] for nc in cfg.noun_counts}

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            for nc in cfg.noun_counts:
                self.log(f"    Scale: {nc} nouns ...")
                trial = run_scale_trial(cfg, nc, self.seed + s)
                scale_results[nc].append(trial)

        # Aggregate per scale
        aggregated = {}
        for nc in cfg.noun_counts:
            trials = scale_results[nc]
            agg = {
                "n_nouns": nc,
                "pairwise_overlap": summarize(
                    [t["pairwise_overlap"] for t in trials]),
                "n400_d": summarize([t["n400_d"] for t in trials]),
                "p600_d": summarize([t["p600_d"] for t in trials]),
                "n_train": trials[0]["n_train"],
            }
            aggregated[nc] = agg

        # Report
        self.log(f"\n  {'Nouns':>6} | {'Overlap':>8} | {'N400 d':>8} | "
                 f"{'P600 d':>8} | {'n_train':>7}")
        self.log(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
        for nc in cfg.noun_counts:
            a = aggregated[nc]
            self.log(
                f"  {nc:>6} | "
                f"{a['pairwise_overlap']['mean']:>8.3f} | "
                f"{a['n400_d']['mean']:>8.2f} | "
                f"{a['p600_d']['mean']:>8.2f} | "
                f"{a['n_train']:>7}")

        # Hypotheses
        # H1: overlap < 0.5 at 20 nouns
        h1 = False
        if 20 in aggregated:
            h1 = aggregated[20]["pairwise_overlap"]["mean"] < 0.5

        # H2: N400 d > 2.0 at all scales
        h2 = all(aggregated[nc]["n400_d"]["mean"] > 2.0
                  for nc in cfg.noun_counts)

        # H3: Find critical scale where P600 d drops below 1.0
        critical_scale = None
        for nc in cfg.noun_counts:
            if aggregated[nc]["p600_d"]["mean"] < 1.0:
                critical_scale = nc
                break
        h3 = critical_scale is not None

        # H4: Sublinear training (N400 d doesn't halve when vocab doubles)
        h4 = True
        for i in range(len(cfg.noun_counts) - 1):
            nc1 = cfg.noun_counts[i]
            nc2 = cfg.noun_counts[i + 1]
            if nc2 >= nc1 * 2:
                d1 = aggregated[nc1]["n400_d"]["mean"]
                d2 = aggregated[nc2]["n400_d"]["mean"]
                if d1 > 0 and d2 < d1 * 0.25:
                    h4 = False

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Overlap < 0.5 at 20 nouns):    "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" ({aggregated.get(20, {}).get('pairwise_overlap', {}).get('mean', 'N/A')})")
        self.log(f"    H2 (N400 d > 2.0 at all scales):   "
                 f"{'PASS' if h2 else 'FAIL'}")
        self.log(f"    H3 (Critical P600 breakdown):       "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (scale={critical_scale})")
        self.log(f"    H4 (Sublinear training):            "
                 f"{'PASS' if h4 else 'FAIL'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "scales": {
                str(nc): {
                    "pairwise_overlap": aggregated[nc]["pairwise_overlap"],
                    "n400_d": aggregated[nc]["n400_d"],
                    "p600_d": aggregated[nc]["p600_d"],
                    "n_train": aggregated[nc]["n_train"],
                }
                for nc in cfg.noun_counts
            },
            "critical_p600_scale": critical_scale,
            "hypotheses": {
                "H1_overlap_below_05": h1,
                "H2_prediction_robust": h2,
                "H3_binding_breakdown": h3,
                "H4_sublinear_training": h4,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "noun_counts": cfg.noun_counts,
                "sents_per_noun": cfg.sents_per_noun,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Vocabulary Scaling Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = VocabularyScalingExperiment(verbose=True)

    if args.quick:
        cfg = VocabularyScalingConfig(
            n=5000, k=50,
            sents_per_noun=10,
            n_test_items=4,
            noun_counts=[5, 10, 20])
        n_seeds = args.seeds or 3
    else:
        cfg = VocabularyScalingConfig()
        n_seeds = args.seeds or 5

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("VOCABULARY SCALING SUMMARY")
    print("=" * 70)
    print(f"\nH1 Overlap < 0.5 at 20:    {'PASS' if h['H1_overlap_below_05'] else 'FAIL'}")
    print(f"H2 Prediction robust:       {'PASS' if h['H2_prediction_robust'] else 'FAIL'}")
    print(f"H3 Binding breakdown found: {'PASS' if h['H3_binding_breakdown'] else 'FAIL'}")
    print(f"H4 Sublinear training:      {'PASS' if h['H4_sublinear_training'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
