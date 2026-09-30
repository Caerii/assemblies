"""
Semantic Similarity: Lexical-Semantic N400 Effects

Tests whether N400 is modulated by semantic relatedness, controlling for
prediction frequency. Creates overlapping assemblies for semantically
related words via co-projection, then measures N400 at object position.

Design:
  Semantic clusters: ANIMALS = [dog, cat, bird], HUMANS = [boy, girl]
  Co-project within clusters to create shared assembly overlap.

  Train SVO sentences with EQUAL frequency for all verb-noun pairs
  (controls for prediction). Then measure N400:
    - Related:   "dog chases CAT"   (dog-cat share overlap)
    - Unrelated: "dog chases GIRL"  (no cluster overlap)

  Also measure assembly overlap directly to confirm semantic structure.

Hypotheses:
  H1: N400 related < N400 unrelated (semantic priming, frequency controlled)
  H2: Assembly overlap related > overlap unrelated
  H3: Correlation between assembly overlap and N400 is significant

Usage:
    uv run python research/experiments/primitives/test_semantic_similarity.py
    uv run python research/experiments/primitives/test_semantic_similarity.py --quick
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
    measure_overlap,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    build_semantic_structure,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400


# Semantic clusters within DEFAULT_VOCAB nouns
ANIMAL_CLUSTER = ["dog", "cat", "bird"]
HUMAN_CLUSTER = ["boy", "girl"]


@dataclass
class SemanticConfig:
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
    # Semantic structure
    semantic_rounds: int = 15  # co-projection rounds per pair
    # Training
    train_reps_per_pair: int = 5  # equal frequency for all verb-object combos


def _all_pairs(cluster: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique pairs from a cluster."""
    pairs = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            pairs.append((cluster[i], cluster[j]))
    return pairs


def run_trial(
    cfg: SemanticConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one semantic similarity trial."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Build semantic structure via co-projection
    clusters = []
    for a, b in _all_pairs(ANIMAL_CLUSTER):
        clusters.append((a, b, cfg.semantic_rounds))
    for a, b in _all_pairs(HUMAN_CLUSTER):
        clusters.append((a, b, cfg.semantic_rounds))
    build_semantic_structure(brain, clusters, "NOUN_CORE")

    # Train SVO with equal frequency for ALL verb-object pairs
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    sentences = []

    for verb in verbs:
        for obj in nouns:
            for _ in range(cfg.train_reps_per_pair):
                agents = [n for n in nouns if n != obj]
                agent = rng.choice(agents)
                sentences.append({
                    "words": [agent, verb, obj],
                    "roles": ["AGENT", "VERB", "PATIENT"],
                    "categories": ["NOUN", "VERB", "NOUN"],
                    "has_pp": False,
                })

    rng.shuffle(sentences)
    for sent in sentences:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    # Measure assembly overlap in NOUN_CORE (where co-projection happened)
    related_overlaps = []
    unrelated_overlaps = []

    def get_noun_assembly(word):
        activate_word(brain, word, "NOUN_CORE", 3)
        return np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)

    noun_assemblies = {n: get_noun_assembly(n) for n in nouns}

    for a, b in _all_pairs(ANIMAL_CLUSTER):
        related_overlaps.append(
            measure_overlap(noun_assemblies[a], noun_assemblies[b]))
    for a, b in _all_pairs(HUMAN_CLUSTER):
        related_overlaps.append(
            measure_overlap(noun_assemblies[a], noun_assemblies[b]))

    # Cross-cluster pairs (unrelated)
    for a in ANIMAL_CLUSTER:
        for b in HUMAN_CLUSTER:
            unrelated_overlaps.append(
                measure_overlap(noun_assemblies[a], noun_assemblies[b]))

    # Measure N400 at object position
    # Project from BOTH NOUN_CORE and VERB_CORE into PREDICTION
    # to capture agent's semantic influence on prediction
    n400_related = []
    n400_unrelated = []

    for verb in verbs[:3]:
        for agent_noun in ANIMAL_CLUSTER:
            # Related: another animal
            related_objs = [n for n in ANIMAL_CLUSTER if n != agent_noun]
            for obj in related_objs:
                activate_word(brain, agent_noun, "NOUN_CORE", 3)
                activate_word(brain, verb, "VERB_CORE", 3)
                brain.inhibit_areas(["PREDICTION"])
                brain.project({}, {"NOUN_CORE": ["PREDICTION"],
                                   "VERB_CORE": ["PREDICTION"]})
                predicted = np.array(
                    brain.areas["PREDICTION"].winners, dtype=np.uint32)
                n400_related.append(measure_n400(predicted, lexicon[obj]))

            # Unrelated: a human
            for obj in HUMAN_CLUSTER:
                activate_word(brain, agent_noun, "NOUN_CORE", 3)
                activate_word(brain, verb, "VERB_CORE", 3)
                brain.inhibit_areas(["PREDICTION"])
                brain.project({}, {"NOUN_CORE": ["PREDICTION"],
                                   "VERB_CORE": ["PREDICTION"]})
                predicted = np.array(
                    brain.areas["PREDICTION"].winners, dtype=np.uint32)
                n400_unrelated.append(measure_n400(predicted, lexicon[obj]))

    brain.disable_plasticity = False

    # Correlation between overlap and N400
    all_overlaps = related_overlaps + unrelated_overlaps
    # Match N400 means to overlap categories
    all_n400s = ([np.mean(n400_related)] * len(related_overlaps) +
                 [np.mean(n400_unrelated)] * len(unrelated_overlaps))

    if len(set(all_overlaps)) > 1 and len(set(all_n400s)) > 1:
        rho, rho_p = stats.spearmanr(all_overlaps, all_n400s)
    else:
        rho, rho_p = 0.0, 1.0

    return {
        "n400_related": float(np.mean(n400_related)),
        "n400_unrelated": float(np.mean(n400_unrelated)),
        "overlap_related": float(np.mean(related_overlaps)),
        "overlap_unrelated": float(np.mean(unrelated_overlaps)),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "n_train": len(sentences),
    }


class SemanticSimilarityExperiment(ExperimentBase):
    """Semantic similarity and N400 modulation."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="semantic_similarity",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[SemanticConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or SemanticConfig(
            **{k: v for k, v in kwargs.items()
               if k in SemanticConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Semantic Similarity: Lexical-Semantic N400 Effects")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  semantic_rounds={cfg.semantic_rounds}")
        self.log(f"  train_reps_per_pair={cfg.train_reps_per_pair}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = ["n400_related", "n400_unrelated",
                "overlap_related", "overlap_unrelated",
                "spearman_rho", "spearman_p"]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])
            if s == 0:
                self.log(f"    Training: {result['n_train']} sentences")

        # Report
        self.log(f"\n  === N400 by Semantic Relatedness ===")
        self.log(f"    Related:   {np.mean(vals['n400_related']):.3f}"
                 f" +/- {np.std(vals['n400_related']):.3f}")
        self.log(f"    Unrelated: {np.mean(vals['n400_unrelated']):.3f}"
                 f" +/- {np.std(vals['n400_unrelated']):.3f}")

        t_n400 = paired_ttest(vals["n400_unrelated"], vals["n400_related"])
        self.log(f"    Unrelated > Related: d={t_n400['d']:.2f}")

        self.log(f"\n  === Assembly Overlap ===")
        self.log(f"    Related:   {np.mean(vals['overlap_related']):.3f}"
                 f" +/- {np.std(vals['overlap_related']):.3f}")
        self.log(f"    Unrelated: {np.mean(vals['overlap_unrelated']):.3f}"
                 f" +/- {np.std(vals['overlap_unrelated']):.3f}")

        t_overlap = paired_ttest(vals["overlap_related"], vals["overlap_unrelated"])
        self.log(f"    Related > Unrelated: d={t_overlap['d']:.2f}")

        mean_rho = np.mean(vals["spearman_rho"])
        self.log(f"\n  === Correlation ===")
        self.log(f"    Spearman rho (overlap vs N400): {mean_rho:.3f}")

        h1 = np.mean(vals["n400_related"]) < np.mean(vals["n400_unrelated"])
        h2 = np.mean(vals["overlap_related"]) > np.mean(vals["overlap_unrelated"])
        h3 = mean_rho < -0.3  # negative: more overlap -> less N400

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (N400 related < unrelated): "
                 f"{'PASS' if h1 else 'FAIL'} (d={t_n400['d']:.2f})")
        self.log(f"    H2 (Overlap related > unrelated): "
                 f"{'PASS' if h2 else 'FAIL'} (d={t_overlap['d']:.2f})")
        self.log(f"    H3 (Significant correlation): "
                 f"{'PASS' if h3 else 'FAIL'} (rho={mean_rho:.3f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "n400": {
                "related": summarize(vals["n400_related"]),
                "unrelated": summarize(vals["n400_unrelated"]),
                "test": t_n400,
            },
            "overlap": {
                "related": summarize(vals["overlap_related"]),
                "unrelated": summarize(vals["overlap_unrelated"]),
                "test": t_overlap,
            },
            "correlation": {
                "spearman_rho": summarize(vals["spearman_rho"]),
                "spearman_p": summarize(vals["spearman_p"]),
            },
            "hypotheses": {
                "H1_n400_priming": h1,
                "H2_overlap_structure": h2,
                "H3_correlation": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "semantic_rounds": cfg.semantic_rounds,
                "train_reps_per_pair": cfg.train_reps_per_pair,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Similarity Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = SemanticSimilarityExperiment(verbose=True)

    if args.quick:
        cfg = SemanticConfig(
            n=5000, k=50,
            semantic_rounds=8, train_reps_per_pair=3)
        n_seeds = args.seeds or 5
    else:
        cfg = SemanticConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY SUMMARY")
    print("=" * 70)

    print("\nN400 by relatedness:")
    print(f"  Related:   {m['n400']['related']['mean']:.3f}"
          f" +/- {m['n400']['related']['sem']:.3f}")
    print(f"  Unrelated: {m['n400']['unrelated']['mean']:.3f}"
          f" +/- {m['n400']['unrelated']['sem']:.3f}")
    print(f"  d={m['n400']['test']['d']:.2f}")

    print("\nAssembly overlap:")
    print(f"  Related:   {m['overlap']['related']['mean']:.3f}")
    print(f"  Unrelated: {m['overlap']['unrelated']['mean']:.3f}")
    print(f"  d={m['overlap']['test']['d']:.2f}")

    print(f"\nSpearman rho: {m['correlation']['spearman_rho']['mean']:.3f}")

    h = m["hypotheses"]
    print(f"\nH1 N400 priming:     {'PASS' if h['H1_n400_priming'] else 'FAIL'}")
    print(f"H2 Overlap structure: {'PASS' if h['H2_overlap_structure'] else 'FAIL'}")
    print(f"H3 Correlation:       {'PASS' if h['H3_correlation'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
