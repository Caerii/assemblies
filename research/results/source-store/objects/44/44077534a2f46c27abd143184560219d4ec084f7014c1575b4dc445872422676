"""
Subgrammar Decomposition: Per-Production-Rule Learning Curves

Inspired by Schulz, Mitropolsky & Poggio (2025) "Unraveling Syntax: How
Language Models Learn Context-Free Grammars", which shows that transformer
loss decomposes recursively over CFG production rules and that all subgrammars
are learned in parallel.

This experiment tests whether Assembly Calculus shows the same parallel
learning or instead exhibits staged subgrammar acquisition (as predicted
by local Hebbian learning):

  1. Train one brain on full RecursiveCFG across multiple epochs
  2. At each epoch checkpoint, measure N400 separately for test sentences
     from each subgrammar (SVO, SVO+PP, SRC, ORC, etc.)
  3. Track per-subgrammar learning curves

Additionally measures assembly clustering (within-category vs between-category
overlap) as a representation probing check.

Hypotheses:
  H1: All subgrammars show N400 decrease across epochs (learning)
  H2: Simpler subgrammars (SVO) converge faster than complex (SRC/ORC)
  H3: Within-category assembly overlap > between-category (clustering)

Usage:
    uv run python research/experiments/primitives/test_subgrammar_decomposition.py
    uv run python research/experiments/primitives/test_subgrammar_decomposition.py --quick
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
    measure_overlap,
)
from research.experiments.lib.vocabulary import RECURSIVE_VOCAB
from research.experiments.lib.grammar import RecursiveCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400
from research.experiments.lib.subgrammar import (
    classify_sentence,
    partition_by_subgrammar,
    SubgrammarStats,
)


@dataclass
class SubgrammarDecompConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    lexicon_readout_rounds: int = 5
    # Grammar: full grammar with all features
    pp_prob: float = 0.4
    recursive_pp_prob: float = 0.4
    rel_prob: float = 0.3
    orc_prob: float = 0.3
    max_pp_depth: int = 2
    # Training
    n_train_sentences: int = 50
    n_epochs: int = 5
    # Test
    n_test_sentences: int = 50


def measure_per_subgrammar_n400(
    brain, test_sentences, vocab, lexicon,
) -> Dict[str, List[float]]:
    """Measure N400 at object position for each test sentence, grouped by subgrammar."""
    sg_n400s: Dict[str, List[float]] = {}

    for sent in test_sentences:
        words = sent["words"]
        if len(words) < 3:
            continue

        sg = classify_sentence(sent)

        # Forward predict from agent + verb -> object position
        core_0 = vocab.core_area_for(words[0])
        core_1 = vocab.core_area_for(words[1])
        activate_word(brain, words[0], core_0, 3)
        activate_word(brain, words[1], core_1, 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {core_1: ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # N400 for the actual object word
        target = words[2]
        n400 = measure_n400(predicted, lexicon[target])

        sg_n400s.setdefault(sg, []).append(n400)

    return sg_n400s


def measure_assembly_clustering(brain, vocab, lexicon):
    """Measure within-category vs between-category assembly overlap in PREDICTION."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    all_words = nouns + verbs

    within, between = [], []
    for i, w1 in enumerate(all_words):
        if w1 not in lexicon:
            continue
        cat1 = vocab.category_for_word(w1)
        for w2 in all_words[i + 1:]:
            if w2 not in lexicon:
                continue
            ov = measure_overlap(lexicon[w1], lexicon[w2])
            cat2 = vocab.category_for_word(w2)
            if cat1 == cat2:
                within.append(ov)
            else:
                between.append(ov)

    return {
        "within_category": float(np.mean(within)) if within else 0.0,
        "between_category": float(np.mean(between)) if between else 0.0,
    }


def run_trial(
    cfg: SubgrammarDecompConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one subgrammar decomposition trial."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    # Create brain
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Generate fixed test batch with full grammar
    test_grammar = RecursiveCFG(
        vocab=vocab, rng=np.random.default_rng(seed + 10000),
        pp_prob=cfg.pp_prob, recursive_pp_prob=cfg.recursive_pp_prob,
        rel_prob=cfg.rel_prob, orc_prob=cfg.orc_prob,
        max_pp_depth=cfg.max_pp_depth)
    test_sentences = test_grammar.generate_batch(cfg.n_test_sentences)

    # Partition test sentences by subgrammar
    test_partition = partition_by_subgrammar(test_sentences)

    epoch_results = []

    for epoch in range(cfg.n_epochs):
        # Generate and train one epoch of sentences
        train_grammar = RecursiveCFG(
            vocab=vocab, rng=rng,
            pp_prob=cfg.pp_prob, recursive_pp_prob=cfg.recursive_pp_prob,
            rel_prob=cfg.rel_prob, orc_prob=cfg.orc_prob,
            max_pp_depth=cfg.max_pp_depth)
        train_sents = train_grammar.generate_batch(cfg.n_train_sentences)

        for sent in train_sents:
            train_sentence(brain, sent, vocab,
                           cfg.train_rounds_per_pair, cfg.binding_rounds)

        # Measure at checkpoint
        brain.disable_plasticity = True
        lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

        sg_n400s = measure_per_subgrammar_n400(
            brain, test_sentences, vocab, lexicon)

        clustering = measure_assembly_clustering(brain, vocab, lexicon)

        # Production frequency of training batch
        prod_freq = RecursiveCFG.production_frequencies(train_sents)

        epoch_results.append({
            "epoch": epoch + 1,
            "sg_n400_means": {sg: float(np.mean(vals))
                              for sg, vals in sg_n400s.items()},
            "sg_counts": {sg: len(vals) for sg, vals in sg_n400s.items()},
            "clustering": clustering,
            "train_prod_freq": prod_freq,
        })

        brain.disable_plasticity = False

    return {
        "epoch_results": epoch_results,
        "test_partition_sizes": {sg: len(sents)
                                 for sg, sents in test_partition.items()},
    }


class SubgrammarDecompExperiment(ExperimentBase):
    """Per-subgrammar learning curve experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="subgrammar_decomposition",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[SubgrammarDecompConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or SubgrammarDecompConfig(
            **{k: v for k, v in kwargs.items()
               if k in SubgrammarDecompConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Subgrammar Decomposition: Per-Production-Rule Learning")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train={cfg.n_train_sentences}, n_epochs={cfg.n_epochs}")
        self.log(f"  n_test={cfg.n_test_sentences}, n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-epoch, per-subgrammar N400 across seeds
        all_subgrammars = set()
        # epoch_sg_n400s[epoch_idx][subgrammar] = [n400_mean_per_seed]
        epoch_sg_n400s: Dict[int, Dict[str, List[float]]] = {
            e: {} for e in range(cfg.n_epochs)}
        # clustering across seeds at final epoch
        final_within, final_between = [], []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            for er in trial["epoch_results"]:
                e = er["epoch"] - 1
                for sg, n400_mean in er["sg_n400_means"].items():
                    all_subgrammars.add(sg)
                    epoch_sg_n400s[e].setdefault(sg, []).append(n400_mean)

            # Final epoch clustering
            final_er = trial["epoch_results"][-1]
            final_within.append(final_er["clustering"]["within_category"])
            final_between.append(final_er["clustering"]["between_category"])

        # Report learning curves
        sorted_sgs = sorted(all_subgrammars)
        header = f"  {'Epoch':<6s}"
        for sg in sorted_sgs:
            header += f" | {sg:>12s}"
        self.log(f"\n  === Per-Subgrammar N400 Learning Curves ===")
        self.log(header)
        self.log("  " + "-" * len(header))

        for e in range(cfg.n_epochs):
            row = f"  {e + 1:<6d}"
            for sg in sorted_sgs:
                vals = epoch_sg_n400s[e].get(sg, [])
                mean = float(np.mean(vals)) if vals else float("nan")
                row += f" | {mean:12.4f}"
            self.log(row)

        # Check convergence ordering (H2)
        convergence_order = []
        for sg in sorted_sgs:
            first_epoch = epoch_sg_n400s[0].get(sg, [])
            last_epoch = epoch_sg_n400s[cfg.n_epochs - 1].get(sg, [])
            if first_epoch and last_epoch:
                decrease = float(np.mean(first_epoch)) - float(np.mean(last_epoch))
                convergence_order.append((sg, decrease))

        convergence_order.sort(key=lambda x: -x[1])  # most decrease first
        self.log(f"\n  Convergence ordering (most N400 decrease first):")
        for sg, dec in convergence_order:
            self.log(f"    {sg}: {dec:+.4f}")

        # Clustering
        self.log(f"\n  === Assembly Clustering (final epoch) ===")
        self.log(f"    Within-category:  {np.mean(final_within):.4f}")
        self.log(f"    Between-category: {np.mean(final_between):.4f}")

        # Hypotheses
        # H1: All subgrammars show decrease
        all_decrease = True
        for sg in sorted_sgs:
            first = epoch_sg_n400s[0].get(sg, [])
            last = epoch_sg_n400s[cfg.n_epochs - 1].get(sg, [])
            if first and last and np.mean(last) >= np.mean(first):
                all_decrease = False
                break
        h1 = all_decrease

        # H2: SVO converges faster (has larger decrease) than SRC or ORC
        svo_dec = dict(convergence_order).get("SVO", 0)
        complex_decs = [d for sg, d in convergence_order if sg in ("SRC", "ORC")]
        h2 = svo_dec > max(complex_decs) if complex_decs else False

        # H3: Within > between category overlap
        h3 = float(np.mean(final_within)) > float(np.mean(final_between))

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (All subgrammars learn):     "
                 f"{'PASS' if h1 else 'FAIL'}")
        self.log(f"    H2 (SVO converges fastest):     "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" (SVO={svo_dec:+.4f})")
        self.log(f"    H3 (Within > between cluster):  "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" ({np.mean(final_within):.4f} vs {np.mean(final_between):.4f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        # Build metrics
        learning_curves = {}
        for sg in sorted_sgs:
            learning_curves[sg] = {
                "epoch_means": [
                    float(np.mean(epoch_sg_n400s[e].get(sg, [0])))
                    for e in range(cfg.n_epochs)
                ],
            }

        metrics = {
            "learning_curves": learning_curves,
            "convergence_order": convergence_order,
            "clustering": {
                "within_category": summarize(final_within),
                "between_category": summarize(final_between),
            },
            "test_partition_sizes": {
                sg: len(epoch_sg_n400s[0].get(sg, []))
                for sg in sorted_sgs
            },
            "hypotheses": {
                "H1_all_learn": h1,
                "H2_svo_fastest": h2,
                "H3_clustering": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_train_sentences": cfg.n_train_sentences,
                "n_epochs": cfg.n_epochs,
                "n_test_sentences": cfg.n_test_sentences,
                "pp_prob": cfg.pp_prob, "rel_prob": cfg.rel_prob,
                "orc_prob": cfg.orc_prob,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Subgrammar Decomposition Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = SubgrammarDecompExperiment(verbose=True)

    if args.quick:
        cfg = SubgrammarDecompConfig(
            n=5000, k=50,
            n_train_sentences=30, n_epochs=3,
            n_test_sentences=40)
        n_seeds = args.seeds or 3
    else:
        cfg = SubgrammarDecompConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("SUBGRAMMAR DECOMPOSITION SUMMARY")
    print("=" * 70)
    print(f"\nH1 All subgrammars learn:  {'PASS' if h['H1_all_learn'] else 'FAIL'}")
    print(f"H2 SVO converges fastest:  {'PASS' if h['H2_svo_fastest'] else 'FAIL'}")
    print(f"H3 Assembly clustering:    {'PASS' if h['H3_clustering'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
