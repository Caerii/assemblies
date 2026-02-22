"""
Online Word Acquisition: Can new words integrate mid-training?

Tests whether the system can learn a truly novel word after initial training
by dynamically adding a new stimulus and forming an assembly for it.

Setup:
  - Train brain on standard SVO sentences (5 nouns)
  - Introduce novel word "lion" via brain.add_stimulus()
  - Form assembly for lion in NOUN_CORE
  - Train sentences containing "lion" as patient
  - Measure: assembly stability, prediction (N400), binding (P600), forgetting

Hypotheses:
  H1: Novel word forms stable assembly (self-overlap > 0.8)
  H2: Novel word integrates into prediction (N400 drops with training)
  H3: Novel word binds to roles (P600 comparable to trained words)
  H4: Existing assemblies preserved (overlap before/after > 0.8)

Usage:
    uv run python research/experiments/primitives/test_online_word_acquisition.py
    uv run python research/experiments/primitives/test_online_word_acquisition.py --quick
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
from research.experiments.lib.vocabulary import DEFAULT_VOCAB, Vocabulary, CategoryDef
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import (
    train_sentence,
    train_prediction_pair,
    train_binding,
)
from research.experiments.lib.measurement import (
    measure_n400,
    measure_p600,
    forward_predict_from_context,
)


NOVEL_WORD = "lion"


@dataclass
class OnlineAcquisitionConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    # Training
    n_train_sentences: int = 100
    prediction_rounds: int = 5
    binding_rounds: int = 10
    n_novel_train: int = 20
    # Test
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    n_test_items: int = 5


def snapshot_assemblies(
    brain, words: List[str], vocab: Vocabulary, activate_rounds: int = 3,
) -> Dict[str, np.ndarray]:
    """Record current assemblies for specified words in their core areas."""
    snapshots = {}
    for word in words:
        area = vocab.core_area_for(word)
        activate_word(brain, word, area, activate_rounds)
        snapshots[word] = np.array(brain.areas[area].winners, dtype=np.uint32)
    return snapshots


def run_trial(cfg: OnlineAcquisitionConfig, seed: int) -> Dict[str, Any]:
    """Run one trial of online word acquisition.

    Trains on standard vocabulary, then dynamically adds "lion" via
    brain.add_stimulus() and forms an assembly through projection.
    """
    vocab = DEFAULT_VOCAB
    rng = np.random.default_rng(seed)
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")

    # Phase 1: Create brain and train on standard sentences
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    grammar = SimpleCFG(pp_prob=0.4, vocab=vocab, rng=rng)
    train_sents = grammar.generate_batch(cfg.n_train_sentences)
    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.prediction_rounds, cfg.binding_rounds)

    # Snapshot existing assemblies before novel word
    brain.disable_plasticity = True
    pre_snapshots = snapshot_assemblies(brain, nouns, vocab)

    # Measure P600 for existing trained nouns (baseline)
    existing_p600 = []
    for noun in nouns:
        p = measure_p600(brain, noun, "NOUN_CORE", "ROLE_PATIENT",
                         cfg.n_settling_rounds)
        existing_p600.append(p)
    brain.disable_plasticity = False

    # Phase 2: Introduce novel word via add_stimulus
    brain.add_stimulus(f"PHON_{NOVEL_WORD}", cfg.k)

    # Form assembly in NOUN_CORE via repeated projection (plasticity ON)
    brain.inhibit_areas(["NOUN_CORE"])
    for _ in range(cfg.lexicon_rounds):
        brain.project(
            {f"PHON_{NOVEL_WORD}": ["NOUN_CORE"]},
            {"NOUN_CORE": ["NOUN_CORE"]},
        )

    # Measure assembly stability
    brain.disable_plasticity = True
    overlaps = []
    for _ in range(5):
        activate_word(brain, NOVEL_WORD, "NOUN_CORE", 5)
        snap1 = np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
        activate_word(brain, NOVEL_WORD, "NOUN_CORE", 5)
        snap2 = np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
        overlaps.append(measure_overlap(snap1, snap2))
    assembly_stability = float(np.mean(overlaps))

    # Measure N400 for novel word BEFORE training (should be high)
    lexicon_pre = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
    # Build lexicon entry for novel word manually
    brain.inhibit_areas(["PREDICTION"])
    for _ in range(cfg.lexicon_readout_rounds):
        brain.project({f"PHON_{NOVEL_WORD}": ["PREDICTION"]}, {})
    lexicon_pre[NOVEL_WORD] = np.array(
        brain.areas["PREDICTION"].winners, dtype=np.uint32)

    n400_novel_pre = []
    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        predicted = forward_predict_from_context(
            brain, [agent, verb], vocab)
        n400_novel_pre.append(measure_n400(predicted, lexicon_pre[NOVEL_WORD]))
    brain.disable_plasticity = False

    # Phase 3: Train sentences with novel word as patient
    # Train prediction and binding manually since lion isn't in the vocabulary.
    for i in range(cfg.n_novel_train):
        agent = nouns[rng.integers(len(nouns))]
        verb = verbs[rng.integers(len(verbs))]

        # Prediction: agent -> verb (standard)
        train_prediction_pair(brain, agent, "NOUN_CORE", verb,
                              cfg.prediction_rounds)
        # Prediction: verb -> lion
        train_prediction_pair(brain, verb, "VERB_CORE", NOVEL_WORD,
                              cfg.prediction_rounds)
        # Binding: agent to ROLE_AGENT
        train_binding(brain, agent, "NOUN_CORE", "ROLE_AGENT",
                      cfg.binding_rounds)
        # Binding: lion to ROLE_PATIENT
        train_binding(brain, NOVEL_WORD, "NOUN_CORE", "ROLE_PATIENT",
                      cfg.binding_rounds)

    # Phase 4: Post-training measurements
    brain.disable_plasticity = True

    # N400 for novel word AFTER training (should be lower)
    lexicon_post = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
    brain.inhibit_areas(["PREDICTION"])
    for _ in range(cfg.lexicon_readout_rounds):
        brain.project({f"PHON_{NOVEL_WORD}": ["PREDICTION"]}, {})
    lexicon_post[NOVEL_WORD] = np.array(
        brain.areas["PREDICTION"].winners, dtype=np.uint32)

    n400_novel_post = []
    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        predicted = forward_predict_from_context(
            brain, [agent, verb], vocab)
        n400_novel_post.append(measure_n400(predicted, lexicon_post[NOVEL_WORD]))

    # P600 for novel word (binding quality)
    novel_p600 = measure_p600(
        brain, NOVEL_WORD, "NOUN_CORE", "ROLE_PATIENT",
        cfg.n_settling_rounds)

    # Catastrophic forgetting check
    post_snapshots = snapshot_assemblies(brain, nouns, vocab)
    forgetting_overlaps = {}
    for word in nouns:
        if word in pre_snapshots and word in post_snapshots:
            forgetting_overlaps[word] = measure_overlap(
                pre_snapshots[word], post_snapshots[word])

    brain.disable_plasticity = False

    return {
        "assembly_stability": assembly_stability,
        "n400_novel_pre": n400_novel_pre,
        "n400_novel_post": n400_novel_post,
        "novel_p600": novel_p600,
        "existing_p600_mean": float(np.mean(existing_p600)),
        "forgetting_overlaps": forgetting_overlaps,
    }


class OnlineWordAcquisitionExperiment(ExperimentBase):
    """Online word acquisition experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="online_word_acquisition",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[OnlineAcquisitionConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or OnlineAcquisitionConfig(
            **{k: v for k, v in kwargs.items()
               if k in OnlineAcquisitionConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Online Word Acquisition")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train={cfg.n_train_sentences}, "
                 f"n_novel_train={cfg.n_novel_train}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Accumulators
        stabilities = []
        n400_pre_means = []
        n400_post_means = []
        novel_p600s = []
        existing_p600s = []
        min_forgetting = []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            stabilities.append(trial["assembly_stability"])
            n400_pre_means.append(float(np.mean(trial["n400_novel_pre"])))
            n400_post_means.append(float(np.mean(trial["n400_novel_post"])))
            novel_p600s.append(trial["novel_p600"])
            existing_p600s.append(trial["existing_p600_mean"])

            fo = trial["forgetting_overlaps"]
            if fo:
                min_forgetting.append(min(fo.values()))

        # Compute effect sizes
        n400_test = paired_ttest(n400_pre_means, n400_post_means)
        stab_summary = summarize(stabilities)
        forgetting_summary = summarize(min_forgetting) if min_forgetting else {"mean": 0.0}

        # Hypotheses
        h1 = stab_summary["mean"] > 0.8
        h2 = n400_test["d"] > 0.0 and float(np.mean(n400_post_means)) < float(np.mean(n400_pre_means))
        h3_novel_mean = float(np.mean(novel_p600s))
        h3_existing_mean = float(np.mean(existing_p600s))
        h3 = h3_novel_mean < h3_existing_mean * 3  # novel P600 within 3x of existing
        h4 = forgetting_summary["mean"] > 0.8

        self.log(f"\n  === Results ===")
        self.log(f"    Assembly stability:   {stab_summary['mean']:.3f} "
                 f"+/- {stab_summary.get('sem', 0):.3f}")
        self.log(f"    N400 pre-training:    {float(np.mean(n400_pre_means)):.3f}")
        self.log(f"    N400 post-training:   {float(np.mean(n400_post_means)):.3f}")
        self.log(f"    N400 improvement d:   {n400_test['d']:.2f}")
        self.log(f"    Novel P600:           {h3_novel_mean:.3f}")
        self.log(f"    Existing P600 (mean): {h3_existing_mean:.3f}")
        self.log(f"    Min forgetting overlap: {forgetting_summary['mean']:.3f}")

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Stable assembly > 0.8):      "
                 f"{'PASS' if h1 else 'FAIL'} ({stab_summary['mean']:.3f})")
        self.log(f"    H2 (N400 drops with training):   "
                 f"{'PASS' if h2 else 'FAIL'} (d={n400_test['d']:.2f})")
        self.log(f"    H3 (P600 comparable to existing):"
                 f" {'PASS' if h3 else 'FAIL'} "
                 f"({h3_novel_mean:.2f} vs {h3_existing_mean:.2f})")
        self.log(f"    H4 (No forgetting > 0.8):        "
                 f"{'PASS' if h4 else 'FAIL'} ({forgetting_summary['mean']:.3f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "assembly_stability": stab_summary,
            "n400_pre_mean": float(np.mean(n400_pre_means)),
            "n400_post_mean": float(np.mean(n400_post_means)),
            "n400_improvement_d": n400_test["d"],
            "novel_p600_mean": h3_novel_mean,
            "existing_p600_mean": h3_existing_mean,
            "min_forgetting_overlap": forgetting_summary,
            "hypotheses": {
                "H1_stable_assembly": h1,
                "H2_prediction_integration": h2,
                "H3_binding_integration": h3,
                "H4_no_forgetting": h4,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_train_sentences": cfg.n_train_sentences,
                "n_novel_train": cfg.n_novel_train,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Online Word Acquisition Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = OnlineWordAcquisitionExperiment(verbose=True)

    if args.quick:
        cfg = OnlineAcquisitionConfig(
            n=5000, k=50,
            n_train_sentences=50, n_novel_train=10,
            n_test_items=4)
        n_seeds = args.seeds or 3
    else:
        cfg = OnlineAcquisitionConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("ONLINE WORD ACQUISITION SUMMARY")
    print("=" * 70)
    print(f"\nH1 Stable assembly:       {'PASS' if h['H1_stable_assembly'] else 'FAIL'}")
    print(f"H2 Prediction integration: {'PASS' if h['H2_prediction_integration'] else 'FAIL'}")
    print(f"H3 Binding integration:    {'PASS' if h['H3_binding_integration'] else 'FAIL'}")
    print(f"H4 No forgetting:          {'PASS' if h['H4_no_forgetting'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
