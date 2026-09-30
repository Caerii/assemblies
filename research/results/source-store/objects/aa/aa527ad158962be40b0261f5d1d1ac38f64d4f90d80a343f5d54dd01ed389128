"""
Sentence Generation: From Comprehension to Production

Tests whether the bidirectional pathways created during training support
sentence generation — going from structural representations back to words.

Two generation modes:
  1. Role-based: Activate ROLE_AGENT -> project to NOUN_CORE -> identify word
  2. Prediction chain: Start with noun, chain forward predictions to produce
     a word sequence

Tested at each developmental stage (same cumulative brain as curriculum):
  1. Single words (no training yet)
  2. Two-word combinations
  3. SVO
  4. SVO+PP
  5. SRC
  6. ORC

Hypotheses:
  H1: Role accuracy increases from ~chance to >0.8 by stage 3
  H2: Chain grammaticality increases with training stage
  H3: System produces novel grammatical sequences not in training data

Usage:
    uv run python research/experiments/primitives/test_sentence_generation.py
    uv run python research/experiments/primitives/test_sentence_generation.py --quick
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
from research.experiments.lib.generation import (
    build_core_lexicon,
    readout_from_role,
    generate_from_prediction_chain,
    score_generation,
    check_novelty,
)


STAGE_NAMES = [
    "1_single_words",
    "2_two_word",
    "3_svo",
    "4_svo_pp",
    "5_src",
    "6_orc",
]


@dataclass
class GenerationConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    lexicon_readout_rounds: int = 5
    # Training (same as curriculum)
    stage_sentences: tuple = (0, 30, 60, 60, 50, 50)
    training_reps: int = 3
    # Generation
    n_role_tests: int = 10
    n_chain_tests: int = 5
    max_chain_length: int = 5


def generate_stage_sentences(stage_idx, cfg, vocab, rng):
    """Generate training sentences for a given stage."""
    n_sents = cfg.stage_sentences[stage_idx] * cfg.training_reps

    if stage_idx == 0:
        return []
    if stage_idx == 1:
        nouns = vocab.words_for_category("NOUN")
        verbs = vocab.words_for_category("VERB")
        sentences = []
        for _ in range(n_sents):
            sentences.append({
                "words": [rng.choice(nouns), rng.choice(verbs)],
                "roles": ["AGENT", "VERB"],
                "categories": ["NOUN", "VERB"],
                "has_pp": False,
            })
        return sentences

    grammar_params = {
        2: dict(pp_prob=0.0, rel_prob=0.0, orc_prob=0.0, max_pp_depth=1),
        3: dict(pp_prob=0.6, rel_prob=0.0, orc_prob=0.0, max_pp_depth=1),
        4: dict(pp_prob=0.3, rel_prob=0.5, orc_prob=0.0, max_pp_depth=1),
        5: dict(pp_prob=0.3, rel_prob=0.5, orc_prob=0.5, max_pp_depth=1),
    }
    grammar = RecursiveCFG(vocab=vocab, rng=rng, **grammar_params[stage_idx])
    return grammar.generate_batch(n_sents)


def measure_generation(
    brain,
    vocab,
    prediction_lexicon,
    core_lexicon,
    all_training_sentences,
    cfg,
    rng,
) -> Dict[str, Any]:
    """Measure generation quality at current training state."""
    nouns = vocab.words_for_category("NOUN")
    noun_set = set(nouns)
    if "LOCATION" in vocab.categories:
        noun_set |= set(vocab.words_for_category("LOCATION"))

    # Role-based generation: read from ROLE_AGENT and ROLE_PATIENT
    role_agent_correct = 0
    role_patient_correct = 0
    role_confidences = []

    for i in range(cfg.n_role_tests):
        # ROLE_AGENT -> NOUN_CORE
        word_a, conf_a = readout_from_role(
            brain, "ROLE_AGENT", "NOUN_CORE", vocab, core_lexicon)
        if word_a in noun_set:
            role_agent_correct += 1
        role_confidences.append(conf_a)

        # ROLE_PATIENT -> NOUN_CORE
        word_p, conf_p = readout_from_role(
            brain, "ROLE_PATIENT", "NOUN_CORE", vocab, core_lexicon)
        if word_p in noun_set:
            role_patient_correct += 1
        role_confidences.append(conf_p)

    role_accuracy = (role_agent_correct + role_patient_correct) / (2 * cfg.n_role_tests)

    # Prediction chain generation
    chain_coherences = []
    chain_grammaticalities = []
    chain_novelties = []
    example_chains = []

    for start_noun in nouns[:cfg.n_chain_tests]:
        chain = generate_from_prediction_chain(
            brain, start_noun, vocab, prediction_lexicon,
            max_length=cfg.max_chain_length)

        words = [w for w, _ in chain]
        confidences = [c for _, c in chain]
        chain_coherences.append(float(np.mean(confidences[1:])) if len(confidences) > 1 else 0.0)

        score = score_generation(words, vocab)
        chain_grammaticalities.append(score["category_accuracy"])
        chain_novelties.append(
            1.0 if check_novelty(words, all_training_sentences) else 0.0)
        example_chains.append(words)

    return {
        "role_accuracy": role_accuracy,
        "role_mean_confidence": float(np.mean(role_confidences)),
        "chain_coherence": float(np.mean(chain_coherences)),
        "chain_grammaticality": float(np.mean(chain_grammaticalities)),
        "chain_novelty": float(np.mean(chain_novelties)),
        "chain_svo_rate": sum(
            1 for ch in example_chains
            if score_generation(ch, vocab)["is_svo"]
        ) / max(len(example_chains), 1),
        "example_chains": example_chains[:3],
    }


def run_trial(
    cfg: GenerationConfig,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Run one generation trial: train through stages, test generation at each."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    stage_results = {}
    all_training_sentences = []

    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        sentences = generate_stage_sentences(stage_idx, cfg, vocab, rng)
        for sent in sentences:
            train_sentence(brain, sent, vocab,
                           cfg.train_rounds_per_pair, cfg.binding_rounds)
        all_training_sentences.extend(sentences)

        brain.disable_plasticity = True
        prediction_lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
        core_lexicon = build_core_lexicon(brain, vocab)

        gen = measure_generation(
            brain, vocab, prediction_lexicon, core_lexicon,
            all_training_sentences, cfg, rng)
        gen["n_trained"] = len(all_training_sentences)
        brain.disable_plasticity = False

        stage_results[stage_name] = gen

    return stage_results


class SentenceGenerationExperiment(ExperimentBase):
    """Sentence generation at each developmental stage."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="sentence_generation",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or GenerationConfig(
            **{k: v for k, v in kwargs.items()
               if k in GenerationConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Sentence Generation: From Comprehension to Production")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  stage_sentences={cfg.stage_sentences}")
        self.log(f"  n_role_tests={cfg.n_role_tests}, n_chain_tests={cfg.n_chain_tests}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = ["role_accuracy", "role_mean_confidence",
                "chain_coherence", "chain_grammaticality",
                "chain_novelty", "chain_svo_rate"]
        stage_vals = {s: {k: [] for k in keys} for s in STAGE_NAMES}
        example_chains_all = {s: [] for s in STAGE_NAMES}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)
            for stage_name, gen in trial.items():
                for k in keys:
                    stage_vals[stage_name][k].append(gen[k])
                if s == 0:
                    example_chains_all[stage_name] = gen.get("example_chains", [])

        # Report
        self.log(f"\n  {'Stage':<18s} | {'RoleAcc':>7s} | {'ChainCoh':>8s} | "
                 f"{'Gram':>5s} | {'SVO':>5s} | {'Novel':>5s}")
        self.log("  " + "-" * 60)

        for stage_name in STAGE_NAMES:
            sv = stage_vals[stage_name]
            self.log(
                f"  {stage_name:<18s} | "
                f"{np.mean(sv['role_accuracy']):7.3f} | "
                f"{np.mean(sv['chain_coherence']):8.3f} | "
                f"{np.mean(sv['chain_grammaticality']):5.3f} | "
                f"{np.mean(sv['chain_svo_rate']):5.3f} | "
                f"{np.mean(sv['chain_novelty']):5.3f}"
            )

        # Show example chains from seed 0
        self.log(f"\n  === Example Chains (seed 0) ===")
        for stage_name in STAGE_NAMES:
            chains = example_chains_all[stage_name]
            if chains:
                for ch in chains[:2]:
                    self.log(f"    {stage_name}: {' '.join(ch)}")

        # Hypotheses
        final = stage_vals[STAGE_NAMES[-1]]
        stage3 = stage_vals["3_svo"]
        stage1 = stage_vals["1_single_words"]

        h1 = np.mean(stage3["role_accuracy"]) > 0.8
        h2 = np.mean(final["chain_grammaticality"]) > np.mean(
            stage1["chain_grammaticality"]) + 0.1
        h3 = np.mean(final["chain_novelty"]) > 0.3

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Role acc > 0.8 by SVO):   "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" ({np.mean(stage3['role_accuracy']):.3f})")
        self.log(f"    H2 (Gram improves):            "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" ({np.mean(stage1['chain_grammaticality']):.3f}"
                 f" -> {np.mean(final['chain_grammaticality']):.3f})")
        self.log(f"    H3 (Novel sequences > 0.3):    "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" ({np.mean(final['chain_novelty']):.3f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "stage_metrics": {
                stage: {k: summarize(v) for k, v in sv.items()}
                for stage, sv in stage_vals.items()
            },
            "example_chains": example_chains_all,
            "hypotheses": {
                "H1_role_accuracy": h1,
                "H2_gram_improves": h2,
                "H3_novelty": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "stage_sentences": list(cfg.stage_sentences),
                "training_reps": cfg.training_reps,
                "n_role_tests": cfg.n_role_tests,
                "n_chain_tests": cfg.n_chain_tests,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Sentence Generation Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = SentenceGenerationExperiment(verbose=True)

    if args.quick:
        cfg = GenerationConfig(
            n=5000, k=50,
            stage_sentences=(0, 15, 30, 30, 25, 25),
            training_reps=2,
            n_role_tests=5, n_chain_tests=3)
        n_seeds = args.seeds or 3
    else:
        cfg = GenerationConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("SENTENCE GENERATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Stage':<18s} | {'RoleAcc':>7s} | {'Gram':>5s} | {'SVO':>5s} | {'Novel':>5s}")
    print("-" * 55)
    for stage_name in STAGE_NAMES:
        sm = m["stage_metrics"][stage_name]
        print(f"{stage_name:<18s} | "
              f"{sm['role_accuracy']['mean']:7.3f} | "
              f"{sm['chain_grammaticality']['mean']:5.3f} | "
              f"{sm['chain_svo_rate']['mean']:5.3f} | "
              f"{sm['chain_novelty']['mean']:5.3f}")

    print(f"\nExample chains:")
    for stage_name in STAGE_NAMES:
        chains = m["example_chains"][stage_name]
        if chains:
            print(f"  {stage_name}: {' '.join(chains[0])}")

    h = m["hypotheses"]
    print(f"\nH1 Role accuracy: {'PASS' if h['H1_role_accuracy'] else 'FAIL'}")
    print(f"H2 Gram improves: {'PASS' if h['H2_gram_improves'] else 'FAIL'}")
    print(f"H3 Novelty:       {'PASS' if h['H3_novelty'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
