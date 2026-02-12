"""
N400 Vocabulary Scaling — Does the Effect Survive Larger Vocabularies?

Tests whether the global-energy N400 finding (from test_n400_pre_kwta.py)
persists as vocabulary size increases and semantic categories become
denser and more overlapping.

Three vocabulary sizes:
- Small: 12 nouns (6 animals + 6 objects) — same as original
- Medium: 30 nouns (6 animals + 6 tools + 6 foods + 6 vehicles + 6 furniture)
- Large: 50 nouns (5 categories x 10 words each)

All nouns map to NOUN_CORE via visual grounding. As vocabulary grows,
more assemblies compete within the same neural area, potentially diluting
or strengthening the N400 effect.

The N400 literature shows smaller priming effects with weaker associations
and larger vocabularies, so we expect the effect to weaken but remain present.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import List

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.metrics import measure_pre_kwta_activation
from research.experiments.infrastructure import build_core_lexicon
from research.experiments.vocab.scaling import (
    build_small_vocab, build_medium_vocab, build_large_vocab,
    build_training_for_vocab, make_test_pairs,
)
from src.assembly_calculus.emergent import EmergentParser


@dataclass
class ScalingConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 3




# Vocabulary builders and test pair generators are now in the scaling module.
# See research/experiments/vocab/scaling.py for build_small_vocab,
# build_medium_vocab, build_large_vocab, build_training_for_vocab,
# make_test_pairs.


class N400VocabScalingExperiment(ExperimentBase):
    """Test N400 global energy across vocabulary sizes."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="n400_vocab_scaling",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = ScalingConfig()
        if quick:
            cfg.n_seeds = 2

        scales = [
            ("small", build_small_vocab),
            ("medium", build_medium_vocab),
            ("large", build_large_vocab),
        ]

        all_scale_results = {}

        for scale_name, vocab_fn in scales:
            vocab = vocab_fn()
            training = build_training_for_vocab(vocab)
            test_pairs = make_test_pairs(vocab)

            n_nouns = sum(1 for c in vocab.values()
                         if c.dominant_modality == "visual")
            self.log(f"\n{'='*60}")
            self.log(f"Scale: {scale_name} — {n_nouns} nouns, "
                     f"{len(vocab)} total words, "
                     f"{len(test_pairs)} test pairs, "
                     f"{len(training)} training sentences")
            self.log(f"{'='*60}")

            test_words = set()
            for t, r, u in test_pairs:
                test_words.update([t, r, u])

            energy_rel_seeds = []
            energy_unrel_seeds = []

            for seed_idx in range(cfg.n_seeds):
                seed = seed_idx
                self.log(f"  Seed {seed_idx + 1}/{cfg.n_seeds}")

                try:
                    parser = EmergentParser(
                        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                        seed=seed, rounds=cfg.rounds, vocabulary=vocab,
                    )
                    parser.train(sentences=training)

                    lexicon = build_core_lexicon(
                        parser, list(test_words), cfg.rounds)

                    e_r, e_u = [], []
                    for target, rel_prime, unrel_prime in test_pairs:
                        if target not in lexicon or rel_prime not in lexicon:
                            continue
                        if unrel_prime not in lexicon:
                            continue

                        m_rel = measure_pre_kwta_activation(
                            parser, rel_prime, target, lexicon, cfg.rounds)
                        m_unrel = measure_pre_kwta_activation(
                            parser, unrel_prime, target, lexicon, cfg.rounds)
                        e_r.append(m_rel["global_energy"])
                        e_u.append(m_unrel["global_energy"])

                    if e_r:
                        energy_rel_seeds.append(float(np.mean(e_r)))
                        energy_unrel_seeds.append(float(np.mean(e_u)))
                        self.log(f"    rel={np.mean(e_r):.1f}  "
                                 f"unrel={np.mean(e_u):.1f}  "
                                 f"({len(e_r)} pairs)")

                except Exception as e:
                    self.log(f"    FAILED: {e}")
                    continue

            # Analyze this scale
            scale_result = {
                "n_nouns": n_nouns,
                "n_vocab": len(vocab),
                "n_test_pairs": len(test_pairs),
                "n_training": len(training),
                "n_seeds": len(energy_rel_seeds),
            }

            if len(energy_rel_seeds) >= 2:
                stats = paired_ttest(energy_rel_seeds, energy_unrel_seeds)
                rel_s = summarize(energy_rel_seeds)
                unrel_s = summarize(energy_unrel_seeds)
                direction = "CORRECT" if rel_s["mean"] < unrel_s["mean"] else "REVERSED"
                scale_result.update({
                    "related": rel_s, "unrelated": unrel_s,
                    "test": stats, "direction": direction,
                })
                self.log(f"  RESULT: rel={rel_s['mean']:.1f}  "
                         f"unrel={unrel_s['mean']:.1f}  "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}  "
                         f"{direction}")
            else:
                scale_result["direction"] = "INSUFFICIENT_DATA"
                self.log("  RESULT: insufficient data")

            all_scale_results[scale_name] = scale_result

        # Summary
        self.log(f"\n{'='*60}")
        self.log("VOCABULARY SCALING SUMMARY")
        self.log(f"{'='*60}")
        self.log(f"{'Scale':<10} {'Nouns':<6} {'Pairs':<6} "
                 f"{'d':<9} {'p':<9} {'Direction'}")
        self.log("-" * 55)
        for scale_name, r in all_scale_results.items():
            d_val = r.get("test", {}).get("d", float("nan"))
            p_val = r.get("test", {}).get("p", float("nan"))
            direction = r.get("direction", "N/A")
            self.log(f"{scale_name:<10} {r['n_nouns']:<6} "
                     f"{r['n_test_pairs']:<6} "
                     f"{d_val:<9.3f} {p_val:<9.4f} {direction}")

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
            },
            metrics=all_scale_results,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="N400 Vocabulary Scaling Experiment")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds")
    args = parser.parse_args()

    exp = N400VocabScalingExperiment()
    exp.run(quick=args.quick)
