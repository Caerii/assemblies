"""
Developmental Trajectory — P600 as a Function of Consolidation Rounds

Prediction 3.6 from IMPLICATIONS_AND_PREDICTIONS.md:

Varies the number of consolidation passes (0, 1, 5, 10, 50) to model
developmental stages. More consolidation = more syntactic experience.

Predictions:
  1. P600 for grammatical sentences DECREASES with more consolidation
     (stronger Hebbian connections -> faster structural convergence)
  2. P600 for category violations stays CONSTANT regardless of consolidation
     (VERB_CORE->ROLE_* is never consolidated, so instability doesn't change)
  3. The separation between grammatical and violation P600 INCREASES
     with consolidation (developmental maturation of the P600 component)

This maps to developmental ERP literature:
  - Hahne et al. 2004: Children show delayed P600 relative to adults
  - Friedrich & Friederici 2005: P600 matures with syntactic experience
  - Osterhout et al. 1997: Proficiency modulates P600 amplitude

The model predicts:
  P600_gram(consolidation) ~ 1/log(passes)   (decreasing)
  P600_cat(consolidation) ~ constant           (flat)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.vocab import build_svo_vocab, build_svo_sentences
from research.experiments.infrastructure import setup_p600_pipeline
from research.experiments.metrics.measurement import measure_critical_word
from research.experiments.vocab import make_p600_test_sentences
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP,
)


@dataclass
class DevTrajectoryConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 5
    consolidation_levels: Tuple[int, ...] = (0, 1, 5, 10, 50)


class DevelopmentalTrajectoryExperiment(ExperimentBase):
    """Test P600 as a function of consolidation rounds."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="developmental_trajectory",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = DevTrajectoryConfig()
        if quick:
            cfg.n_seeds = 3
            cfg.consolidation_levels = (0, 1, 10)

        vocab = build_svo_vocab()
        training = build_svo_sentences(vocab)
        test_sentences = make_p600_test_sentences(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        # Results: {n_passes: {"gram": [...], "sem": [...], "cat": [...]}}
        all_results = {}

        for n_passes in cfg.consolidation_levels:
            self.log(f"\n{'='*60}")
            self.log(f"Consolidation passes: {n_passes}")
            self.log(f"{'='*60}")

            inst_gram_seeds = []
            inst_sem_seeds = []
            inst_cat_seeds = []

            for seed_idx, seed in enumerate(seeds):
                self.log(f"  Seed {seed_idx + 1}/{len(seeds)}")

                parser = EmergentParser(
                    n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                    seed=seed, rounds=cfg.rounds, vocabulary=vocab,
                )
                parser.train(sentences=training)

                setup_p600_pipeline(
                    parser, training, p600_areas,
                    consolidation_passes=n_passes, log_fn=self.log)

                # Measure P600 for each test sentence
                gram_vals, sem_vals, cat_vals = [], [], []

                for test in test_sentences:
                    for cond_label, cond_key, accumulator in [
                        ("gram", "grammatical", gram_vals),
                        ("sem", "semantic_violation", sem_vals),
                        ("cat", "category_violation", cat_vals),
                    ]:
                        result = measure_critical_word(
                            parser,
                            test["context_words"],
                            test[cond_key],
                            p600_areas,
                            cfg.rounds,
                            cfg.p600_settling_rounds,
                        )
                        accumulator.append(result["p600_mean_instability"])

                if gram_vals:
                    inst_gram_seeds.append(float(np.mean(gram_vals)))
                    inst_sem_seeds.append(float(np.mean(sem_vals)))
                    inst_cat_seeds.append(float(np.mean(cat_vals)))
                    self.log(f"    gram={np.mean(gram_vals):.3f}  "
                             f"sem={np.mean(sem_vals):.3f}  "
                             f"cat={np.mean(cat_vals):.3f}")

            all_results[n_passes] = {
                "gram": inst_gram_seeds,
                "sem": inst_sem_seeds,
                "cat": inst_cat_seeds,
            }

        # -- Analysis --
        self.log(f"\n{'='*60}")
        self.log("DEVELOPMENTAL TRAJECTORY RESULTS")
        self.log(f"{'='*60}")
        self.log("Prediction: P600_gram decreases with consolidation; "
                 "P600_cat stays constant")

        metrics = {}

        self.log(f"\n{'Passes':<8} {'P600_gram':<12} {'P600_sem':<12} "
                 f"{'P600_cat':<12}")
        self.log("-" * 50)

        gram_means = []
        sem_means = []
        cat_means = []

        for n_passes in cfg.consolidation_levels:
            r = all_results[n_passes]
            if len(r["gram"]) >= 2:
                gs = summarize(r["gram"])
                ss = summarize(r["sem"])
                cs = summarize(r["cat"])
                self.log(f"{n_passes:<8} {gs['mean']:<12.4f} "
                         f"{ss['mean']:<12.4f} {cs['mean']:<12.4f}")
                gram_means.append(gs["mean"])
                sem_means.append(ss["mean"])
                cat_means.append(cs["mean"])
                metrics[f"passes_{n_passes}"] = {
                    "gram": gs, "sem": ss, "cat": cs,
                }
            else:
                self.log(f"{n_passes:<8} insufficient data")

        # Check predictions
        if len(gram_means) >= 2:
            # Prediction 1: gram_means should decrease
            gram_decreasing = all(
                gram_means[i] >= gram_means[i + 1]
                for i in range(len(gram_means) - 1))
            # Weaker check: first > last
            gram_first_gt_last = gram_means[0] > gram_means[-1]

            # Prediction 2: cat_means should stay flat
            cat_range = max(cat_means) - min(cat_means)
            cat_mean_val = np.mean(cat_means)
            cat_flat = (cat_range / cat_mean_val < 0.20) if cat_mean_val > 0 else True

            # Prediction 3: separation increases
            sep_first = cat_means[0] - gram_means[0]
            sep_last = cat_means[-1] - gram_means[-1]
            sep_increases = sep_last > sep_first

            self.log(f"\nPrediction 1 — P600_gram decreases: "
                     f"{'MONOTONIC' if gram_decreasing else 'NON-MONOTONIC'} "
                     f"(first > last: {gram_first_gt_last})")
            self.log(f"  gram means: {' -> '.join(f'{m:.4f}' for m in gram_means)}")
            self.log(f"Prediction 2 — P600_cat stays flat: "
                     f"{'YES' if cat_flat else 'NO'} "
                     f"(range/mean = {cat_range/cat_mean_val:.3f})"
                     if cat_mean_val > 0 else "N/A")
            self.log(f"  cat means: {' -> '.join(f'{m:.4f}' for m in cat_means)}")
            self.log(f"Prediction 3 — Separation increases: "
                     f"{'YES' if sep_increases else 'NO'} "
                     f"(first={sep_first:.4f} last={sep_last:.4f})")

            metrics["predictions"] = {
                "gram_monotonically_decreasing": gram_decreasing,
                "gram_first_gt_last": gram_first_gt_last,
                "cat_flat": cat_flat,
                "cat_range_over_mean": float(cat_range / cat_mean_val) if cat_mean_val > 0 else 0.0,
                "separation_increases": sep_increases,
                "separation_first": float(sep_first),
                "separation_last": float(sep_last),
            }

            # Pairwise: first vs last consolidation level
            first_key = cfg.consolidation_levels[0]
            last_key = cfg.consolidation_levels[-1]
            if (len(all_results[first_key]["gram"]) >= 2 and
                    len(all_results[last_key]["gram"]) >= 2):
                self.log(f"\nFirst ({first_key}) vs Last ({last_key}) "
                         f"consolidation comparison:")
                for cond in ["gram", "sem", "cat"]:
                    stats = paired_ttest(
                        all_results[first_key][cond],
                        all_results[last_key][cond],
                    )
                    direction = ("DECREASED" if
                                 np.mean(all_results[first_key][cond]) >
                                 np.mean(all_results[last_key][cond])
                                 else "INCREASED")
                    self.log(f"  {cond}: d={stats['d']:.3f}  "
                             f"p={stats['p']:.4f}  {direction}")
                    metrics[f"{cond}_first_vs_last"] = {
                        "test": stats, "direction": direction,
                    }

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "p600_settling_rounds": cfg.p600_settling_rounds,
                "consolidation_levels": list(cfg.consolidation_levels),
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Developmental Trajectory Experiment (Prediction 3.6)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds and consolidation levels")
    args = parser.parse_args()

    exp = DevelopmentalTrajectoryExperiment()
    exp.run(quick=args.quick)
