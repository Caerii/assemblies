"""
L2 Processing — Unconsolidated Grammar Produces Elevated P600

Prediction 3.5 from IMPLICATIONS_AND_PREDICTIONS.md:

Compares a fully consolidated parser (L1 native speaker) with an
unconsolidated parser (L2 beginner) on the same test sentences.

The unconsolidated parser has been trained (episodic encoding) but NOT
consolidated (no systems consolidation). This means all core->structural
pathways remain at random baseline — even for grammatical sentences,
structural integration must proceed through unstrengthened connections.

Predictions:
  1. L2 shows elevated P600 for ALL conditions, including grammatical
  2. P600(L2_gram) approx P600(L1_sem_viol) — L2 processing grammatical
     sentences is as effortful as L1 processing semantic violations
  3. L1 shows the standard P600 pattern: cat > sem > gram
  4. L2 shows compressed/elevated pattern: all conditions high

This maps to the L2 processing literature:
  - Steinhauer et al. 2009: L2 learners show delayed/reduced P600
  - Tanner et al. 2014: L2 P600 depends on proficiency
  - Clahsen & Felser 2006: Shallow structure hypothesis
  - Hahne 2001: L2 ERP signatures differ from L1
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.vocab import build_svo_vocab, build_svo_sentences
from research.experiments.infrastructure import (
    bootstrap_structural_connectivity,
    consolidate_role_connections,
    consolidate_vp_connections,
)
from research.experiments.metrics.measurement import measure_critical_word
from research.experiments.applications.test_p600_syntactic import (
    _make_test_sentences,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP,
)


@dataclass
class L2Config:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 5
    l1_consolidation_passes: int = 10
    l2_consolidation_passes: int = 0


class L2ProcessingExperiment(ExperimentBase):
    """Compare L1 (consolidated) vs L2 (unconsolidated) P600 patterns."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="l2_processing",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = L2Config()
        if quick:
            cfg.n_seeds = 3

        vocab = build_svo_vocab()
        training = build_svo_sentences(vocab)
        test_sentences = _make_test_sentences(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        conditions = [
            ("L1", cfg.l1_consolidation_passes),
            ("L2", cfg.l2_consolidation_passes),
        ]

        # Per-condition, per-seed accumulators
        results_by_cond = {}
        for cond_label, _ in conditions:
            results_by_cond[cond_label] = {
                "gram": [], "sem": [], "cat": [],
            }

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            for cond_label, n_passes in conditions:
                self.log(f"  {cond_label} (n_passes={n_passes})")

                parser = EmergentParser(
                    n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                    seed=seed, rounds=cfg.rounds, vocabulary=vocab,
                )
                parser.train(sentences=training)

                bootstrap_structural_connectivity(
                    parser, p600_areas, log_fn=self.log)
                consolidate_role_connections(
                    parser, training, n_passes=n_passes, log_fn=self.log)
                consolidate_vp_connections(
                    parser, training, n_passes=n_passes, log_fn=self.log)

                gram_vals, sem_vals, cat_vals = [], [], []

                for test in test_sentences:
                    for cond_key, accumulator in [
                        ("grammatical", gram_vals),
                        ("semantic_violation", sem_vals),
                        ("category_violation", cat_vals),
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
                    results_by_cond[cond_label]["gram"].append(
                        float(np.mean(gram_vals)))
                    results_by_cond[cond_label]["sem"].append(
                        float(np.mean(sem_vals)))
                    results_by_cond[cond_label]["cat"].append(
                        float(np.mean(cat_vals)))
                    self.log(f"    gram={np.mean(gram_vals):.3f}  "
                             f"sem={np.mean(sem_vals):.3f}  "
                             f"cat={np.mean(cat_vals):.3f}")

        # -- Analysis --
        self.log(f"\n{'='*60}")
        self.log("L2 PROCESSING RESULTS")
        self.log(f"{'='*60}")

        metrics = {}

        # Summary table
        self.log(f"\n{'Condition':<12} {'P600_gram':<12} {'P600_sem':<12} "
                 f"{'P600_cat':<12}")
        self.log("-" * 50)

        for cond_label in ["L1", "L2"]:
            r = results_by_cond[cond_label]
            if len(r["gram"]) >= 2:
                gs = summarize(r["gram"])
                ss = summarize(r["sem"])
                cs = summarize(r["cat"])
                self.log(f"{cond_label:<12} {gs['mean']:<12.4f} "
                         f"{ss['mean']:<12.4f} {cs['mean']:<12.4f}")
                metrics[cond_label] = {
                    "gram": gs, "sem": ss, "cat": cs,
                }

        # Prediction tests
        l1 = results_by_cond["L1"]
        l2 = results_by_cond["L2"]

        if len(l1["gram"]) >= 2 and len(l2["gram"]) >= 2:
            self.log(f"\n--- Predictions ---")

            # P1: L2_gram > L1_gram
            stats_gram = paired_ttest(l2["gram"], l1["gram"])
            l2_gram_higher = np.mean(l2["gram"]) > np.mean(l1["gram"])
            self.log(f"P1 — L2_gram > L1_gram: "
                     f"{'YES' if l2_gram_higher else 'NO'}  "
                     f"d={stats_gram['d']:.3f}  p={stats_gram['p']:.4f}")
            metrics["P1_l2_gram_gt_l1_gram"] = {
                "test": stats_gram,
                "confirmed": l2_gram_higher,
            }

            # P2: L2_gram approx L1_sem
            l2_gram_mean = np.mean(l2["gram"])
            l1_sem_mean = np.mean(l1["sem"])
            ratio = l2_gram_mean / l1_sem_mean if l1_sem_mean > 0 else 0
            approx = 0.5 < ratio < 2.0
            stats_l2g_l1s = paired_ttest(l2["gram"], l1["sem"])
            self.log(f"P2 — L2_gram approx L1_sem: "
                     f"{'YES' if approx else 'NO'}  "
                     f"(ratio={ratio:.3f})  "
                     f"d={stats_l2g_l1s['d']:.3f}  p={stats_l2g_l1s['p']:.4f}")
            metrics["P2_l2_gram_approx_l1_sem"] = {
                "test": stats_l2g_l1s,
                "ratio": float(ratio),
                "approximate": approx,
            }

            # P3: L1 shows standard pattern (cat > sem > gram)
            l1_pattern = (np.mean(l1["cat"]) > np.mean(l1["sem"]) >
                          np.mean(l1["gram"]))
            self.log(f"P3 — L1 pattern cat > sem > gram: "
                     f"{'YES' if l1_pattern else 'NO'}")
            metrics["P3_l1_standard_pattern"] = l1_pattern

            # P4: L2 shows compressed pattern (all elevated)
            l2_gram_mean = np.mean(l2["gram"])
            l2_cat_mean = np.mean(l2["cat"])
            l1_gram_mean = np.mean(l1["gram"])
            l1_cat_mean = np.mean(l1["cat"])
            l1_range = l1_cat_mean - l1_gram_mean
            l2_range = l2_cat_mean - l2_gram_mean
            compressed = l2_range < l1_range if l1_range > 0 else False
            self.log(f"P4 — L2 compressed pattern: "
                     f"{'YES' if compressed else 'NO'}  "
                     f"(L1 range={l1_range:.4f}  L2 range={l2_range:.4f})")
            metrics["P4_l2_compressed"] = {
                "compressed": compressed,
                "l1_range": float(l1_range),
                "l2_range": float(l2_range),
            }

            # Cross-condition comparisons
            self.log(f"\n--- Cross-condition comparisons ---")
            for cond in ["gram", "sem", "cat"]:
                stats = paired_ttest(l2[cond], l1[cond])
                direction = ("L2 > L1" if np.mean(l2[cond]) > np.mean(l1[cond])
                             else "L1 > L2")
                self.log(f"  {cond}: d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                metrics[f"l2_vs_l1_{cond}"] = {
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
                "l1_consolidation_passes": cfg.l1_consolidation_passes,
                "l2_consolidation_passes": cfg.l2_consolidation_passes,
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L2 Processing Experiment (Prediction 3.5)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = L2ProcessingExperiment()
    exp.run(quick=args.quick)
