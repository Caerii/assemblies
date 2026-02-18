"""
Cross-Linguistic Word Order — SVO vs SOV P600 Effects

Prediction 3.4 from IMPLICATIONS_AND_PREDICTIONS.md:

Trains separate parsers on SVO ("dog chases cat") and SOV ("dog cat chases")
word orders, then tests each parser with both word orders. Each parser
should show normal P600 patterns for its own trained word order and elevated
P600 for the other word order.

Predictions:
  1. SVO parser + SVO input -> normal P600 pattern (cat > gram)
  2. SVO parser + SOV input -> elevated P600 (word order violation)
  3. SOV parser + SOV input -> normal P600 pattern (cat > gram)
  4. SOV parser + SVO input -> elevated P600 (word order violation)
  5. Cross-order P600 > same-order P600 for grammatical sentences

The specific structural areas showing instability should differ:
  - SVO parser with SOV: violation at object position (where verb expected)
  - SOV parser with SVO: violation at verb position (where object expected)

Literature:
  - Erdocia et al. 2009: SOV vs SVO P600 effects in Basque
  - Hagoort 2003: Word order violations elicit P600
  - Friederici 2002: P600 for structural reanalysis
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
from research.experiments.vocab import (
    build_svo_vocab, build_svo_sentences, build_sov_sentences,
)
from research.experiments.infrastructure import (
    bootstrap_structural_connectivity,
    consolidate_role_connections,
    consolidate_vp_connections,
)
from research.experiments.metrics.measurement import measure_critical_word
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP,
)


@dataclass
class CrossLingConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 5
    consolidation_passes: int = 10


def _make_svo_test_sentences(vocab):
    """SVO test sentences: "the SUBJ VERB the ___" with critical at object position."""
    return [
        {
            "frame": "the dog chases the ___",
            "context_words": ["the", "dog", "chases", "the"],
            "grammatical": "cat",
            "category_violation": "likes",
        },
        {
            "frame": "the cat sees the ___",
            "context_words": ["the", "cat", "sees", "the"],
            "grammatical": "bird",
            "category_violation": "finds",
        },
        {
            "frame": "the bird chases the ___",
            "context_words": ["the", "bird", "chases", "the"],
            "grammatical": "fish",
            "category_violation": "sees",
        },
        {
            "frame": "the horse finds the ___",
            "context_words": ["the", "horse", "finds", "the"],
            "grammatical": "mouse",
            "category_violation": "chases",
        },
    ]


def _make_sov_test_sentences(vocab):
    """SOV test sentences: "the SUBJ the ___ VERB" with critical at object position."""
    return [
        {
            "frame": "the dog the ___ chases",
            "context_words": ["the", "dog", "the"],
            "grammatical": "cat",              # noun in object slot (correct for SOV)
            "category_violation": "likes",     # verb in object slot (wrong category)
        },
        {
            "frame": "the cat the ___ sees",
            "context_words": ["the", "cat", "the"],
            "grammatical": "bird",
            "category_violation": "finds",
        },
        {
            "frame": "the bird the ___ chases",
            "context_words": ["the", "bird", "the"],
            "grammatical": "fish",
            "category_violation": "sees",
        },
        {
            "frame": "the horse the ___ finds",
            "context_words": ["the", "horse", "the"],
            "grammatical": "mouse",
            "category_violation": "chases",
        },
    ]


class CrossLinguisticExperiment(ExperimentBase):
    """Test P600 word order effects across SVO and SOV parsers."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="cross_linguistic",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = CrossLingConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = build_svo_vocab()
        svo_training = build_svo_sentences(vocab)
        sov_training = build_sov_sentences(vocab)
        svo_tests = _make_svo_test_sentences(vocab)
        sov_tests = _make_sov_test_sentences(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        # Conditions: (parser_type, training, test_type, test_sentences)
        conditions = [
            ("SVO_parser", svo_training, "SVO_input", svo_tests),  # same order
            ("SVO_parser", svo_training, "SOV_input", sov_tests),  # cross order
            ("SOV_parser", sov_training, "SOV_input", sov_tests),  # same order
            ("SOV_parser", sov_training, "SVO_input", svo_tests),  # cross order
        ]

        # Results: {key: {"gram": [...], "cat": [...]}}
        results_by_key = {}
        for _, _, test_type, _ in conditions:
            pass  # initialized per parser below

        # Group by parser type to avoid re-training
        parser_groups = {
            "SVO_parser": svo_training,
            "SOV_parser": sov_training,
        }

        for parser_label, training in parser_groups.items():
            # Get test conditions for this parser
            my_conditions = [
                (test_type, tests) for p, t, test_type, tests in conditions
                if p == parser_label
            ]

            for test_type, _ in my_conditions:
                key = f"{parser_label}+{test_type}"
                results_by_key[key] = {"gram": [], "cat": []}

            for seed_idx, seed in enumerate(seeds):
                self.log(f"\n=== {parser_label} — Seed {seed_idx + 1}/{len(seeds)} ===")

                parser = EmergentParser(
                    n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                    seed=seed, rounds=cfg.rounds, vocabulary=vocab,
                )
                parser.train(sentences=training)

                bootstrap_structural_connectivity(
                    parser, p600_areas, log_fn=self.log)
                consolidate_role_connections(
                    parser, training, n_passes=cfg.consolidation_passes,
                    log_fn=self.log)
                consolidate_vp_connections(
                    parser, training, n_passes=cfg.consolidation_passes,
                    log_fn=self.log)

                for test_type, tests in my_conditions:
                    key = f"{parser_label}+{test_type}"
                    gram_vals, cat_vals = [], []

                    for test in tests:
                        for cond_key, accumulator in [
                            ("grammatical", gram_vals),
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
                            accumulator.append(
                                result["p600_mean_instability"])

                    if gram_vals:
                        results_by_key[key]["gram"].append(
                            float(np.mean(gram_vals)))
                        results_by_key[key]["cat"].append(
                            float(np.mean(cat_vals)))
                        self.log(f"  {test_type}: gram={np.mean(gram_vals):.3f}  "
                                 f"cat={np.mean(cat_vals):.3f}")

        # -- Analysis --
        self.log(f"\n{'='*60}")
        self.log("CROSS-LINGUISTIC RESULTS")
        self.log(f"{'='*60}")

        metrics = {}

        self.log(f"\n{'Condition':<30} {'P600_gram':<12} {'P600_cat':<12} "
                 f"{'cat-gram':<12}")
        self.log("-" * 70)

        for key, r in results_by_key.items():
            if len(r["gram"]) >= 2:
                gs = summarize(r["gram"])
                cs = summarize(r["cat"])
                diff = cs["mean"] - gs["mean"]
                self.log(f"{key:<30} {gs['mean']:<12.4f} "
                         f"{cs['mean']:<12.4f} {diff:<12.4f}")
                metrics[key] = {"gram": gs, "cat": cs}

        # Prediction tests
        self.log(f"\n--- Predictions ---")

        # P1 + P3: Same-order parsers show cat > gram
        for parser_type, order in [("SVO_parser", "SVO_input"),
                                    ("SOV_parser", "SOV_input")]:
            key = f"{parser_type}+{order}"
            r = results_by_key.get(key, {})
            if len(r.get("gram", [])) >= 2:
                stats = paired_ttest(r["cat"], r["gram"])
                cat_gt_gram = np.mean(r["cat"]) > np.mean(r["gram"])
                self.log(f"  {key} — cat > gram: "
                         f"{'YES' if cat_gt_gram else 'NO'}  "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics[f"{key}_cat_vs_gram"] = {
                    "test": stats, "confirmed": cat_gt_gram,
                }

        # P5: Cross-order gram > same-order gram
        for parser_type in ["SVO_parser", "SOV_parser"]:
            same_order = "SVO_input" if parser_type == "SVO_parser" else "SOV_input"
            cross_order = "SOV_input" if parser_type == "SVO_parser" else "SVO_input"
            same_key = f"{parser_type}+{same_order}"
            cross_key = f"{parser_type}+{cross_order}"

            same_r = results_by_key.get(same_key, {})
            cross_r = results_by_key.get(cross_key, {})

            if (len(same_r.get("gram", [])) >= 2 and
                    len(cross_r.get("gram", [])) >= 2):
                stats = paired_ttest(cross_r["gram"], same_r["gram"])
                cross_higher = np.mean(cross_r["gram"]) > np.mean(same_r["gram"])
                self.log(f"  {parser_type} — cross > same (gram): "
                         f"{'YES' if cross_higher else 'NO'}  "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics[f"{parser_type}_cross_vs_same_gram"] = {
                    "test": stats, "confirmed": cross_higher,
                }

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "p600_settling_rounds": cfg.p600_settling_rounds,
                "consolidation_passes": cfg.consolidation_passes,
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Linguistic Word Order Experiment (Prediction 3.4)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = CrossLinguisticExperiment()
    exp.run(quick=args.quick)
