"""
NEMO Benchmark Experiment (Tier 3)

Benchmarks our numpy_sparse EmergentParser against the reference NEMO
architecture (Mitropolsky & Papadimitriou 2025) across classification
accuracy, convergence speed, assembly stability, and generation quality.

Scientific Question:
    How does our numpy_sparse EmergentParser compare to the reference NEMO
    architecture in classification accuracy, convergence speed, and assembly
    quality?

Hypotheses:
    H1: Classification accuracy >= 80% on the standard vocabulary.
    H2: Role assignment accuracy >= 70% on standard transitive sentences.
    H3: Assembly stability (overlap after 20 autonomous rounds) >= 0.90.
    H4: Training converges within 5 passes over the corpus.
    H5: Generation roundtrip accuracy >= 50%.

Protocol:
    For each seed:
    1. Create EmergentParser with standard parameters (n=10000, k=100).
    2. Build vocabulary using build_vocabulary().
    3. Train using the standard train() pipeline.
    4. Run EvaluationSuite.full_evaluation().
    5. Measure assembly stability, convergence speed, per-category P/R/F1.

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from research.experiments.base import ExperimentBase, ExperimentResult, summarize, ttest_vs_null
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.evaluation import EvaluationSuite
from src.assembly_calculus.emergent.vocabulary_builder import build_vocabulary
from src.assembly_calculus.emergent.training_data import generate_training_sentences
from src.assembly_calculus.emergent.areas import CORE_TO_CATEGORY, GROUNDING_TO_CORE
from src.assembly_calculus.ops import project, _snap


@dataclass
class NEMOBenchmarkConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10
    n_seeds: int = 5
    stability_test_rounds: int = 20


class NEMOBenchmarkExperiment(ExperimentBase):
    """Benchmark EmergentParser against reference NEMO architecture."""

    def __init__(self, results_dir: Optional[Path] = None,
                 seed: int = 42, verbose: bool = True):
        super().__init__(
            name="nemo_benchmark", seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "applications"),
            verbose=verbose,
        )

    def _measure_assembly_stability(
        self, parser: EmergentParser,
        test_words: List[str], config: NEMOBenchmarkConfig,
    ) -> Dict[str, Any]:
        """Project word, snap, run autonomous recurrence, snap again, measure overlap."""
        per_word: Dict[str, float] = {}
        for word in test_words:
            ctx = parser.word_grounding.get(word)
            if ctx is None:
                continue
            core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
            phon = parser.stim_map.get(word)
            if phon is None:
                continue

            stim_dict = {phon: [core_area]}
            for gs in parser._grounding_stim_names(ctx):
                if gs in parser._grounding_stim_names_set:
                    stim_dict[gs] = [core_area]

            # Initial projection
            parser.brain.project(stim_dict, {})
            if config.rounds > 1:
                parser.brain.project_rounds(
                    target=core_area, areas_by_stim=stim_dict,
                    dst_areas_by_src_area={core_area: [core_area]},
                    rounds=config.rounds - 1,
                )
            initial_asm = _snap(parser.brain, core_area)

            # Autonomous recurrence
            for _ in range(config.stability_test_rounds):
                parser.brain.project({}, {core_area: [core_area]})
            final_asm = _snap(parser.brain, core_area)

            initial_set = set(initial_asm.winners.tolist())
            final_set = set(final_asm.winners.tolist())
            if initial_set and final_set:
                overlap = len(initial_set & final_set) / min(len(initial_set), len(final_set))
            else:
                overlap = 0.0
            per_word[word] = overlap
            parser.brain._engine.reset_area_connections(core_area)

        overlaps = list(per_word.values())
        return {
            "per_word": per_word,
            "mean": float(np.mean(overlaps)) if overlaps else 0.0,
            "min": float(np.min(overlaps)) if overlaps else 0.0,
            "max": float(np.max(overlaps)) if overlaps else 0.0,
        }

    def _measure_convergence(
        self, vocab: Dict[str, GroundingContext],
        config: NEMOBenchmarkConfig, seed: int, max_passes: int = 5,
    ) -> Dict[str, Any]:
        """Train with 1..max_passes corpus repetitions, measure accuracy at each."""
        sentences = generate_training_sentences(vocab, n_sentences=100, seed=seed)
        passes_list = list(range(1, max_passes + 1))
        accuracies: List[float] = []

        for n_passes in passes_list:
            p = EmergentParser(
                n=config.n, k=config.k, p=config.p,
                beta=config.beta, seed=seed, rounds=config.rounds,
                vocabulary=vocab,
            )
            p.train(sentences=sentences * n_passes)

            suite = EvaluationSuite(p)
            test_vocab = {}
            for core_area, lex in p.core_lexicons.items():
                cat = CORE_TO_CATEGORY.get(core_area)
                if cat:
                    for w in lex:
                        test_vocab[w] = cat
            if test_vocab:
                accuracies.append(suite.evaluate_classification(test_vocab)["accuracy"])
            else:
                accuracies.append(0.0)

        converged_at = next(
            (passes_list[i] for i, a in enumerate(accuracies) if a >= 0.80), None)
        return {"passes": passes_list, "accuracies": accuracies, "converged_at": converged_at}

    def _run_single_seed(
        self, vocab: Dict[str, GroundingContext],
        config: NEMOBenchmarkConfig, seed: int,
    ) -> Dict[str, Any]:
        """Run full benchmark for one seed and return metrics dict."""
        self.log(f"  Seed {seed}: creating parser (n={config.n}, k={config.k})")
        t0 = time.perf_counter()

        parser = EmergentParser(
            n=config.n, k=config.k, p=config.p,
            beta=config.beta, seed=seed, rounds=config.rounds,
            vocabulary=vocab,
        )
        sentences = generate_training_sentences(vocab, n_sentences=100, seed=seed)
        parser.train(sentences=sentences)
        train_time = time.perf_counter() - t0

        suite = EvaluationSuite(parser)
        ev = suite.full_evaluation()

        cls = ev.get("classification", {})
        gen = ev.get("generation", {})
        wo = ev.get("word_order", {})

        # Role evaluation on transitive sentences
        test_sents = []
        for sent in sentences:
            rd = {}
            for word, role in zip(sent.words, sent.roles):
                if role is not None:
                    rd[word] = role.upper()
            if rd:
                test_sents.append({"words": sent.words, "expected_roles": rd})
        role_result = suite.evaluate_roles(test_sents[:20])

        # Assembly stability on 5 sample words (one per core area)
        stab_words = []
        for core_area, lex in parser.core_lexicons.items():
            if lex:
                stab_words.append(next(iter(lex)))
            if len(stab_words) >= 5:
                break
        stab = self._measure_assembly_stability(parser, stab_words, config)

        self.log(
            f"  Seed {seed}: cls={cls.get('accuracy', 0):.1%}, "
            f"roles={role_result.get('accuracy', 0):.1%}, "
            f"stab={stab['mean']:.3f}, rt={gen.get('roundtrip_accuracy', 0):.1%} "
            f"({train_time:.1f}s)")

        return {
            "seed": seed, "train_time": train_time,
            "classification_accuracy": cls.get("accuracy", 0.0),
            "per_category": cls.get("per_category", {}),
            "role_accuracy": role_result.get("accuracy", 0.0),
            "role_per_role": role_result.get("per_role", {}),
            "word_order_correct": wo.get("correct", False),
            "word_order_confidence": wo.get("confidence", 0.0),
            "roundtrip_accuracy": gen.get("roundtrip_accuracy", 0.0),
            "content_recall": gen.get("content_recall", 0.0),
            "stability_mean": stab["mean"], "stability_min": stab["min"],
            "stability_per_word": stab["per_word"],
        }

    def run(self, config: Optional[NEMOBenchmarkConfig] = None, **kwargs) -> ExperimentResult:
        """Run the full NEMO benchmark across multiple seeds."""
        if config is None:
            config = NEMOBenchmarkConfig()
        self._start_timer()

        self.log("=" * 60)
        self.log("NEMO BENCHMARK: EmergentParser vs Reference NEMO")
        self.log(f"  n={config.n} k={config.k} p={config.p} beta={config.beta} "
                 f"rounds={config.rounds} seeds={config.n_seeds}")
        self.log("=" * 60)

        vocab = build_vocabulary()
        self.log(f"Vocabulary: {len(vocab)} words\n")

        # Phase 1: per-seed evaluation
        seeds = [self.seed + i for i in range(config.n_seeds)]
        seed_results = [self._run_single_seed(vocab, config, s) for s in seeds]

        # Phase 2: convergence analysis (first seed)
        self.log(f"\nConvergence analysis (seed {seeds[0]}):")
        convergence = self._measure_convergence(vocab, config, seeds[0])
        for ps, acc in zip(convergence["passes"], convergence["accuracies"]):
            self.log(f"  {ps} pass(es): {acc:.1%}")

        # Aggregate across seeds
        cls_accs = [r["classification_accuracy"] for r in seed_results]
        role_accs = [r["role_accuracy"] for r in seed_results]
        stab_means = [r["stability_mean"] for r in seed_results]
        rt_accs = [r["roundtrip_accuracy"] for r in seed_results]

        # Per-category aggregation
        all_cats = set()
        for r in seed_results:
            all_cats.update(r["per_category"].keys())
        cat_agg: Dict[str, Dict[str, Any]] = {}
        for cat in sorted(all_cats):
            cat_agg[cat] = {
                m: summarize([r["per_category"].get(cat, {}).get(m, 0.0)
                              for r in seed_results])
                for m in ("precision", "recall", "f1")
            }

        # Hypothesis tests
        h1 = ttest_vs_null(cls_accs, 0.80)
        h2 = ttest_vs_null(role_accs, 0.70)
        h3 = ttest_vs_null(stab_means, 0.90)
        h5 = ttest_vs_null(rt_accs, 0.50)
        h4_pass = convergence["converged_at"] is not None

        cls_s, role_s = summarize(cls_accs), summarize(role_accs)
        stab_s, rt_s = summarize(stab_means), summarize(rt_accs)
        duration = self._stop_timer()

        # Print summary
        self.log("\n" + "=" * 60)
        self.log("RESULTS SUMMARY")
        self.log("=" * 60)
        for label, s, test, tgt in [
            ("H1 Classification (>=80%)", cls_s, h1, "80%"),
            ("H2 Role accuracy (>=70%)", role_s, h2, "70%"),
            ("H3 Stability (>=0.90)", stab_s, h3, "0.90"),
            ("H5 Roundtrip (>=50%)", rt_s, h5, "50%"),
        ]:
            self.log(f"\n{label}:")
            self.log(f"  {s['mean']:.3f} [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}] "
                     f"t={test['t']:.2f} p={test['p']:.4f} d={test['d']:.2f}")
        self.log(f"\nH4 Convergence within 5 passes: {h4_pass} "
                 f"(at pass {convergence['converged_at']})")
        self.log(f"\nPer-category F1:")
        for cat, agg in cat_agg.items():
            self.log(f"  {cat:8s}: P={agg['precision']['mean']:.2f} "
                     f"R={agg['recall']['mean']:.2f} F1={agg['f1']['mean']:.2f}")
        self.log(f"\nDuration: {duration:.1f}s")

        # Build ExperimentResult
        hypotheses = {
            "H1_classification_ge_80pct": {"passed": cls_s["ci95_lo"] >= 0.80, **h1},
            "H2_role_ge_70pct": {"passed": role_s["ci95_lo"] >= 0.70, **h2},
            "H3_stability_ge_090": {"passed": stab_s["ci95_lo"] >= 0.90, **h3},
            "H4_convergence_within_5": {
                "passed": h4_pass, "converged_at": convergence["converged_at"]},
            "H5_roundtrip_ge_50pct": {"passed": rt_s["ci95_lo"] >= 0.50, **h5},
        }
        metrics = {
            "classification_accuracy": cls_s, "role_accuracy": role_s,
            "assembly_stability": stab_s, "roundtrip_accuracy": rt_s,
            "per_category_f1": {c: a["f1"] for c, a in cat_agg.items()},
            "convergence": convergence, "hypotheses": hypotheses,
            "mean_train_time_seconds": float(np.mean([r["train_time"] for r in seed_results])),
            "vocab_size": len(vocab),
        }
        raw_data = {
            "seed_results": seed_results,
            "convergence_trace": convergence,
            "per_category_aggregate": cat_agg,
        }
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": config.n, "k": config.k, "p": config.p,
                "beta": config.beta, "rounds": config.rounds,
                "n_seeds": config.n_seeds,
                "stability_test_rounds": config.stability_test_rounds,
                "base_seed": self.seed,
            },
            metrics=metrics, raw_data=raw_data, duration_seconds=duration,
        )
        self.save_result(result)
        return result


def main():
    """Run the NEMO benchmark from the command line."""
    ap = argparse.ArgumentParser(
        description="NEMO Benchmark: EmergentParser vs reference NEMO")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 2 seeds instead of 5")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--quiet", action="store_true", help="Suppress output")
    args = ap.parse_args()

    config = NEMOBenchmarkConfig()
    if args.quick:
        config.n_seeds = 2

    experiment = NEMOBenchmarkExperiment(seed=args.seed, verbose=not args.quiet)
    result = experiment.run(config=config)

    hyp = result.metrics.get("hypotheses", {})
    n_passed = sum(1 for h in hyp.values() if h.get("passed", False))
    print(f"\nHypotheses passed: {n_passed}/{len(hyp)}")
    for name, h in hyp.items():
        status = "PASS" if h.get("passed") else "FAIL"
        print(f"  [{status}] {name}")


if __name__ == "__main__":
    main()
