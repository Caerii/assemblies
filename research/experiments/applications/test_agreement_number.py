"""
Agreement Violations with NUMBER Area -- P600 via Number-Aware Consolidation

Tests whether adding a dedicated NUMBER brain area and co-projecting number
features during consolidation produces the predicted graded P600 effect:

    P600(category_violation) > P600(agreement_violation) > P600(grammatical)

Background:
    The original test_agreement_violations.py produced a NEGATIVE result
    because SG/PL grounding features do not create different consolidation
    patterns in structural areas. Both SG and PL nouns receive identical
    NOUN_CORE->ROLE_AGENT consolidation.

    This experiment adds a NUMBER area (trained during parser.train() via
    train_number()) and uses number-aware consolidation functions that
    co-project NUMBER into structural areas. This creates:

    - SG_noun + NUMBER(SG) -> ROLE_AGENT: SG-flavored Hebbian pattern
    - PL_noun + NUMBER(PL) -> ROLE_AGENT: PL-flavored Hebbian pattern
    - SG_noun + SG_verb + NUMBER(SG) -> VP: SG-agreement VP pattern
    - PL_noun + PL_verb + NUMBER(PL) -> VP: PL-agreement VP pattern

    At test time, mismatched number (PL subject + SG verb) produces a
    NUMBER signal that conflicts with all consolidated VP patterns,
    causing instability.

Measurement:
    A custom measure_agreement_word() function extends measure_critical_word()
    to co-project NUMBER stimuli during context processing, so the NUMBER
    area is active with the subject's number when the critical word arrives.

    Experiments:
      A: Verb-position -- measure at the verb where agreement mismatch occurs
      B: Object-position -- measure at the object for comparison with original
      C: Category violation baseline -- verb in noun slot (no NUMBER involvement)

    Predictions:
      1. Verb position: P600(agree) > P600(gram) -- number mismatch instability
      2. Object position: P600(cat_viol) > P600(agree) > P600(gram) -- graded
      3. VP and ROLE areas show the biggest agreement effects
      4. NUMBER area shows elevated instability for mismatched conditions

Literature:
  - Hagoort et al. 1993: P600 for agreement violations
  - Kaan et al. 2000: Graded P600 effects
  - Osterhout & Mobley 1995: P600 amplitude scales with violation severity
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.infrastructure import setup_number_p600_pipeline
from research.experiments.metrics.measurement import measure_agreement_word
from research.experiments.metrics.integration_cost import compute_vp_distance
from research.experiments.vocab.agreement import (
    build_agreement_vocab, build_agreement_training,
)
from research.experiments.vocab.test_sentences import (
    VERB_AGREEMENT_TESTS as VERB_TESTS,
    OBJECT_AGREEMENT_TESTS as OBJECT_TESTS,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import (
    NOUN_CORE, VERB_CORE, ROLE_AGENT, ROLE_PATIENT, VP, NUMBER,
)


@dataclass
class AgreementNumberConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 10
    consolidation_passes: int = 1



class AgreementNumberExperiment(ExperimentBase):
    """Test agreement P600 with NUMBER-aware consolidation."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="agreement_number",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = AgreementNumberConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = build_agreement_vocab()
        training = build_agreement_training(vocab)
        seeds = list(range(cfg.n_seeds))

        # Only measure areas where NUMBER was consolidated.
        # SUBJ/OBJ excluded: no NUMBER consolidation, dominated by context contamination.
        p600_areas = [ROLE_AGENT, ROLE_PATIENT, VP]
        # Source areas for bootstrap include NUMBER
        source_areas = [NOUN_CORE, VERB_CORE, NUMBER]

        # -- Accumulators --
        # Exp A: Verb position
        verb_gram_seeds = []
        verb_agree_seeds = []
        verb_gram_per_area = {a: [] for a in p600_areas}
        verb_agree_per_area = {a: [] for a in p600_areas}

        # Exp B: Object position
        obj_gram_seeds = []
        obj_agree_seeds = []
        obj_cat_seeds = []
        obj_gram_per_area = {a: [] for a in p600_areas}
        obj_agree_per_area = {a: [] for a in p600_areas}
        obj_cat_per_area = {a: [] for a in p600_areas}

        # N400 and core instability
        n400_verb_gram_seeds = []
        n400_verb_agree_seeds = []
        n400_obj_gram_seeds = []
        n400_obj_agree_seeds = []

        # VP competition margin (agreement-specific P600 metric)
        margin_verb_gram_seeds = []
        margin_verb_agree_seeds = []
        margin_obj_gram_seeds = []
        margin_obj_agree_seeds = []
        margin_obj_cat_seeds = []

        # VP overlap (paired metric: Jaccard distance between conditions)
        vp_overlap_verb_seeds = []  # Jaccard(VP_gram, VP_agree) at verb
        vp_overlap_obj_agree_seeds = []  # at object position
        vp_overlap_obj_cat_seeds = []

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            setup_number_p600_pipeline(
                parser, training, p600_areas,
                source_areas=source_areas,
                number_consolidation_passes=cfg.consolidation_passes,
                log_fn=self.log,
            )

            # --- Exp A: Verb-position measurement ---
            self.log("  Exp A: Verb-position P600...")
            verb_gram_vals, verb_agree_vals = [], []
            verb_n400_gram, verb_n400_agree = [], []
            verb_margin_gram, verb_margin_agree = [], []
            verb_vp_overlaps = []

            for test in VERB_TESTS:
                # Grammatical: SG context + SG verb
                result_gram = measure_agreement_word(
                    parser, test["sg_context"], test["verb"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                verb_gram_vals.append(result_gram["p600_mean_instability"])
                verb_n400_gram.append(result_gram["n400_energy"])
                verb_margin_gram.append(result_gram["vp_margin"])

                for a in p600_areas:
                    verb_gram_per_area[a].append(
                        result_gram["p600_instability"].get(a, 0.0))

                # Agreement violation: PL context + SG verb
                result_agree = measure_agreement_word(
                    parser, test["pl_context"], test["verb"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                verb_agree_vals.append(result_agree["p600_mean_instability"])
                verb_n400_agree.append(result_agree["n400_energy"])
                verb_margin_agree.append(result_agree["vp_margin"])

                for a in p600_areas:
                    verb_agree_per_area[a].append(
                        result_agree["p600_instability"].get(a, 0.0))

                # VP overlap: Jaccard distance between gram and agree VP
                verb_vp_overlaps.append(compute_vp_distance(
                    result_gram["vp_winners"], result_agree["vp_winners"]))

            verb_gram_seeds.append(float(np.mean(verb_gram_vals)))
            verb_agree_seeds.append(float(np.mean(verb_agree_vals)))
            n400_verb_gram_seeds.append(float(np.mean(verb_n400_gram)))
            n400_verb_agree_seeds.append(float(np.mean(verb_n400_agree)))
            margin_verb_gram_seeds.append(float(np.mean(verb_margin_gram)))
            margin_verb_agree_seeds.append(float(np.mean(verb_margin_agree)))
            vp_overlap_verb_seeds.append(float(np.mean(verb_vp_overlaps)))

            self.log(f"    inst: gram={verb_gram_seeds[-1]:.4f}  "
                     f"agree={verb_agree_seeds[-1]:.4f}  "
                     f"margin: gram={margin_verb_gram_seeds[-1]:.2f}  "
                     f"agree={margin_verb_agree_seeds[-1]:.2f}  "
                     f"VP_dist={vp_overlap_verb_seeds[-1]:.3f}")

            # --- Exp B: Object-position measurement ---
            self.log("  Exp B: Object-position P600...")
            obj_gram_vals, obj_agree_vals, obj_cat_vals = [], [], []
            obj_n400_gram, obj_n400_agree = [], []
            obj_margin_gram, obj_margin_agree, obj_margin_cat = [], [], []
            obj_vp_agree_overlaps, obj_vp_cat_overlaps = [], []

            for test in OBJECT_TESTS:
                # Grammatical: SG context + SG object
                result_gram = measure_agreement_word(
                    parser, test["sg_context"], test["grammatical_obj"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                obj_gram_vals.append(result_gram["p600_mean_instability"])
                obj_n400_gram.append(result_gram["n400_energy"])
                obj_margin_gram.append(result_gram["vp_margin"])

                for a in p600_areas:
                    obj_gram_per_area[a].append(
                        result_gram["p600_instability"].get(a, 0.0))

                # Agreement violation: PL context + SG object
                result_agree = measure_agreement_word(
                    parser, test["pl_context"], test["grammatical_obj"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                obj_agree_vals.append(result_agree["p600_mean_instability"])
                obj_n400_agree.append(result_agree["n400_energy"])
                obj_margin_agree.append(result_agree["vp_margin"])

                for a in p600_areas:
                    obj_agree_per_area[a].append(
                        result_agree["p600_instability"].get(a, 0.0))

                # Category violation: SG context + verb in noun slot
                result_cat = measure_agreement_word(
                    parser, test["sg_context"], test["category_violation"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                obj_cat_vals.append(result_cat["p600_mean_instability"])
                obj_margin_cat.append(result_cat["vp_margin"])

                for a in p600_areas:
                    obj_cat_per_area[a].append(
                        result_cat["p600_instability"].get(a, 0.0))

                # VP overlap: Jaccard distance from grammatical reference
                vp_g = result_gram["vp_winners"]
                obj_vp_agree_overlaps.append(compute_vp_distance(
                    vp_g, result_agree["vp_winners"]))
                obj_vp_cat_overlaps.append(compute_vp_distance(
                    vp_g, result_cat["vp_winners"]))

            obj_gram_seeds.append(float(np.mean(obj_gram_vals)))
            obj_agree_seeds.append(float(np.mean(obj_agree_vals)))
            obj_cat_seeds.append(float(np.mean(obj_cat_vals)))
            n400_obj_gram_seeds.append(float(np.mean(obj_n400_gram)))
            n400_obj_agree_seeds.append(float(np.mean(obj_n400_agree)))
            margin_obj_gram_seeds.append(float(np.mean(obj_margin_gram)))
            margin_obj_agree_seeds.append(float(np.mean(obj_margin_agree)))
            margin_obj_cat_seeds.append(float(np.mean(obj_margin_cat)))
            vp_overlap_obj_agree_seeds.append(
                float(np.mean(obj_vp_agree_overlaps)))
            vp_overlap_obj_cat_seeds.append(
                float(np.mean(obj_vp_cat_overlaps)))

            self.log(f"    inst: gram={obj_gram_seeds[-1]:.4f}  "
                     f"agree={obj_agree_seeds[-1]:.4f}  "
                     f"cat={obj_cat_seeds[-1]:.4f}")
            self.log(f"    VP_dist: agree={vp_overlap_obj_agree_seeds[-1]:.3f}  "
                     f"cat={vp_overlap_obj_cat_seeds[-1]:.3f}")

        # ================================================================
        # Analysis
        # ================================================================
        self.log(f"\n{'='*60}")
        self.log("AGREEMENT NUMBER RESULTS")
        self.log(f"{'='*60}")

        metrics = {}

        # -- Exp A: Verb position --
        self.log("\n--- Exp A: Verb-position P600 ---")
        metrics["verb_gram"] = summarize(verb_gram_seeds)
        metrics["verb_agree"] = summarize(verb_agree_seeds)

        if len(verb_gram_seeds) >= 2:
            verb_test = paired_ttest(verb_agree_seeds, verb_gram_seeds)
            metrics["verb_agree_vs_gram"] = {
                "test": verb_test,
                "agree_higher": str(np.mean(verb_agree_seeds) >
                                    np.mean(verb_gram_seeds)),
            }
            self.log(f"  gram={np.mean(verb_gram_seeds):.4f}  "
                     f"agree={np.mean(verb_agree_seeds):.4f}")
            self.log(f"  agree vs gram: d={verb_test['d']:.3f}  "
                     f"p={verb_test['p']:.4f}  "
                     f"agree>gram={metrics['verb_agree_vs_gram']['agree_higher']}")

        # Per-area verb position
        self.log("\n  Per-area verb position:")
        for a in p600_areas:
            if len(verb_gram_per_area[a]) >= 2:
                # Average per seed (4 tests per seed)
                n_tests = len(VERB_TESTS)
                gram_by_seed = [
                    float(np.mean(verb_gram_per_area[a][i*n_tests:(i+1)*n_tests]))
                    for i in range(len(seeds))
                ]
                agree_by_seed = [
                    float(np.mean(verb_agree_per_area[a][i*n_tests:(i+1)*n_tests]))
                    for i in range(len(seeds))
                ]
                if min(len(gram_by_seed), len(agree_by_seed)) >= 2:
                    t = paired_ttest(agree_by_seed, gram_by_seed)
                    direction = "agree>gram" if np.mean(agree_by_seed) > np.mean(gram_by_seed) else "gram>agree"
                    self.log(f"    {a}: agree={np.mean(agree_by_seed):.4f}  "
                             f"gram={np.mean(gram_by_seed):.4f}  "
                             f"d={t['d']:.2f}  p={t['p']:.3f}  {direction}")
                    metrics[f"verb_per_area_{a}"] = {
                        "agree": float(np.mean(agree_by_seed)),
                        "gram": float(np.mean(gram_by_seed)),
                        "test": t,
                        "direction": direction,
                    }

        # -- Exp B: Object position --
        self.log("\n--- Exp B: Object-position P600 ---")
        metrics["obj_gram"] = summarize(obj_gram_seeds)
        metrics["obj_agree"] = summarize(obj_agree_seeds)
        metrics["obj_cat"] = summarize(obj_cat_seeds)

        if len(obj_gram_seeds) >= 2:
            obj_agree_test = paired_ttest(obj_agree_seeds, obj_gram_seeds)
            obj_cat_test = paired_ttest(obj_cat_seeds, obj_gram_seeds)
            obj_cat_agree_test = paired_ttest(obj_cat_seeds, obj_agree_seeds)

            metrics["obj_agree_vs_gram"] = {
                "test": obj_agree_test,
                "direction": "HIGHER" if np.mean(obj_agree_seeds) > np.mean(obj_gram_seeds) else "LOWER",
            }
            metrics["obj_cat_vs_gram"] = {
                "test": obj_cat_test,
                "direction": "HIGHER" if np.mean(obj_cat_seeds) > np.mean(obj_gram_seeds) else "LOWER",
            }
            metrics["obj_cat_vs_agree"] = {
                "test": obj_cat_agree_test,
                "direction": "HIGHER" if np.mean(obj_cat_seeds) > np.mean(obj_agree_seeds) else "LOWER",
            }

            # Check predicted ordering: cat > agree > gram
            ordering_correct = (
                np.mean(obj_cat_seeds) > np.mean(obj_agree_seeds) >
                np.mean(obj_gram_seeds)
            )
            metrics["ordering_correct"] = bool(ordering_correct)

            self.log(f"  gram={np.mean(obj_gram_seeds):.4f}  "
                     f"agree={np.mean(obj_agree_seeds):.4f}  "
                     f"cat={np.mean(obj_cat_seeds):.4f}")
            self.log(f"  Predicted ordering cat>agree>gram: "
                     f"{'YES' if ordering_correct else 'NO'}")
            self.log(f"  agree vs gram: d={obj_agree_test['d']:.3f}  "
                     f"p={obj_agree_test['p']:.4f}  "
                     f"{metrics['obj_agree_vs_gram']['direction']}")
            self.log(f"  cat vs gram:   d={obj_cat_test['d']:.3f}  "
                     f"p={obj_cat_test['p']:.4f}  "
                     f"{metrics['obj_cat_vs_gram']['direction']}")
            self.log(f"  cat vs agree:  d={obj_cat_agree_test['d']:.3f}  "
                     f"p={obj_cat_agree_test['p']:.4f}  "
                     f"{metrics['obj_cat_vs_agree']['direction']}")

        # Per-area object position
        self.log("\n  Per-area object position:")
        for a in p600_areas:
            if len(obj_gram_per_area[a]) >= 2:
                n_tests = len(OBJECT_TESTS)
                gram_by_seed = [
                    float(np.mean(obj_gram_per_area[a][i*n_tests:(i+1)*n_tests]))
                    for i in range(len(seeds))
                ]
                agree_by_seed = [
                    float(np.mean(obj_agree_per_area[a][i*n_tests:(i+1)*n_tests]))
                    for i in range(len(seeds))
                ]
                if min(len(gram_by_seed), len(agree_by_seed)) >= 2:
                    t = paired_ttest(agree_by_seed, gram_by_seed)
                    direction = "agree>gram" if np.mean(agree_by_seed) > np.mean(gram_by_seed) else "gram>agree"
                    self.log(f"    {a}: agree={np.mean(agree_by_seed):.4f}  "
                             f"gram={np.mean(gram_by_seed):.4f}  "
                             f"d={t['d']:.2f}  p={t['p']:.3f}  {direction}")
                    metrics[f"obj_per_area_{a}"] = {
                        "agree": float(np.mean(agree_by_seed)),
                        "gram": float(np.mean(gram_by_seed)),
                        "test": t,
                        "direction": direction,
                    }

        # -- VP Competition Margin (agreement-specific P600 metric) --
        # Lower margin = more competition between mismatched number patterns
        # = harder unification = larger P600
        # Prediction: margin(gram) > margin(agree) [> margin(cat)]
        self.log("\n--- VP Competition Margin (lower = harder integration) ---")

        if len(margin_verb_gram_seeds) >= 2:
            margin_verb_test = paired_ttest(
                margin_verb_gram_seeds, margin_verb_agree_seeds)
            gram_m = np.mean(margin_verb_gram_seeds)
            agree_m = np.mean(margin_verb_agree_seeds)
            self.log(f"  Verb: gram={gram_m:.2f}  agree={agree_m:.2f}  "
                     f"d={margin_verb_test['d']:.3f}  "
                     f"p={margin_verb_test['p']:.4f}  "
                     f"gram>agree={'YES' if gram_m > agree_m else 'NO'}")
            metrics["margin_verb_gram_vs_agree"] = {
                "test": margin_verb_test,
                "gram_mean": float(gram_m),
                "agree_mean": float(agree_m),
                "gram_higher": bool(gram_m > agree_m),
            }

        if len(margin_obj_gram_seeds) >= 2:
            margin_obj_ag_test = paired_ttest(
                margin_obj_gram_seeds, margin_obj_agree_seeds)
            margin_obj_ct_test = paired_ttest(
                margin_obj_gram_seeds, margin_obj_cat_seeds)
            gram_m = np.mean(margin_obj_gram_seeds)
            agree_m = np.mean(margin_obj_agree_seeds)
            cat_m = np.mean(margin_obj_cat_seeds)
            self.log(f"  Obj:  gram={gram_m:.2f}  agree={agree_m:.2f}  "
                     f"cat={cat_m:.2f}")
            self.log(f"    gram>agree: d={margin_obj_ag_test['d']:.3f}  "
                     f"p={margin_obj_ag_test['p']:.4f}  "
                     f"{'YES' if gram_m > agree_m else 'NO'}")
            self.log(f"    gram>cat:   d={margin_obj_ct_test['d']:.3f}  "
                     f"p={margin_obj_ct_test['p']:.4f}  "
                     f"{'YES' if gram_m > cat_m else 'NO'}")

            margin_ordering = gram_m > agree_m > cat_m
            self.log(f"    Predicted ordering gram>agree>cat: "
                     f"{'YES' if margin_ordering else 'NO'}")
            metrics["margin_obj_ordering"] = bool(margin_ordering)

        # -- VP Assembly Distance (paired Jaccard distance from grammatical) --
        # Prediction: cat_distance > agree_distance > 0
        # Higher distance = VP representation deviates more from grammatical = larger P600
        self.log("\n--- VP Assembly Distance (higher = more deviation from grammatical) ---")

        if len(vp_overlap_verb_seeds) >= 2:
            mean_vd = np.mean(vp_overlap_verb_seeds)
            self.log(f"  Verb agree-vs-gram distance: {mean_vd:.4f} "
                     f"(sd={np.std(vp_overlap_verb_seeds):.4f})")
            metrics["vp_dist_verb"] = summarize(vp_overlap_verb_seeds)

        if (len(vp_overlap_obj_agree_seeds) >= 2
                and len(vp_overlap_obj_cat_seeds) >= 2):
            agree_d = np.mean(vp_overlap_obj_agree_seeds)
            cat_d = np.mean(vp_overlap_obj_cat_seeds)
            dist_test = paired_ttest(
                vp_overlap_obj_cat_seeds, vp_overlap_obj_agree_seeds)
            self.log(f"  Obj agree-vs-gram distance: {agree_d:.4f}")
            self.log(f"  Obj cat-vs-gram distance:   {cat_d:.4f}")
            self.log(f"  cat > agree: d={dist_test['d']:.3f}  "
                     f"p={dist_test['p']:.4f}  "
                     f"{'YES' if cat_d > agree_d else 'NO'}")
            metrics["vp_dist_obj_agree"] = summarize(vp_overlap_obj_agree_seeds)
            metrics["vp_dist_obj_cat"] = summarize(vp_overlap_obj_cat_seeds)
            metrics["vp_dist_cat_vs_agree"] = {
                "test": dist_test,
                "cat_higher": bool(cat_d > agree_d),
            }

        # -- N400 comparison --
        self.log("\n--- N400 Comparison ---")
        if len(n400_verb_gram_seeds) >= 2:
            n400_verb_test = paired_ttest(
                n400_verb_agree_seeds, n400_verb_gram_seeds)
            self.log(f"  Verb N400: gram={np.mean(n400_verb_gram_seeds):.1f}  "
                     f"agree={np.mean(n400_verb_agree_seeds):.1f}  "
                     f"d={n400_verb_test['d']:.2f}")
            metrics["n400_verb_agree_vs_gram"] = {"test": n400_verb_test}

        if len(n400_obj_gram_seeds) >= 2:
            n400_obj_test = paired_ttest(
                n400_obj_agree_seeds, n400_obj_gram_seeds)
            self.log(f"  Obj  N400: gram={np.mean(n400_obj_gram_seeds):.1f}  "
                     f"agree={np.mean(n400_obj_agree_seeds):.1f}  "
                     f"d={n400_obj_test['d']:.2f}")
            metrics["n400_obj_agree_vs_gram"] = {"test": n400_obj_test}

        # -- Summary --
        self.log(f"\n{'='*60}")
        self.log("SUMMARY")
        self.log(f"{'='*60}")

        verb_inst_success = (len(verb_agree_seeds) >= 2 and
                             np.mean(verb_agree_seeds) > np.mean(verb_gram_seeds))
        obj_inst_ordering = (len(obj_gram_seeds) >= 2 and
                             np.mean(obj_cat_seeds) > np.mean(obj_agree_seeds) >
                             np.mean(obj_gram_seeds))
        verb_margin_success = (len(margin_verb_gram_seeds) >= 2 and
                               np.mean(margin_verb_gram_seeds) >
                               np.mean(margin_verb_agree_seeds))

        self.log("  Instability metric:")
        self.log(f"    Verb agree>gram: {'YES' if verb_inst_success else 'NO'}")
        self.log(f"    Obj cat>agree>gram: {'YES' if obj_inst_ordering else 'NO'}")
        self.log("  Competition margin metric:")
        self.log(f"    Verb margin gram>agree: {'YES' if verb_margin_success else 'NO'}")

        any_success = verb_inst_success or verb_margin_success
        both_success = verb_inst_success and verb_margin_success
        self.log(f"  NUMBER area resolves negative result: "
                 f"{'YES' if both_success else 'PARTIALLY' if any_success else 'NO'}")

        metrics["verb_inst_success"] = bool(verb_inst_success)
        metrics["verb_margin_success"] = bool(verb_margin_success)
        metrics["object_ordering_success"] = bool(obj_inst_ordering)

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
        description="Agreement P600 with NUMBER Area")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = AgreementNumberExperiment()
    exp.run(quick=args.quick)
