"""
Curriculum Ablation: Does Training Order Matter?

Inspired by Schulz et al. (2025), which shows that subgrammar pretraining
helps smaller transformer models learn more structured representations.

Tests whether the staged curriculum (SVO -> SVO+PP -> SRC -> ORC) produces
better final performance than alternative training orders:

  A. Standard:   SVO -> SVO+PP -> SRC -> ORC  (child-like development)
  B. Reversed:   ORC -> SRC -> SVO+PP -> SVO
  C. All-at-once: Full grammar from start (no staging)
  D. Skipped PP: SVO -> SRC -> ORC  (skip PP stage entirely)

All conditions receive the same total number of training sentences, just
in different orders. Final measurement uses the same 7-phenomenon battery
as the developmental curriculum experiment.

Hypotheses:
  H1: Standard >= all-at-once for prediction N400 (curriculum helps)
  H2: Reversed shows lower final performance overall
  H3: Skipped-PP impairs PP P600 but not SRC/ORC phenomena

Usage:
    uv run python research/experiments/primitives/test_curriculum_ablation.py
    uv run python research/experiments/primitives/test_curriculum_ablation.py --quick
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
from research.experiments.lib.vocabulary import RECURSIVE_VOCAB
from research.experiments.lib.grammar import RecursiveCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400, measure_p600


@dataclass
class CurriculumAblationConfig:
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
    sentences_per_stage: int = 50
    training_reps: int = 3
    n_test_items: int = 5


# Grammar parameters for each stage type
STAGE_GRAMMARS = {
    "SVO": dict(pp_prob=0.0, rel_prob=0.0, orc_prob=0.0, max_pp_depth=1),
    "SVO_PP": dict(pp_prob=0.6, rel_prob=0.0, orc_prob=0.0, max_pp_depth=1),
    "SRC": dict(pp_prob=0.3, rel_prob=0.5, orc_prob=0.0, max_pp_depth=1),
    "ORC": dict(pp_prob=0.3, rel_prob=0.5, orc_prob=0.5, max_pp_depth=1),
    "FULL": dict(pp_prob=0.3, rel_prob=0.3, orc_prob=0.3, max_pp_depth=1),
}

# Four curriculum conditions
CONDITIONS = {
    "standard": ["SVO", "SVO_PP", "SRC", "ORC"],
    "reversed": ["ORC", "SRC", "SVO_PP", "SVO"],
    "all_at_once": ["FULL", "FULL", "FULL", "FULL"],
    "skipped_pp": ["SVO", "SVO", "SRC", "ORC"],  # extra SVO replaces SVO+PP
}


def measure_battery(brain, lexicon, vocab, cfg) -> Dict[str, Any]:
    """Measure all 7 phenomena. Returns raw paired values for effect size.

    Same battery as test_developmental_curriculum.py, duplicated here for
    self-containment (each experiment is independent).
    """
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    locs = vocab.words_for_category("LOCATION")
    ni = cfg.n_test_items

    results = {}

    # 1. Assembly stability
    stability_scores = []
    for noun in nouns[:ni]:
        activate_word(brain, noun, "NOUN_CORE", 3)
        a1 = np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
        activate_word(brain, noun, "NOUN_CORE", 3)
        a2 = np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
        stability_scores.append(measure_overlap(a1, a2))
    results["assembly_stability"] = float(np.mean(stability_scores))

    # 2. Forward prediction N400 at object position
    n400_gram, n400_cv = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        n400_gram.append(measure_n400(predicted, lexicon[gram_obj]))
        n400_cv.append(measure_n400(predicted, lexicon[cv_obj]))

    results["prediction_n400_gram"] = n400_gram
    results["prediction_n400_cv"] = n400_cv

    # 3. Binding P600 at object position
    p600_gram, p600_cv = [], []
    for i in range(ni):
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]
        p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    results["binding_p600_gram"] = p600_gram
    results["binding_p600_cv"] = p600_cv

    # 4. PP binding P600
    pp_p600_gram, pp_p600_cv = [], []
    for i in range(ni):
        gram_pp = locs[i % len(locs)]
        cv_pp = verbs[(i + 2) % len(verbs)]
        pp_p600_gram.append(measure_p600(
            brain, gram_pp, "NOUN_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))
        pp_p600_cv.append(measure_p600(
            brain, cv_pp, "VERB_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))

    results["pp_p600_gram"] = pp_p600_gram
    results["pp_p600_cv"] = pp_p600_cv

    # 5. SRC dual binding P600
    src_dual = []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        src_dual.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))
    results["src_dual_binding"] = src_dual

    # 6. Garden-path N400
    gp_n400, unamb_n400 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # Unambiguous (with "that")
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_u = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        unamb_n400.append(measure_n400(pred_u, lexicon[main_verb]))

        # Garden-path (no "that")
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_g = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        gp_n400.append(measure_n400(pred_g, lexicon[main_verb]))

    results["garden_path_gp"] = gp_n400
    results["garden_path_unamb"] = unamb_n400

    # 7. SRC/ORC asymmetry
    src_p600, orc_p600 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        src_p600.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))
        orc_p600.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_PATIENT", cfg.n_settling_rounds))

    results["src_orc_src"] = src_p600
    results["src_orc_orc"] = orc_p600

    return results


def run_trial(
    cfg: CurriculumAblationConfig,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Run one ablation trial: four conditions, each with separate brain."""
    vocab = RECURSIVE_VOCAB
    n_per_stage = cfg.sentences_per_stage * cfg.training_reps

    condition_results = {}

    for cond_name, stage_list in CONDITIONS.items():
        rng = np.random.default_rng(seed)
        brain = create_language_brain(
            BrainConfig(n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds),
            vocab, seed)

        # Train through stages
        for stage_name in stage_list:
            params = STAGE_GRAMMARS[stage_name]
            grammar = RecursiveCFG(vocab=vocab, rng=rng, **params)
            sents = grammar.generate_batch(n_per_stage)
            for s in sents:
                train_sentence(brain, s, vocab,
                               cfg.train_rounds_per_pair, cfg.binding_rounds)

        # Measure
        brain.disable_plasticity = True
        lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
        battery = measure_battery(brain, lexicon, vocab, cfg)
        condition_results[cond_name] = battery

    return condition_results


def compute_effects(battery: Dict[str, Any]) -> Dict[str, float]:
    """Compute Cohen's d for each phenomenon from a battery result."""
    effects = {}

    # Prediction N400
    if battery["prediction_n400_gram"] and battery["prediction_n400_cv"]:
        t = paired_ttest(battery["prediction_n400_gram"],
                         battery["prediction_n400_cv"])
        effects["prediction_n400_d"] = t["d"]

    # Binding P600
    if battery["binding_p600_gram"] and battery["binding_p600_cv"]:
        t = paired_ttest(battery["binding_p600_gram"],
                         battery["binding_p600_cv"])
        effects["binding_p600_d"] = t["d"]

    # PP P600
    if battery["pp_p600_gram"] and battery["pp_p600_cv"]:
        t = paired_ttest(battery["pp_p600_gram"],
                         battery["pp_p600_cv"])
        effects["pp_p600_d"] = t["d"]

    # SRC dual binding (mean value, not d)
    effects["src_dual_binding_mean"] = float(np.mean(battery["src_dual_binding"]))

    # Garden-path
    if battery["garden_path_gp"] and battery["garden_path_unamb"]:
        t = paired_ttest(battery["garden_path_gp"],
                         battery["garden_path_unamb"])
        effects["garden_path_d"] = t["d"]

    # SRC/ORC
    if battery["src_orc_src"] and battery["src_orc_orc"]:
        t = paired_ttest(battery["src_orc_src"],
                         battery["src_orc_orc"])
        effects["src_orc_d"] = t["d"]

    return effects


class CurriculumAblationExperiment(ExperimentBase):
    """Curriculum ordering ablation experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="curriculum_ablation",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[CurriculumAblationConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or CurriculumAblationConfig(
            **{k: v for k, v in kwargs.items()
               if k in CurriculumAblationConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Curriculum Ablation: Does Training Order Matter?")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  sentences_per_stage={cfg.sentences_per_stage}, "
                 f"reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-seed effect sizes for each condition
        cond_effects: Dict[str, Dict[str, List[float]]] = {
            cond: {} for cond in CONDITIONS}

        effect_keys = ["prediction_n400_d", "binding_p600_d", "pp_p600_d",
                       "src_dual_binding_mean", "garden_path_d", "src_orc_d"]

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            for cond_name, battery in trial.items():
                effects = compute_effects(battery)
                for k in effect_keys:
                    cond_effects[cond_name].setdefault(k, []).append(
                        effects.get(k, 0.0))

        # Report
        self.log(f"\n  {'Condition':<16s} | {'N400':>7s} | {'P600':>7s} | "
                 f"{'PP':>7s} | {'GP':>7s} | {'SRC/ORC':>7s}")
        self.log("  " + "-" * 65)

        for cond_name in CONDITIONS:
            ce = cond_effects[cond_name]
            self.log(
                f"  {cond_name:<16s} | "
                f"{np.mean(ce.get('prediction_n400_d', [0])):7.2f} | "
                f"{np.mean(ce.get('binding_p600_d', [0])):7.2f} | "
                f"{np.mean(ce.get('pp_p600_d', [0])):7.2f} | "
                f"{np.mean(ce.get('garden_path_d', [0])):7.2f} | "
                f"{np.mean(ce.get('src_orc_d', [0])):7.2f}"
            )

        # Hypotheses
        std_n400 = np.mean(cond_effects["standard"].get("prediction_n400_d", [0]))
        aao_n400 = np.mean(cond_effects["all_at_once"].get("prediction_n400_d", [0]))
        h1 = std_n400 >= aao_n400

        # H2: Reversed lower overall
        std_overall = np.mean([
            np.mean(cond_effects["standard"].get(k, [0]))
            for k in effect_keys if k != "src_dual_binding_mean"])
        rev_overall = np.mean([
            np.mean(cond_effects["reversed"].get(k, [0]))
            for k in effect_keys if k != "src_dual_binding_mean"])
        h2 = rev_overall < std_overall

        # H3: Skipped PP impairs PP but not SRC/ORC
        std_pp = np.mean(cond_effects["standard"].get("pp_p600_d", [0]))
        skip_pp = np.mean(cond_effects["skipped_pp"].get("pp_p600_d", [0]))
        std_src = np.mean(cond_effects["standard"].get("src_orc_d", [0]))
        skip_src = np.mean(cond_effects["skipped_pp"].get("src_orc_d", [0]))
        pp_impaired = skip_pp < std_pp
        src_preserved = abs(skip_src - std_src) < abs(std_src) * 0.5 + 0.5
        h3 = pp_impaired

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Standard >= all-at-once N400): "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" ({std_n400:.2f} vs {aao_n400:.2f})")
        self.log(f"    H2 (Reversed < standard overall):  "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" ({rev_overall:.2f} vs {std_overall:.2f})")
        self.log(f"    H3 (Skipped PP impairs PP P600):   "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (PP: {skip_pp:.2f} vs {std_pp:.2f},"
                 f" SRC/ORC: {skip_src:.2f} vs {std_src:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "condition_effects": {
                cond: {k: summarize(v) for k, v in ce.items()}
                for cond, ce in cond_effects.items()
            },
            "hypotheses": {
                "H1_standard_ge_all_at_once": h1,
                "H2_reversed_lower": h2,
                "H3_skipped_pp_impairs": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "sentences_per_stage": cfg.sentences_per_stage,
                "training_reps": cfg.training_reps,
                "n_test_items": cfg.n_test_items,
                "n_seeds": n_seeds,
                "conditions": {k: v for k, v in CONDITIONS.items()},
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Ablation Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = CurriculumAblationExperiment(verbose=True)

    if args.quick:
        cfg = CurriculumAblationConfig(
            n=5000, k=50,
            sentences_per_stage=30, training_reps=2,
            n_test_items=4)
        n_seeds = args.seeds or 3
    else:
        cfg = CurriculumAblationConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("CURRICULUM ABLATION SUMMARY")
    print("=" * 70)
    print(f"\nH1 Standard >= all-at-once: {'PASS' if h['H1_standard_ge_all_at_once'] else 'FAIL'}")
    print(f"H2 Reversed lower:         {'PASS' if h['H2_reversed_lower'] else 'FAIL'}")
    print(f"H3 Skipped PP impairs:     {'PASS' if h['H3_skipped_pp_impairs'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
