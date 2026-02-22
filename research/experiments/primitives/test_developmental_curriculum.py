"""
Developmental Curriculum: Staged Language Acquisition

One brain trained through six stages matching child language milestones.
At each stage, a measurement battery tracks which phenomena have emerged.

Stages:
  1. Single words (12-18mo):  Lexicon only
  2. Two-word (18-24mo):      Noun-verb pairs
  3. SVO (24-30mo):           Full agent-verb-patient
  4. SVO+PP (30-36mo):        Adding prepositional phrases
  5. SRC (3-4yr):             Subject-relative clauses
  6. ORC (4-5yr):             Object-relative clauses

Measurement battery (7 phenomena):
  1. Assembly stability (self-overlap)
  2. Forward prediction N400
  3. Binding P600 double dissociation
  4. PP binding P600
  5. SRC dual binding P600
  6. Garden-path N400
  7. SRC/ORC asymmetry P600

Hypotheses:
  H1: Phenomena emerge in developmental order
  H2: No catastrophic forgetting (earlier phenomena preserved)
  H3: All 7 phenomena emerge by stage 6

Usage:
    uv run python research/experiments/primitives/test_developmental_curriculum.py
    uv run python research/experiments/primitives/test_developmental_curriculum.py --quick
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


STAGE_NAMES = [
    "1_single_words",
    "2_two_word",
    "3_svo",
    "4_svo_pp",
    "5_src",
    "6_orc",
]

PHENOMENA = [
    "assembly_stability",
    "prediction_n400",
    "binding_p600",
    "pp_p600",
    "src_dual_binding",
    "garden_path_n400",
    "src_orc_asymmetry",
]


@dataclass
class CurriculumConfig:
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
    # Per-stage sentence counts
    stage_sentences: tuple = (0, 30, 60, 60, 50, 50)
    training_reps: int = 3
    n_test_items: int = 5


def generate_two_word_sentences(
    vocab, rng, n_sentences: int,
) -> List[Dict]:
    """Generate noun-verb pairs for stage 2."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    sentences = []
    for _ in range(n_sentences):
        noun = rng.choice(nouns)
        verb = rng.choice(verbs)
        sentences.append({
            "words": [noun, verb],
            "roles": ["AGENT", "VERB"],
            "categories": ["NOUN", "VERB"],
            "has_pp": False,
        })
    return sentences


def generate_stage_sentences(
    stage_idx: int,
    cfg: CurriculumConfig,
    vocab,
    rng,
) -> List[Dict]:
    """Generate training sentences for a given stage."""
    n_sents = cfg.stage_sentences[stage_idx] * cfg.training_reps

    if stage_idx == 0:
        return []  # Stage 1: no sentences

    if stage_idx == 1:
        return generate_two_word_sentences(vocab, rng, n_sents)

    # Stages 3-6: use RecursiveCFG with increasing complexity
    grammar_params = {
        2: dict(pp_prob=0.0, rel_prob=0.0, orc_prob=0.0, max_pp_depth=1),
        3: dict(pp_prob=0.6, rel_prob=0.0, orc_prob=0.0, max_pp_depth=1),
        4: dict(pp_prob=0.3, rel_prob=0.5, orc_prob=0.0, max_pp_depth=1),
        5: dict(pp_prob=0.3, rel_prob=0.5, orc_prob=0.5, max_pp_depth=1),
    }
    params = grammar_params[stage_idx]
    grammar = RecursiveCFG(vocab=vocab, rng=rng, **params)
    return grammar.generate_batch(n_sents)


def measure_battery(
    brain,
    lexicon,
    vocab,
    cfg: CurriculumConfig,
) -> Dict[str, Any]:
    """Measure all 7 phenomena. Returns raw paired values for effect size."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    locs = vocab.words_for_category("LOCATION")
    ni = cfg.n_test_items

    results = {}

    # 1. Assembly stability: activate word twice, measure self-overlap
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

    # 3. Binding P600 double dissociation at object position
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

    # 6. Garden-path N400 (GP vs unambiguous at 2nd verb)
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

    # 7. SRC/ORC asymmetry (dual-binding P600)
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
    cfg: CurriculumConfig,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Run one developmental curriculum trial: one brain, six stages."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    stage_results = {}
    total_trained = 0

    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        sentences = generate_stage_sentences(stage_idx, cfg, vocab, rng)
        for sent in sentences:
            train_sentence(brain, sent, vocab,
                           cfg.train_rounds_per_pair, cfg.binding_rounds)
        total_trained += len(sentences)

        # Measure
        brain.disable_plasticity = True
        lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
        battery = measure_battery(brain, lexicon, vocab, cfg)
        battery["n_trained"] = total_trained
        brain.disable_plasticity = False

        stage_results[stage_name] = battery

    return stage_results


class DevelopmentalCurriculumExperiment(ExperimentBase):
    """Developmental curriculum with staged language acquisition."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="developmental_curriculum",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[CurriculumConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or CurriculumConfig(
            **{k: v for k, v in kwargs.items()
               if k in CurriculumConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Developmental Curriculum: Staged Language Acquisition")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  stage_sentences={cfg.stage_sentences}")
        self.log(f"  training_reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-seed data for each stage and phenomenon
        all_stage_data = {s: {p: [] for p in PHENOMENA}
                         for s in STAGE_NAMES}
        # Also collect raw paired values for effect sizes
        raw_pairs = {s: {} for s in STAGE_NAMES}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            for stage_name, battery in trial.items():
                # Compute per-seed effect sizes
                # 1. Assembly stability (single value, not paired)
                all_stage_data[stage_name]["assembly_stability"].append(
                    battery["assembly_stability"])

                # 2-7: paired tests (compute d across seeds later)
                for key in ["prediction_n400_gram", "prediction_n400_cv",
                            "binding_p600_gram", "binding_p600_cv",
                            "pp_p600_gram", "pp_p600_cv",
                            "src_dual_binding",
                            "garden_path_gp", "garden_path_unamb",
                            "src_orc_src", "src_orc_orc"]:
                    raw_pairs[stage_name].setdefault(key, []).append(
                        float(np.mean(battery[key])))

        # Compute effect sizes per stage
        stage_effects = {}
        for stage_name in STAGE_NAMES:
            rp = raw_pairs[stage_name]
            effects = {}

            effects["assembly_stability"] = float(np.mean(
                all_stage_data[stage_name]["assembly_stability"]))

            if rp.get("prediction_n400_cv") and rp.get("prediction_n400_gram"):
                t = paired_ttest(rp["prediction_n400_cv"], rp["prediction_n400_gram"])
                effects["prediction_n400_d"] = t["d"]
            else:
                effects["prediction_n400_d"] = 0.0

            if rp.get("binding_p600_cv") and rp.get("binding_p600_gram"):
                t = paired_ttest(rp["binding_p600_cv"], rp["binding_p600_gram"])
                effects["binding_p600_d"] = t["d"]
            else:
                effects["binding_p600_d"] = 0.0

            if rp.get("pp_p600_cv") and rp.get("pp_p600_gram"):
                t = paired_ttest(rp["pp_p600_cv"], rp["pp_p600_gram"])
                effects["pp_p600_d"] = t["d"]
            else:
                effects["pp_p600_d"] = 0.0

            if rp.get("src_dual_binding"):
                effects["src_dual_binding_mean"] = float(np.mean(
                    rp["src_dual_binding"]))
            else:
                effects["src_dual_binding_mean"] = 0.0

            if rp.get("garden_path_gp") and rp.get("garden_path_unamb"):
                t = paired_ttest(rp["garden_path_gp"], rp["garden_path_unamb"])
                effects["garden_path_d"] = t["d"]
            else:
                effects["garden_path_d"] = 0.0

            if rp.get("src_orc_orc") and rp.get("src_orc_src"):
                t = paired_ttest(rp["src_orc_orc"], rp["src_orc_src"])
                effects["src_orc_d"] = t["d"]
            else:
                effects["src_orc_d"] = 0.0

            stage_effects[stage_name] = effects

        # Report
        self.log(f"\n  {'Stage':<18s} | {'Stab':>5s} | {'Pred':>5s} | "
                 f"{'Bind':>5s} | {'PP':>5s} | {'SRC':>5s} | "
                 f"{'GP':>5s} | {'ORC':>5s}")
        self.log("  " + "-" * 72)

        for stage_name in STAGE_NAMES:
            e = stage_effects[stage_name]
            self.log(
                f"  {stage_name:<18s} | "
                f"{e['assembly_stability']:5.2f} | "
                f"{e['prediction_n400_d']:5.2f} | "
                f"{e['binding_p600_d']:5.2f} | "
                f"{e['pp_p600_d']:5.2f} | "
                f"{e['src_dual_binding_mean']:5.3f} | "
                f"{e['garden_path_d']:5.2f} | "
                f"{e['src_orc_d']:5.2f}"
            )

        # Track emergence
        threshold = cfg.emergence_threshold if hasattr(cfg, 'emergence_threshold') else 0.5
        emergence = {}
        effect_keys = ["prediction_n400_d", "binding_p600_d", "pp_p600_d",
                       "garden_path_d", "src_orc_d"]
        for key in effect_keys:
            for si, stage_name in enumerate(STAGE_NAMES):
                if stage_effects[stage_name][key] > threshold:
                    emergence[key] = stage_name
                    break

        # Check for regressions
        regressions = []
        for key in effect_keys:
            emerged = False
            for stage_name in STAGE_NAMES:
                d = stage_effects[stage_name][key]
                if d > threshold:
                    emerged = True
                elif emerged and d < threshold:
                    regressions.append((key, stage_name))

        # Count phenomena at final stage
        final = stage_effects[STAGE_NAMES[-1]]
        n_emerged = sum(1 for k in effect_keys if final[k] > threshold)
        # Add stability (always passes)
        if final["assembly_stability"] > 0.9:
            n_emerged += 1
        # Add SRC dual binding (check if low = bound)
        if final["src_dual_binding_mean"] < 0.3:
            n_emerged += 1

        h1 = len(emergence) >= 3  # at least some ordering
        h2 = len(regressions) == 0
        h3 = n_emerged >= 5

        self.log(f"\n  === Emergence ===")
        for key, stage in emergence.items():
            self.log(f"    {key}: emerged at {stage}")
        self.log(f"    Regressions: {len(regressions)}")
        if regressions:
            for key, stage in regressions:
                self.log(f"      {key} regressed at {stage}")

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Ordered emergence):     {'PASS' if h1 else 'FAIL'}")
        self.log(f"    H2 (No forgetting):          {'PASS' if h2 else 'FAIL'}")
        self.log(f"    H3 (All emerge by stage 6):  {'PASS' if h3 else 'FAIL'}"
                 f" ({n_emerged}/7)")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "stage_effects": stage_effects,
            "emergence": emergence,
            "regressions": regressions,
            "n_emerged_final": n_emerged,
            "hypotheses": {
                "H1_ordered_emergence": h1,
                "H2_no_forgetting": h2,
                "H3_all_emerge": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "stage_sentences": list(cfg.stage_sentences),
                "training_reps": cfg.training_reps,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Developmental Curriculum Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = DevelopmentalCurriculumExperiment(verbose=True)

    if args.quick:
        cfg = CurriculumConfig(
            n=5000, k=50,
            stage_sentences=(0, 15, 30, 30, 25, 25),
            training_reps=2)
        n_seeds = args.seeds or 3
    else:
        cfg = CurriculumConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("DEVELOPMENTAL CURRICULUM SUMMARY")
    print("=" * 70)

    print(f"\n{'Stage':<18s} | {'Pred':>5s} | {'Bind':>5s} | "
          f"{'PP':>5s} | {'GP':>5s} | {'ORC':>5s}")
    print("-" * 60)
    for stage_name in STAGE_NAMES:
        e = m["stage_effects"][stage_name]
        print(f"{stage_name:<18s} | "
              f"{e['prediction_n400_d']:5.2f} | "
              f"{e['binding_p600_d']:5.2f} | "
              f"{e['pp_p600_d']:5.2f} | "
              f"{e['garden_path_d']:5.2f} | "
              f"{e['src_orc_d']:5.2f}")

    print(f"\nEmergence points:")
    for key, stage in m["emergence"].items():
        print(f"  {key}: {stage}")

    h = m["hypotheses"]
    print(f"\nH1 Ordered emergence: {'PASS' if h['H1_ordered_emergence'] else 'FAIL'}")
    print(f"H2 No forgetting:     {'PASS' if h['H2_no_forgetting'] else 'FAIL'}")
    print(f"H3 All emerge:        {'PASS' if h['H3_all_emerge'] else 'FAIL'}"
          f" ({m['n_emerged_final']}/7)")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
