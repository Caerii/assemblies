"""
Integrated Learner: Closing the Unsupervised-Supervised Gap

Ablation study testing three interventions that address the gaps found in
unsupervised binding (P600 at 14% of supervised) and grounded language
(semantic N400 fails to propagate through prediction pathway):

  1. Hebbian routing — plasticity ON during role discovery stabilization,
     so NOUN_MARKER->STRUCT connections strengthen over training
  2. Semantic prediction — co-project sensory features into PREDICTION
     during training, so predictions encode "what kind of thing"
  3. P600-guided retraining — measure instability after binding and do
     extra rounds when it's high (error-driven learning)

Five conditions (2x2x2 with corners):
  A. Baseline unsupervised (none)
  B. + Hebbian routing only
  C. + Semantic prediction only
  D. + P600-guided retraining only
  E. All three combined

Plus a supervised comparison using pre-specified role areas.

Hypotheses:
  H1: Hebbian routing improves P600 (B > A)
  H2: Semantic prediction creates semantic N400 (C has d > 0.5)
  H3: P600 feedback improves binding (D > A)
  H4: Combined shows all three improvements
  H5: Combined P600 > 50% of supervised P600

Usage:
    uv run python research/experiments/primitives/test_integrated_learner.py
    uv run python research/experiments/primitives/test_integrated_learner.py --quick
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
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400, measure_p600
from research.experiments.lib.unsupervised import (
    UnsupervisedConfig,
    build_role_mapping,
    train_sentence_integrated,
)
from research.experiments.lib.grounding import (
    SENSORY_FEATURES,
    SEMANTIC_GROUPS,
    IntegratedConfig,
    create_integrated_brain,
    build_grounded_lexicon,
)


@dataclass
class IntegratedLearnerConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    grounding_rounds: int = 10
    # Structural pool
    n_struct_areas: int = 6
    refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    # Training
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    extra_binding_rounds: int = 5
    instability_threshold: float = 0.3
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    n_train_sentences: int = 100
    training_reps: int = 3
    pp_prob: float = 0.4
    # Test
    n_test_items: int = 5


# Ablation conditions: (hebbian, grounded_pred, p600_feedback)
CONDITIONS = {
    "A_baseline":     (False, False, False),
    "B_hebbian":      (True,  False, False),
    "C_semantic":     (False, True,  False),
    "D_p600fb":       (False, False, True),
    "E_combined":     (True,  True,  True),
}


def run_condition(
    cfg: IntegratedLearnerConfig,
    train_sents: List[Dict[str, Any]],
    seed: int,
    use_hebbian: bool,
    use_grounded_pred: bool,
    use_p600_fb: bool,
) -> Dict[str, Any]:
    """Run one condition of the ablation study."""
    vocab = DEFAULT_VOCAB

    # Create integrated brain (unsupervised + grounded)
    icfg = IntegratedConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds,
        grounding_rounds=cfg.grounding_rounds,
        n_struct_areas=cfg.n_struct_areas,
        refractory_period=cfg.refractory_period,
        inhibition_strength=cfg.inhibition_strength,
        stabilize_rounds=cfg.stabilize_rounds,
        train_rounds_per_pair=cfg.train_rounds_per_pair,
        binding_rounds=cfg.binding_rounds,
        extra_binding_rounds=cfg.extra_binding_rounds,
        instability_threshold=cfg.instability_threshold,
    )
    brain, struct_areas = create_integrated_brain(icfg, vocab, seed)

    # Train
    for sent in train_sents:
        train_sentence_integrated(
            brain, sent, vocab, struct_areas, icfg,
            use_hebbian_routing=use_hebbian,
            use_grounded_prediction=use_grounded_pred,
            use_p600_feedback=use_p600_fb,
            extra_binding_rounds=cfg.extra_binding_rounds,
            instability_threshold=cfg.instability_threshold,
        )

    brain.disable_plasticity = True

    # Role mapping and purity
    role_map = build_role_mapping(
        brain, vocab, struct_areas, train_sents[:30], cfg.stabilize_rounds)

    patient_area = None
    for area, pos_label in role_map.items():
        if pos_label == "pos_1":
            patient_area = area
            break

    # Purity measurement
    test_grammar = SimpleCFG(
        pp_prob=0.0, vocab=vocab,
        rng=np.random.default_rng(seed + 5000))
    test_sents = test_grammar.generate_batch(30)

    from collections import Counter
    from research.experiments.lib.unsupervised import discover_role_area

    pos_assignments = []
    for sent in test_sents:
        words = sent["words"]
        cats = sent["categories"]
        for name in struct_areas:
            brain.clear_refractory(name)
        brain.inhibit_areas(struct_areas)

        noun_idx = 0
        for w, c in zip(words, cats):
            if c in ("NOUN", "LOCATION"):
                core = vocab.core_area_for(w)
                activate_word(brain, w, core, 3)
                winner = discover_role_area(
                    brain, "NOUN_MARKER", struct_areas, cfg.stabilize_rounds)
                if winner:
                    pos_assignments.append((winner, noun_idx))
                noun_idx += 1

    area_counts = {}
    for area, pos in pos_assignments:
        area_counts.setdefault(area, Counter())[pos] += 1
    correct = sum(c.most_common(1)[0][1] for c in area_counts.values())
    purity = correct / len(pos_assignments) if pos_assignments else 0.0

    # Build lexicon — grounded if semantic prediction was used
    if use_grounded_pred:
        lexicon = build_grounded_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
    else:
        lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    # N400 measurement at object position
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    ni = cfg.n_test_items

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
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        n400_gram.append(measure_n400(predicted, lexicon[gram_obj]))
        n400_cv.append(measure_n400(predicted, lexicon[cv_obj]))

    # Semantic N400: same-group vs different-group
    animal_words = set(SEMANTIC_GROUPS.get("ANIMAL", []))
    animals = [w for w in nouns if w in animal_words]
    non_animals = [w for w in nouns if w not in animal_words]

    sem_same, sem_diff = [], []
    if animals and non_animals:
        for i in range(ni):
            agent = animals[i % len(animals)]
            verb = verbs[i % len(verbs)]

            activate_word(brain, agent, "NOUN_CORE", 3)
            activate_word(brain, verb, "VERB_CORE", 3)
            brain.inhibit_areas(["PREDICTION"])
            brain.project({}, {"VERB_CORE": ["PREDICTION"]})
            predicted = np.array(
                brain.areas["PREDICTION"].winners, dtype=np.uint32)

            same_obj = animals[(i + 1) % len(animals)]
            diff_obj = non_animals[i % len(non_animals)]
            sem_same.append(measure_n400(predicted, lexicon[same_obj]))
            sem_diff.append(measure_n400(predicted, lexicon[diff_obj]))

    # P600 measurement using discovered patient area
    p600_gram, p600_cv = [], []
    if patient_area:
        for i in range(ni):
            gram_obj = nouns[(i + 1) % len(nouns)]
            cv_obj = verbs[(i + 1) % len(verbs)]
            p600_gram.append(measure_p600(
                brain, gram_obj, "NOUN_CORE", patient_area,
                cfg.n_settling_rounds))
            p600_cv.append(measure_p600(
                brain, cv_obj, "VERB_CORE", patient_area,
                cfg.n_settling_rounds))

    brain.disable_plasticity = False

    return {
        "purity": purity,
        "n400_gram": n400_gram,
        "n400_cv": n400_cv,
        "sem_same": sem_same,
        "sem_diff": sem_diff,
        "p600_gram": p600_gram,
        "p600_cv": p600_cv,
    }


def run_supervised(
    cfg: IntegratedLearnerConfig,
    train_sents: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    """Run supervised baseline for comparison."""
    vocab = DEFAULT_VOCAB
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    ni = cfg.n_test_items

    p600_gram, p600_cv = [], []
    for i in range(ni):
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]
        p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

    brain.disable_plasticity = False

    return {
        "p600_gram": p600_gram,
        "p600_cv": p600_cv,
    }


def run_trial(
    cfg: IntegratedLearnerConfig,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Run all conditions for one seed."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB
    n_train = cfg.n_train_sentences * cfg.training_reps

    grammar = SimpleCFG(pp_prob=cfg.pp_prob, vocab=vocab, rng=rng)
    train_sents = grammar.generate_batch(n_train)

    results = {}
    for cond_name, (heb, grd, p6) in CONDITIONS.items():
        results[cond_name] = run_condition(
            cfg, train_sents, seed, heb, grd, p6)

    results["supervised"] = run_supervised(cfg, train_sents, seed)

    return results


class IntegratedLearnerExperiment(ExperimentBase):
    """Integrated learner ablation experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="integrated_learner",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[IntegratedLearnerConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or IntegratedLearnerConfig(
            **{k: v for k, v in kwargs.items()
               if k in IntegratedLearnerConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Integrated Learner: Ablation Study")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_struct={cfg.n_struct_areas}, "
                 f"grounding={cfg.grounding_rounds}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  p600_threshold={cfg.instability_threshold}, "
                 f"extra_rounds={cfg.extra_binding_rounds}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Per-condition accumulators
        cond_data = {}
        for cond in list(CONDITIONS.keys()) + ["supervised"]:
            cond_data[cond] = {
                "purity": [], "p600_gram": [], "p600_cv": [],
                "sem_same": [], "sem_diff": [],
            }

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            for cond_name in CONDITIONS:
                r = trial[cond_name]
                cond_data[cond_name]["purity"].append(r["purity"])
                if r["p600_gram"]:
                    cond_data[cond_name]["p600_gram"].append(
                        float(np.mean(r["p600_gram"])))
                    cond_data[cond_name]["p600_cv"].append(
                        float(np.mean(r["p600_cv"])))
                if r["sem_same"]:
                    cond_data[cond_name]["sem_same"].append(
                        float(np.mean(r["sem_same"])))
                    cond_data[cond_name]["sem_diff"].append(
                        float(np.mean(r["sem_diff"])))

            # Supervised P600
            sv = trial["supervised"]
            cond_data["supervised"]["p600_gram"].append(
                float(np.mean(sv["p600_gram"])))
            cond_data["supervised"]["p600_cv"].append(
                float(np.mean(sv["p600_cv"])))

        # Compute per-condition effect sizes
        cond_effects = {}
        for cond_name in CONDITIONS:
            cd = cond_data[cond_name]
            p600_d = (paired_ttest(cd["p600_gram"], cd["p600_cv"])
                      if cd["p600_gram"] else {"d": 0.0})
            sem_test = (paired_ttest(cd["sem_diff"], cd["sem_same"])
                        if cd["sem_same"] else {"d": 0.0})
            cond_effects[cond_name] = {
                "purity": float(np.mean(cd["purity"])) if cd["purity"] else 0.0,
                "p600_d": abs(p600_d["d"]) if isinstance(p600_d, dict) else 0.0,
                "sem_d": sem_test["d"] if isinstance(sem_test, dict) else 0.0,
            }

        # Supervised P600 effect
        sv_cd = cond_data["supervised"]
        sv_p600 = paired_ttest(sv_cd["p600_gram"], sv_cd["p600_cv"])
        sup_p600_d = abs(sv_p600["d"])

        # Report
        self.log(f"\n  {'Condition':<18} | {'Purity':>7} | {'P600 |d|':>9} | {'Sem d':>7}")
        self.log(f"  {'-'*18}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}")
        for cond_name in CONDITIONS:
            ce = cond_effects[cond_name]
            self.log(f"  {cond_name:<18} | {ce['purity']:7.3f} | "
                     f"{ce['p600_d']:9.2f} | {ce['sem_d']:7.2f}")
        self.log(f"  {'supervised':<18} | {'  ---':>7} | "
                 f"{sup_p600_d:9.2f} | {'  ---':>7}")

        # Hypotheses
        a_p600 = cond_effects["A_baseline"]["p600_d"]
        b_p600 = cond_effects["B_hebbian"]["p600_d"]
        c_sem = cond_effects["C_semantic"]["sem_d"]
        d_p600 = cond_effects["D_p600fb"]["p600_d"]
        e_p600 = cond_effects["E_combined"]["p600_d"]
        e_sem = cond_effects["E_combined"]["sem_d"]

        h1 = b_p600 > a_p600
        h2 = c_sem > 0.5
        h3 = d_p600 > a_p600
        h4 = (e_p600 > a_p600 and e_sem > 0.0)
        h5 = e_p600 > sup_p600_d * 0.5 if sup_p600_d > 0 else False

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Hebbian improves P600):     "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" (B={b_p600:.2f} vs A={a_p600:.2f})")
        self.log(f"    H2 (Semantic N400 d > 0.5):     "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" (C sem_d={c_sem:.2f})")
        self.log(f"    H3 (P600 feedback improves):    "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (D={d_p600:.2f} vs A={a_p600:.2f})")
        self.log(f"    H4 (Combined improves both):    "
                 f"{'PASS' if h4 else 'FAIL'}"
                 f" (E p600={e_p600:.2f}, sem={e_sem:.2f})")
        self.log(f"    H5 (Combined >= 50% supervised):"
                 f" {'PASS' if h5 else 'FAIL'}"
                 f" (E={e_p600:.2f} vs sup={sup_p600_d:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "conditions": cond_effects,
            "supervised_p600_d": sup_p600_d,
            "hypotheses": {
                "H1_hebbian_improves": h1,
                "H2_semantic_n400": h2,
                "H3_p600_feedback": h3,
                "H4_combined": h4,
                "H5_comparable": h5,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_struct_areas": cfg.n_struct_areas,
                "grounding_rounds": cfg.grounding_rounds,
                "extra_binding_rounds": cfg.extra_binding_rounds,
                "instability_threshold": cfg.instability_threshold,
                "n_train_sentences": cfg.n_train_sentences,
                "training_reps": cfg.training_reps,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Learner Ablation Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = IntegratedLearnerExperiment(verbose=True)

    if args.quick:
        cfg = IntegratedLearnerConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2,
            n_test_items=4)
        n_seeds = args.seeds or 3
    else:
        cfg = IntegratedLearnerConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("INTEGRATED LEARNER SUMMARY")
    print("=" * 70)
    print(f"\nH1 Hebbian routing:   {'PASS' if h['H1_hebbian_improves'] else 'FAIL'}")
    print(f"H2 Semantic N400:     {'PASS' if h['H2_semantic_n400'] else 'FAIL'}")
    print(f"H3 P600 feedback:     {'PASS' if h['H3_p600_feedback'] else 'FAIL'}")
    print(f"H4 Combined:          {'PASS' if h['H4_combined'] else 'FAIL'}")
    print(f"H5 Comparable to sup: {'PASS' if h['H5_comparable'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
