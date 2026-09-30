"""
Unsupervised Role Binding: N400/P600 Without Role Labels

Tests whether dynamically discovered structural roles (via shared category
marker + LRI + mutual inhibition) support the same prediction and binding
phenomena as pre-specified role areas.

The system receives sentences as word sequences with category tags but
NO role annotations. Structural roles are discovered from word order:
each noun gets routed to a STRUCT area via competition, and LRI forces
successive nouns to different areas.

Three conditions:
  A. SVO only: test N400/P600 at object position with discovered roles
  B. SVO+PP: test PP-object binding with discovered roles
  C. Comparison: same sentences through supervised pipeline for comparison

Hypotheses:
  H1: Role separation — discovered areas specialize by position (purity > 0.8)
  H2: N400 works — prediction N400 |d| > 2.0 (unaffected by binding method)
  H3: P600 works — binding P600 using discovered roles produces |d| > 1.0
  H4: Comparable — unsupervised P600 |d| within 50% of supervised P600 |d|

Usage:
    uv run python research/experiments/primitives/test_unsupervised_binding.py
    uv run python research/experiments/primitives/test_unsupervised_binding.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from collections import Counter
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
    create_unsupervised_brain,
    discover_role_area,
    train_sentence_unsupervised,
    build_role_mapping,
)


@dataclass
class UnsupervisedBindingConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    # Structural pool
    n_struct_areas: int = 6
    refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    # Training
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    n_train_sentences: int = 100
    training_reps: int = 3
    pp_prob: float = 0.4
    # Test
    n_test_items: int = 5


def run_trial(
    cfg: UnsupervisedBindingConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one unsupervised binding trial."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB
    n_train = cfg.n_train_sentences * cfg.training_reps

    # Generate training sentences (with roles for supervised comparison,
    # but unsupervised path only uses words + categories)
    grammar = SimpleCFG(
        pp_prob=cfg.pp_prob, vocab=vocab, rng=rng)
    train_sents = grammar.generate_batch(n_train)

    results = {}

    # ── Condition A: Unsupervised ──────────────────────────────────
    ucfg = UnsupervisedConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds,
        n_struct_areas=cfg.n_struct_areas,
        refractory_period=cfg.refractory_period,
        inhibition_strength=cfg.inhibition_strength,
        stabilize_rounds=cfg.stabilize_rounds,
        train_rounds_per_pair=cfg.train_rounds_per_pair,
        binding_rounds=cfg.binding_rounds,
    )
    brain_u, struct_areas = create_unsupervised_brain(ucfg, vocab, seed)

    # Track which areas bind to which positions
    all_bindings = []
    for sent in train_sents:
        bindings = train_sentence_unsupervised(
            brain_u, sent, vocab, struct_areas, ucfg)
        all_bindings.append(bindings)

    # Discover role mapping
    brain_u.disable_plasticity = True
    role_map = build_role_mapping(
        brain_u, vocab, struct_areas, train_sents[:30], cfg.stabilize_rounds)

    # Find which struct area maps to patient position (pos_1 in SVO)
    patient_area = None
    agent_area = None
    for area, pos_label in role_map.items():
        if pos_label == "pos_1":
            patient_area = area
        if pos_label == "pos_0":
            agent_area = area

    # Role separation purity
    # Run test sentences and check position consistency
    test_svo = SimpleCFG(pp_prob=0.0, vocab=vocab,
                         rng=np.random.default_rng(seed + 5000))
    test_sents = test_svo.generate_batch(30)

    pos_assignments = []
    for sent in test_sents:
        words = sent["words"]
        cats = sent["categories"]
        for name in struct_areas:
            brain_u.clear_refractory(name)
        brain_u.inhibit_areas(struct_areas)

        noun_idx = 0
        for i, (w, c) in enumerate(zip(words, cats)):
            if c in ("NOUN", "LOCATION"):
                core = vocab.core_area_for(w)
                activate_word(brain_u, w, core, 3)
                winner = discover_role_area(
                    brain_u, "NOUN_MARKER", struct_areas, cfg.stabilize_rounds)
                if winner:
                    pos_assignments.append((winner, noun_idx))
                noun_idx += 1

    # Compute purity
    area_counts = {}
    for area, pos in pos_assignments:
        area_counts.setdefault(area, Counter())[pos] += 1
    correct = sum(c.most_common(1)[0][1] for c in area_counts.values())
    purity = correct / len(pos_assignments) if pos_assignments else 0.0

    # Build lexicon for N400 measurement
    lexicon_u = build_lexicon(brain_u, vocab, cfg.lexicon_readout_rounds)

    # N400 measurement at object position
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    ni = cfg.n_test_items

    u_n400_gram, u_n400_cv = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        activate_word(brain_u, agent, "NOUN_CORE", 3)
        activate_word(brain_u, verb, "VERB_CORE", 3)
        brain_u.inhibit_areas(["PREDICTION"])
        brain_u.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain_u.areas["PREDICTION"].winners, dtype=np.uint32)

        u_n400_gram.append(measure_n400(predicted, lexicon_u[gram_obj]))
        u_n400_cv.append(measure_n400(predicted, lexicon_u[cv_obj]))

    # P600 measurement using discovered patient area
    u_p600_gram, u_p600_cv = [], []
    if patient_area:
        for i in range(ni):
            gram_obj = nouns[(i + 1) % len(nouns)]
            cv_obj = verbs[(i + 1) % len(verbs)]
            u_p600_gram.append(measure_p600(
                brain_u, gram_obj, "NOUN_CORE", patient_area,
                cfg.n_settling_rounds))
            u_p600_cv.append(measure_p600(
                brain_u, cv_obj, "VERB_CORE", patient_area,
                cfg.n_settling_rounds))

    brain_u.disable_plasticity = False

    results["unsupervised"] = {
        "purity": purity,
        "role_map": role_map,
        "patient_area": patient_area,
        "agent_area": agent_area,
        "n400_gram": u_n400_gram,
        "n400_cv": u_n400_cv,
        "p600_gram": u_p600_gram,
        "p600_cv": u_p600_cv,
    }

    # ── Condition C: Supervised comparison ─────────────────────────
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain_s = create_language_brain(bcfg, vocab, seed)

    for sent in train_sents:
        train_sentence(brain_s, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain_s.disable_plasticity = True
    lexicon_s = build_lexicon(brain_s, vocab, cfg.lexicon_readout_rounds)

    s_n400_gram, s_n400_cv = [], []
    s_p600_gram, s_p600_cv = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        activate_word(brain_s, agent, "NOUN_CORE", 3)
        activate_word(brain_s, verb, "VERB_CORE", 3)
        brain_s.inhibit_areas(["PREDICTION"])
        brain_s.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain_s.areas["PREDICTION"].winners, dtype=np.uint32)

        s_n400_gram.append(measure_n400(predicted, lexicon_s[gram_obj]))
        s_n400_cv.append(measure_n400(predicted, lexicon_s[cv_obj]))

        s_p600_gram.append(measure_p600(
            brain_s, gram_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))
        s_p600_cv.append(measure_p600(
            brain_s, cv_obj, "VERB_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

    results["supervised"] = {
        "n400_gram": s_n400_gram,
        "n400_cv": s_n400_cv,
        "p600_gram": s_p600_gram,
        "p600_cv": s_p600_cv,
    }

    return results


class UnsupervisedBindingExperiment(ExperimentBase):
    """Unsupervised role binding experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="unsupervised_binding",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[UnsupervisedBindingConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or UnsupervisedBindingConfig(
            **{k: v for k, v in kwargs.items()
               if k in UnsupervisedBindingConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Unsupervised Role Binding: N400/P600 Without Role Labels")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_struct={cfg.n_struct_areas}, "
                 f"LRI={cfg.refractory_period}/{cfg.inhibition_strength}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        purity_vals = []
        # Unsupervised
        u_n400_gram_vals, u_n400_cv_vals = [], []
        u_p600_gram_vals, u_p600_cv_vals = [], []
        # Supervised
        s_n400_gram_vals, s_n400_cv_vals = [], []
        s_p600_gram_vals, s_p600_cv_vals = [], []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            u = trial["unsupervised"]
            purity_vals.append(u["purity"])
            u_n400_gram_vals.append(float(np.mean(u["n400_gram"])))
            u_n400_cv_vals.append(float(np.mean(u["n400_cv"])))
            if u["p600_gram"]:
                u_p600_gram_vals.append(float(np.mean(u["p600_gram"])))
                u_p600_cv_vals.append(float(np.mean(u["p600_cv"])))

            if s == 0:
                self.log(f"    Role mapping: {u['role_map']}")
                self.log(f"    Patient area: {u['patient_area']}")

            sv = trial["supervised"]
            s_n400_gram_vals.append(float(np.mean(sv["n400_gram"])))
            s_n400_cv_vals.append(float(np.mean(sv["n400_cv"])))
            s_p600_gram_vals.append(float(np.mean(sv["p600_gram"])))
            s_p600_cv_vals.append(float(np.mean(sv["p600_cv"])))

        # Compute effect sizes
        u_n400 = paired_ttest(u_n400_gram_vals, u_n400_cv_vals)
        s_n400 = paired_ttest(s_n400_gram_vals, s_n400_cv_vals)

        u_p600 = (paired_ttest(u_p600_gram_vals, u_p600_cv_vals)
                  if u_p600_gram_vals else {"d": 0.0})
        s_p600 = paired_ttest(s_p600_gram_vals, s_p600_cv_vals)

        # Report
        self.log(f"\n  === Role Discovery ===")
        self.log(f"    Purity: {np.mean(purity_vals):.3f} "
                 f"+/- {np.std(purity_vals) / max(np.sqrt(n_seeds), 1):.3f}")

        self.log(f"\n  === N400 (Prediction) ===")
        self.log(f"    Unsupervised: d={u_n400['d']:.2f}"
                 f" (gram={np.mean(u_n400_gram_vals):.4f},"
                 f" cv={np.mean(u_n400_cv_vals):.4f})")
        self.log(f"    Supervised:   d={s_n400['d']:.2f}"
                 f" (gram={np.mean(s_n400_gram_vals):.4f},"
                 f" cv={np.mean(s_n400_cv_vals):.4f})")

        self.log(f"\n  === P600 (Binding) ===")
        self.log(f"    Unsupervised: d={u_p600['d']:.2f}")
        self.log(f"    Supervised:   d={s_p600['d']:.2f}")

        # Hypotheses
        h1 = float(np.mean(purity_vals)) > 0.8
        h2 = abs(u_n400["d"]) > 2.0
        h3 = abs(u_p600["d"]) > 1.0
        h4 = (abs(u_p600["d"]) > abs(s_p600["d"]) * 0.5
               if abs(s_p600["d"]) > 0 else False)

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Purity > 0.8):            "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" ({np.mean(purity_vals):.3f})")
        self.log(f"    H2 (N400 |d| > 2.0):          "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" (|d|={abs(u_n400['d']):.2f})")
        self.log(f"    H3 (P600 |d| > 1.0):          "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (|d|={abs(u_p600['d']):.2f})")
        self.log(f"    H4 (Unsup >= 50% of sup):     "
                 f"{'PASS' if h4 else 'FAIL'}"
                 f" ({abs(u_p600['d']):.2f} vs {abs(s_p600['d']):.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "purity": summarize(purity_vals),
            "unsupervised_n400": u_n400,
            "unsupervised_p600": u_p600,
            "supervised_n400": s_n400,
            "supervised_p600": s_p600,
            "hypotheses": {
                "H1_role_separation": h1,
                "H2_n400_works": h2,
                "H3_p600_works": h3,
                "H4_comparable": h4,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_struct_areas": cfg.n_struct_areas,
                "n_train_sentences": cfg.n_train_sentences,
                "training_reps": cfg.training_reps,
                "pp_prob": cfg.pp_prob,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised Role Binding Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = UnsupervisedBindingExperiment(verbose=True)

    if args.quick:
        cfg = UnsupervisedBindingConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2,
            n_test_items=4)
        n_seeds = args.seeds or 3
    else:
        cfg = UnsupervisedBindingConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("UNSUPERVISED BINDING SUMMARY")
    print("=" * 70)
    print(f"\nH1 Role separation: {'PASS' if h['H1_role_separation'] else 'FAIL'}")
    print(f"H2 N400 works:      {'PASS' if h['H2_n400_works'] else 'FAIL'}")
    print(f"H3 P600 works:      {'PASS' if h['H3_p600_works'] else 'FAIL'}")
    print(f"H4 Comparable:      {'PASS' if h['H4_comparable'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
