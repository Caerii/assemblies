"""
Generalization to Untrained Constructions

Tests whether phenomena emerge from composition of trained components
rather than requiring direct experience with each construction.

Three sub-conditions, each with an independently trained brain:

A. ORC Generalization:
   Train on SVO + SRC only (orc_prob=0.0). Test ORC.
   The system has learned AGENT, REL_AGENT, and REL_PATIENT bindings,
   but has NEVER seen AGENT+REL_PATIENT dual binding.
   Does the conflicting ORC pattern still produce elevated P600?

B. Garden-Path Generalization:
   Train on SVO + SVO+PP only (rel_prob=0.0). No relative clauses.
   Test "agent verb patient VERB2" — system never saw a second verb.
   Does N400 still spike for the unexpected verb?

C. Recursive Depth Generalization:
   Train with max_pp_depth=1 only (depth-0 PPs only). Test at depth 1.
   Does the P600 double dissociation hold at the untrained depth?

If all three pass, the mechanism generalizes compositionally — it doesn't
need to memorize each specific construction.

Usage:
    uv run python research/experiments/primitives/test_generalization.py
    uv run python research/experiments/primitives/test_generalization.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
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
class GeneralizationConfig:
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
    # Training
    n_train_sentences: int = 100
    training_reps: int = 3
    # Test
    n_test_items: int = 5


def _make_brain(cfg, vocab, seed):
    """Create and return a fresh brain."""
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    return create_language_brain(bcfg, vocab, seed)


def run_trial(
    cfg: GeneralizationConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one generalization trial with three sub-conditions."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    preps = vocab.words_for_category("PREP")
    locs = vocab.words_for_category("LOCATION")

    n_train = cfg.n_train_sentences * cfg.training_reps

    # ══════════════════════════════════════════════════════════════
    # Condition A: ORC generalization (train SRC only, test ORC)
    # ══════════════════════════════════════════════════════════════
    brain_a = _make_brain(cfg, vocab, seed)
    grammar_a = RecursiveCFG(
        pp_prob=0.3, rel_prob=0.5, orc_prob=0.0,  # SRC only
        max_pp_depth=1, vocab=vocab, rng=np.random.default_rng(seed))
    sents_a = grammar_a.generate_batch(n_train)
    for s in sents_a:
        train_sentence(brain_a, s, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)
    brain_a.disable_plasticity = True
    lexicon_a = build_lexicon(brain_a, vocab, cfg.lexicon_readout_rounds)

    # Measure: SRC dual-binding (trained) vs ORC dual-binding (untrained)
    src_p600_rel, orc_p600_rel = [], []
    src_n400_main, orc_n400_main = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        orc_agent = nouns[(i + 1) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # SRC dual-binding: agent -> ROLE_REL_AGENT (trained)
        src_p600_rel.append(measure_p600(
            brain_a, agent, "NOUN_CORE", "ROLE_REL_AGENT",
            cfg.n_settling_rounds))

        # ORC dual-binding: agent -> ROLE_REL_PATIENT (NEVER trained)
        orc_p600_rel.append(measure_p600(
            brain_a, agent, "NOUN_CORE", "ROLE_REL_PATIENT",
            cfg.n_settling_rounds))

        # SRC main verb N400
        activate_word(brain_a, agent, "NOUN_CORE", 3)
        activate_word(brain_a, "that", "COMP_CORE", 3)
        activate_word(brain_a, rel_verb, "VERB_CORE", 3)
        activate_word(brain_a, rel_patient, "NOUN_CORE", 3)
        activate_word(brain_a, agent, "NOUN_CORE", 3)
        brain_a.inhibit_areas(["PREDICTION"])
        brain_a.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_src = np.array(brain_a.areas["PREDICTION"].winners, dtype=np.uint32)
        src_n400_main.append(measure_n400(pred_src, lexicon_a[main_verb]))

        # ORC main verb N400 (novel construction)
        activate_word(brain_a, agent, "NOUN_CORE", 3)
        activate_word(brain_a, "that", "COMP_CORE", 3)
        activate_word(brain_a, orc_agent, "NOUN_CORE", 3)
        activate_word(brain_a, rel_verb, "VERB_CORE", 3)
        activate_word(brain_a, agent, "NOUN_CORE", 3)
        brain_a.inhibit_areas(["PREDICTION"])
        brain_a.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_orc = np.array(brain_a.areas["PREDICTION"].winners, dtype=np.uint32)
        orc_n400_main.append(measure_n400(pred_orc, lexicon_a[main_verb]))

    brain_a.disable_plasticity = False

    # ══════════════════════════════════════════════════════════════
    # Condition B: Garden-path generalization (no RC training)
    # ══════════════════════════════════════════════════════════════
    brain_b = _make_brain(cfg, vocab, seed + 1000)
    grammar_b = RecursiveCFG(
        pp_prob=0.4, rel_prob=0.0,  # SVO + PP only, no relative clauses
        max_pp_depth=1, vocab=vocab, rng=np.random.default_rng(seed + 1000))
    sents_b = grammar_b.generate_batch(n_train)
    for s in sents_b:
        train_sentence(brain_b, s, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)
    brain_b.disable_plasticity = True
    lexicon_b = build_lexicon(brain_b, vocab, cfg.lexicon_readout_rounds)

    # After apparent SVO, measure N400 for verb vs preposition
    gp_n400_verb, gp_n400_prep = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 1) % len(nouns)]
        second_verb = verbs[(i + 1) % len(verbs)]
        prep = preps[i % len(preps)]

        activate_word(brain_b, agent, "NOUN_CORE", 3)
        activate_word(brain_b, verb, "VERB_CORE", 3)
        activate_word(brain_b, patient, "NOUN_CORE", 3)
        brain_b.inhibit_areas(["PREDICTION"])
        brain_b.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_post = np.array(
            brain_b.areas["PREDICTION"].winners, dtype=np.uint32)

        gp_n400_verb.append(measure_n400(pred_post, lexicon_b[second_verb]))
        gp_n400_prep.append(measure_n400(pred_post, lexicon_b[prep]))

    brain_b.disable_plasticity = False

    # ══════════════════════════════════════════════════════════════
    # Condition C: Recursive depth generalization (depth-0 only)
    # ══════════════════════════════════════════════════════════════
    brain_c = _make_brain(cfg, vocab, seed + 2000)
    grammar_c = RecursiveCFG(
        pp_prob=0.5, recursive_pp_prob=0.0, max_pp_depth=1,  # depth 0 only
        rel_prob=0.0, vocab=vocab, rng=np.random.default_rng(seed + 2000))
    sents_c = grammar_c.generate_batch(n_train)
    for s in sents_c:
        train_sentence(brain_c, s, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)
    brain_c.disable_plasticity = True
    lexicon_c = build_lexicon(brain_c, vocab, cfg.lexicon_readout_rounds)

    # Measure P600 at depth 0 (trained) and depth 1 (untrained)
    d0_p600_gram, d0_p600_cv = [], []
    d1_p600_gram, d1_p600_cv = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 1) % len(nouns)]
        prep0 = preps[i % len(preps)]
        gram_pp0 = locs[i % len(locs)]
        cv_pp0 = verbs[(i + 2) % len(verbs)]

        # Depth 0 (trained)
        d0_p600_gram.append(measure_p600(
            brain_c, gram_pp0, "NOUN_CORE", "ROLE_PP_OBJ",
            cfg.n_settling_rounds))
        d0_p600_cv.append(measure_p600(
            brain_c, cv_pp0, "VERB_CORE", "ROLE_PP_OBJ",
            cfg.n_settling_rounds))

        # Depth 1 (untrained — ROLE_PP_OBJ_1 was never bound)
        prep1 = preps[(i + 1) % len(preps)]
        gram_pp1 = locs[(i + 1) % len(locs)]
        cv_pp1 = verbs[(i + 3) % len(verbs)]

        d1_p600_gram.append(measure_p600(
            brain_c, gram_pp1, "NOUN_CORE", "ROLE_PP_OBJ_1",
            cfg.n_settling_rounds))
        d1_p600_cv.append(measure_p600(
            brain_c, cv_pp1, "VERB_CORE", "ROLE_PP_OBJ_1",
            cfg.n_settling_rounds))

    brain_c.disable_plasticity = False

    return {
        # Condition A: ORC generalization
        "orc_p600_rel_patient": float(np.mean(orc_p600_rel)),
        "src_p600_rel_agent": float(np.mean(src_p600_rel)),
        "orc_n400_main": float(np.mean(orc_n400_main)),
        "src_n400_main": float(np.mean(src_n400_main)),
        # Condition B: Garden-path generalization
        "gp_n400_verb": float(np.mean(gp_n400_verb)),
        "gp_n400_prep": float(np.mean(gp_n400_prep)),
        # Condition C: Depth generalization
        "d0_p600_gram": float(np.mean(d0_p600_gram)),
        "d0_p600_cv": float(np.mean(d0_p600_cv)),
        "d1_p600_gram": float(np.mean(d1_p600_gram)),
        "d1_p600_cv": float(np.mean(d1_p600_cv)),
    }


class GeneralizationExperiment(ExperimentBase):
    """Generalization to untrained constructions."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="generalization",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[GeneralizationConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or GeneralizationConfig(
            **{k: v for k, v in kwargs.items()
               if k in GeneralizationConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Generalization to Untrained Constructions")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = [
            "orc_p600_rel_patient", "src_p600_rel_agent",
            "orc_n400_main", "src_n400_main",
            "gp_n400_verb", "gp_n400_prep",
            "d0_p600_gram", "d0_p600_cv",
            "d1_p600_gram", "d1_p600_cv",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])

        # ── Condition A: ORC generalization ──
        t_dual = paired_ttest(
            vals["orc_p600_rel_patient"], vals["src_p600_rel_agent"])
        self.log(f"\n  === A: ORC Generalization (trained SRC only) ===")
        self.log(f"    SRC dual P600 (REL_AGENT, trained): "
                 f"{np.mean(vals['src_p600_rel_agent']):.3f}")
        self.log(f"    ORC dual P600 (REL_PATIENT, UNTRAINED): "
                 f"{np.mean(vals['orc_p600_rel_patient']):.3f}")
        self.log(f"    ORC > SRC: d={t_dual['d']:.2f}")

        t_n400_orc = paired_ttest(vals["orc_n400_main"], vals["src_n400_main"])
        self.log(f"    Main verb N400 — SRC: {np.mean(vals['src_n400_main']):.3f}"
                 f"  ORC: {np.mean(vals['orc_n400_main']):.3f}"
                 f"  d={t_n400_orc['d']:.2f}")

        # ── Condition B: Garden-path generalization ──
        t_gp = paired_ttest(vals["gp_n400_verb"], vals["gp_n400_prep"])
        self.log(f"\n  === B: Garden-Path Generalization (no RC training) ===")
        self.log(f"    Post-object N400 — prep: {np.mean(vals['gp_n400_prep']):.3f}"
                 f"  verb: {np.mean(vals['gp_n400_verb']):.3f}")
        self.log(f"    Verb > Prep: d={t_gp['d']:.2f}")

        # ── Condition C: Depth generalization ──
        t_d0 = paired_ttest(vals["d0_p600_cv"], vals["d0_p600_gram"])
        t_d1 = paired_ttest(vals["d1_p600_cv"], vals["d1_p600_gram"])
        self.log(f"\n  === C: Recursive Depth Generalization (trained depth 0 only) ===")
        self.log(f"    Depth 0 (trained) — gram: {np.mean(vals['d0_p600_gram']):.3f}"
                 f"  cv: {np.mean(vals['d0_p600_cv']):.3f}"
                 f"  d={t_d0['d']:.2f}")
        self.log(f"    Depth 1 (UNTRAINED) — gram: {np.mean(vals['d1_p600_gram']):.3f}"
                 f"  cv: {np.mean(vals['d1_p600_cv']):.3f}"
                 f"  d={t_d1['d']:.2f}")

        # Hypotheses
        h1 = np.mean(vals["orc_p600_rel_patient"]) > np.mean(vals["src_p600_rel_agent"])
        h2 = np.mean(vals["gp_n400_verb"]) > np.mean(vals["gp_n400_prep"])
        h3 = t_d1["d"] > 0.5

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (ORC untrained > SRC trained): "
                 f"{'PASS' if h1 else 'FAIL'} (d={t_dual['d']:.2f})")
        self.log(f"    H2 (GP verb > prep, no RC training): "
                 f"{'PASS' if h2 else 'FAIL'} (d={t_gp['d']:.2f})")
        self.log(f"    H3 (Depth-1 dissociation, untrained): "
                 f"{'PASS' if h3 else 'FAIL'} (d={t_d1['d']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "orc_generalization": {
                "src_p600_rel_agent": summarize(vals["src_p600_rel_agent"]),
                "orc_p600_rel_patient": summarize(vals["orc_p600_rel_patient"]),
                "dual_binding_test": t_dual,
                "src_n400_main": summarize(vals["src_n400_main"]),
                "orc_n400_main": summarize(vals["orc_n400_main"]),
            },
            "garden_path_generalization": {
                "n400_prep": summarize(vals["gp_n400_prep"]),
                "n400_verb": summarize(vals["gp_n400_verb"]),
                "test": t_gp,
            },
            "depth_generalization": {
                "depth_0": {"p600_gram": summarize(vals["d0_p600_gram"]),
                            "p600_cv": summarize(vals["d0_p600_cv"]),
                            "test": t_d0},
                "depth_1": {"p600_gram": summarize(vals["d1_p600_gram"]),
                            "p600_cv": summarize(vals["d1_p600_cv"]),
                            "test": t_d1},
            },
            "hypotheses": {
                "H1_orc_untrained": h1,
                "H2_gp_no_rc": h2,
                "H3_depth_untrained": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
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
        description="Generalization Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = GeneralizationExperiment(verbose=True)

    if args.quick:
        cfg = GeneralizationConfig(
            n=5000, k=50,
            n_train_sentences=60, training_reps=2)
        n_seeds = args.seeds or 5
    else:
        cfg = GeneralizationConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("GENERALIZATION SUMMARY")
    print("=" * 70)

    print("\nA: ORC Generalization (trained SRC only):")
    og = m["orc_generalization"]
    print(f"  SRC REL_AGENT P600 (trained):   {og['src_p600_rel_agent']['mean']:.3f}")
    print(f"  ORC REL_PATIENT P600 (untrained): {og['orc_p600_rel_patient']['mean']:.3f}")
    print(f"  d={og['dual_binding_test']['d']:.2f}")

    print("\nB: Garden-Path Generalization (no RC training):")
    gp = m["garden_path_generalization"]
    print(f"  Prep N400: {gp['n400_prep']['mean']:.3f}"
          f"  Verb N400: {gp['n400_verb']['mean']:.3f}"
          f"  d={gp['test']['d']:.2f}")

    print("\nC: Depth Generalization (trained depth 0 only):")
    for d in [0, 1]:
        dd = m["depth_generalization"][f"depth_{d}"]
        label = "trained" if d == 0 else "UNTRAINED"
        print(f"  Depth {d} ({label}) — gram: {dd['p600_gram']['mean']:.3f}"
              f"  cv: {dd['p600_cv']['mean']:.3f}"
              f"  d={dd['test']['d']:.2f}")

    h = m["hypotheses"]
    print(f"\nH1 ORC untrained:    {'PASS' if h['H1_orc_untrained'] else 'FAIL'}")
    print(f"H2 GP no RC:         {'PASS' if h['H2_gp_no_rc'] else 'FAIL'}")
    print(f"H3 Depth untrained:  {'PASS' if h['H3_depth_untrained'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
