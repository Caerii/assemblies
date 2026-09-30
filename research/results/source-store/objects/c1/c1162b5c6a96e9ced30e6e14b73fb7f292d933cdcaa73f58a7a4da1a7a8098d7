"""
Depth Generalization: Does Prediction/Binding Transfer Across Recursion Depths?

Inspired by Schulz et al. (2025), which shows that language models struggle
with recursion depth (not length) — models trained at depth d fail at depth d+1.

This experiment systematically tests whether Assembly Calculus shows the same
depth limitation:

  A. PP depth 0 -> test at depth 0 (within) and depth 1 (cross)
  B. No relative clauses -> test SRC (cross-depth for embedding)
  C. Full grammar -> compare within-depth vs cross-depth (positive control)

The key question: does local Hebbian learning create depth-specific pathways,
or do the learned patterns generalize to unseen depths?

Hypotheses:
  H1: Within-depth N400/P600 d > 2.0 (strong effects at trained depth)
  H2: Cross-depth prediction degrades (cross N400 d < within N400 d)
  H3: Binding (P600) generalizes better than prediction (N400) across depths

Usage:
    uv run python research/experiments/primitives/test_depth_generalization.py
    uv run python research/experiments/primitives/test_depth_generalization.py --quick
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
class DepthGenConfig:
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
    n_train_sentences: int = 80
    training_reps: int = 3
    n_test_items: int = 5


def _make_brain(cfg, vocab, seed):
    """Create a fresh initialized brain."""
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    return create_language_brain(bcfg, vocab, seed)


def test_pp_depth(brain, lexicon, vocab, cfg, depth, role_area):
    """Measure N400 and P600 at a PP-object position at given depth.

    Constructs context: agent verb patient prep0 [pp_obj0 prep1]* ...
    and measures at the PP object position for the target depth.
    """
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    locs = vocab.words_for_category("LOCATION")
    preps = vocab.words_for_category("PREP")
    ni = cfg.n_test_items

    n400_gram, n400_cv = [], []
    p600_gram, p600_cv = [], []

    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 1) % len(nouns)]

        # Build context up to the target depth
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)

        for d in range(depth + 1):
            prep = preps[(i + d) % len(preps)]
            activate_word(brain, prep, "PREP_CORE", 3)
            if d < depth:
                pp_obj = locs[(i + d) % len(locs)]
                activate_word(brain, pp_obj, "NOUN_CORE", 3)

        # Predict at current depth
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"PREP_CORE": ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        gram_pp = locs[(i + depth) % len(locs)]
        cv_pp = verbs[(i + 2) % len(verbs)]

        n400_gram.append(measure_n400(predicted, lexicon[gram_pp]))
        n400_cv.append(measure_n400(predicted, lexicon[cv_pp]))

        p600_gram.append(measure_p600(
            brain, gram_pp, "NOUN_CORE", role_area, cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_pp, "VERB_CORE", role_area, cfg.n_settling_rounds))

    return {
        "n400_gram": n400_gram, "n400_cv": n400_cv,
        "p600_gram": p600_gram, "p600_cv": p600_cv,
    }


def test_rc_cross_depth(brain, lexicon, vocab, cfg):
    """Test relative clause processing when trained only on simple SVO.

    Construct SRC context: agent "that" rel_verb rel_patient agent
    Measure N400 at main verb and P600 for dual binding.
    """
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    ni = cfg.n_test_items

    # N400 at main verb (after relative clause)
    n400_main_verb = []
    # P600 for dual binding: agent in ROLE_REL_AGENT (untrained)
    p600_rel_agent = []
    # P600 for agent in ROLE_AGENT (trained from SVO)
    p600_agent = []

    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # SRC context: agent "that" rel_verb rel_patient -> predict main_verb
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        n400_main_verb.append(measure_n400(predicted, lexicon[main_verb]))

        # P600 for dual binding
        p600_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))
        p600_rel_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))

    return {
        "n400_main_verb": n400_main_verb,
        "p600_agent": p600_agent,
        "p600_rel_agent": p600_rel_agent,
    }


def test_svo_baseline(brain, lexicon, vocab, cfg):
    """Measure baseline SVO N400 and P600 (within-depth positive control)."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    ni = cfg.n_test_items

    n400_gram, n400_cv = [], []
    p600_gram, p600_cv = [], []

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

        p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    return {
        "n400_gram": n400_gram, "n400_cv": n400_cv,
        "p600_gram": p600_gram, "p600_cv": p600_cv,
    }


def run_trial(
    cfg: DepthGenConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one depth generalization trial with three sub-experiments."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB
    n_train = cfg.n_train_sentences * cfg.training_reps

    results = {}

    # ── Sub-experiment A: PP depth 0 only, test at depth 0 and 1 ──────
    brain_a = _make_brain(cfg, vocab, seed)
    grammar_a = RecursiveCFG(
        pp_prob=0.5, recursive_pp_prob=0.0, max_pp_depth=1,
        rel_prob=0.0, vocab=vocab, rng=rng)
    sents_a = grammar_a.generate_batch(n_train)
    for s in sents_a:
        train_sentence(brain_a, s, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain_a.disable_plasticity = True
    lex_a = build_lexicon(brain_a, vocab, cfg.lexicon_readout_rounds)

    # Within-depth: PP depth 0
    pp0_within = test_pp_depth(brain_a, lex_a, vocab, cfg, 0, "ROLE_PP_OBJ")
    # Cross-depth: PP depth 1 (untrained)
    pp1_cross = test_pp_depth(brain_a, lex_a, vocab, cfg, 1, "ROLE_PP_OBJ_1")
    # SVO baseline
    svo_a = test_svo_baseline(brain_a, lex_a, vocab, cfg)

    results["A_pp_depth"] = {
        "svo_baseline": svo_a,
        "pp0_within": pp0_within,
        "pp1_cross": pp1_cross,
    }

    # ── Sub-experiment B: No relative clauses, test SRC ───────────────
    brain_b = _make_brain(cfg, vocab, seed + 1000)
    grammar_b = RecursiveCFG(
        pp_prob=0.3, recursive_pp_prob=0.0, max_pp_depth=1,
        rel_prob=0.0, vocab=vocab, rng=np.random.default_rng(seed + 1000))
    sents_b = grammar_b.generate_batch(n_train)
    for s in sents_b:
        train_sentence(brain_b, s, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain_b.disable_plasticity = True
    lex_b = build_lexicon(brain_b, vocab, cfg.lexicon_readout_rounds)

    rc_cross = test_rc_cross_depth(brain_b, lex_b, vocab, cfg)
    svo_b = test_svo_baseline(brain_b, lex_b, vocab, cfg)

    results["B_rc_depth"] = {
        "svo_baseline": svo_b,
        "rc_cross": rc_cross,
    }

    # ── Sub-experiment C: Full grammar (positive control) ─────────────
    brain_c = _make_brain(cfg, vocab, seed + 2000)
    grammar_c = RecursiveCFG(
        pp_prob=0.5, recursive_pp_prob=0.5, max_pp_depth=2,
        rel_prob=0.4, orc_prob=0.3,
        vocab=vocab, rng=np.random.default_rng(seed + 2000))
    sents_c = grammar_c.generate_batch(n_train)
    for s in sents_c:
        train_sentence(brain_c, s, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain_c.disable_plasticity = True
    lex_c = build_lexicon(brain_c, vocab, cfg.lexicon_readout_rounds)

    pp0_full = test_pp_depth(brain_c, lex_c, vocab, cfg, 0, "ROLE_PP_OBJ")
    pp1_full = test_pp_depth(brain_c, lex_c, vocab, cfg, 1, "ROLE_PP_OBJ_1")
    svo_c = test_svo_baseline(brain_c, lex_c, vocab, cfg)
    rc_full = test_rc_cross_depth(brain_c, lex_c, vocab, cfg)

    results["C_full"] = {
        "svo_baseline": svo_c,
        "pp0_within": pp0_full,
        "pp1_within": pp1_full,
        "rc_within": rc_full,
    }

    return results


class DepthGeneralizationExperiment(ExperimentBase):
    """Systematic depth generalization experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="depth_generalization",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[DepthGenConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or DepthGenConfig(
            **{k: v for k, v in kwargs.items()
               if k in DepthGenConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Depth Generalization: Transfer Across Recursion Depths")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_test_items={cfg.n_test_items}, n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect paired values across seeds
        # Sub-exp A
        a_svo_n400_gram, a_svo_n400_cv = [], []
        a_svo_p600_gram, a_svo_p600_cv = [], []
        a_pp0_n400_gram, a_pp0_n400_cv = [], []
        a_pp0_p600_gram, a_pp0_p600_cv = [], []
        a_pp1_n400_gram, a_pp1_n400_cv = [], []
        a_pp1_p600_gram, a_pp1_p600_cv = [], []
        # Sub-exp B
        b_svo_n400_gram, b_svo_n400_cv = [], []
        b_rc_n400_main = []
        b_rc_p600_agent, b_rc_p600_rel_agent = [], []
        # Sub-exp C
        c_pp0_n400_gram, c_pp0_n400_cv = [], []
        c_pp0_p600_gram, c_pp0_p600_cv = [], []
        c_pp1_n400_gram, c_pp1_n400_cv = [], []
        c_pp1_p600_gram, c_pp1_p600_cv = [], []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            # A: PP depth
            a = trial["A_pp_depth"]
            a_svo_n400_gram.append(float(np.mean(a["svo_baseline"]["n400_gram"])))
            a_svo_n400_cv.append(float(np.mean(a["svo_baseline"]["n400_cv"])))
            a_svo_p600_gram.append(float(np.mean(a["svo_baseline"]["p600_gram"])))
            a_svo_p600_cv.append(float(np.mean(a["svo_baseline"]["p600_cv"])))
            a_pp0_n400_gram.append(float(np.mean(a["pp0_within"]["n400_gram"])))
            a_pp0_n400_cv.append(float(np.mean(a["pp0_within"]["n400_cv"])))
            a_pp0_p600_gram.append(float(np.mean(a["pp0_within"]["p600_gram"])))
            a_pp0_p600_cv.append(float(np.mean(a["pp0_within"]["p600_cv"])))
            a_pp1_n400_gram.append(float(np.mean(a["pp1_cross"]["n400_gram"])))
            a_pp1_n400_cv.append(float(np.mean(a["pp1_cross"]["n400_cv"])))
            a_pp1_p600_gram.append(float(np.mean(a["pp1_cross"]["p600_gram"])))
            a_pp1_p600_cv.append(float(np.mean(a["pp1_cross"]["p600_cv"])))

            # B: RC depth
            b = trial["B_rc_depth"]
            b_svo_n400_gram.append(float(np.mean(b["svo_baseline"]["n400_gram"])))
            b_svo_n400_cv.append(float(np.mean(b["svo_baseline"]["n400_cv"])))
            b_rc_n400_main.append(float(np.mean(b["rc_cross"]["n400_main_verb"])))
            b_rc_p600_agent.append(float(np.mean(b["rc_cross"]["p600_agent"])))
            b_rc_p600_rel_agent.append(float(np.mean(b["rc_cross"]["p600_rel_agent"])))

            # C: Full
            c = trial["C_full"]
            c_pp0_n400_gram.append(float(np.mean(c["pp0_within"]["n400_gram"])))
            c_pp0_n400_cv.append(float(np.mean(c["pp0_within"]["n400_cv"])))
            c_pp0_p600_gram.append(float(np.mean(c["pp0_within"]["p600_gram"])))
            c_pp0_p600_cv.append(float(np.mean(c["pp0_within"]["p600_cv"])))
            c_pp1_n400_gram.append(float(np.mean(c["pp1_within"]["n400_gram"])))
            c_pp1_n400_cv.append(float(np.mean(c["pp1_within"]["n400_cv"])))
            c_pp1_p600_gram.append(float(np.mean(c["pp1_within"]["p600_gram"])))
            c_pp1_p600_cv.append(float(np.mean(c["pp1_within"]["p600_cv"])))

        # Compute effect sizes
        a_svo_n400 = paired_ttest(a_svo_n400_gram, a_svo_n400_cv)
        a_svo_p600 = paired_ttest(a_svo_p600_gram, a_svo_p600_cv)
        a_pp0_n400 = paired_ttest(a_pp0_n400_gram, a_pp0_n400_cv)
        a_pp0_p600 = paired_ttest(a_pp0_p600_gram, a_pp0_p600_cv)
        a_pp1_n400 = paired_ttest(a_pp1_n400_gram, a_pp1_n400_cv)
        a_pp1_p600 = paired_ttest(a_pp1_p600_gram, a_pp1_p600_cv)

        b_svo_n400 = paired_ttest(b_svo_n400_gram, b_svo_n400_cv)

        c_pp0_n400 = paired_ttest(c_pp0_n400_gram, c_pp0_n400_cv)
        c_pp0_p600 = paired_ttest(c_pp0_p600_gram, c_pp0_p600_cv)
        c_pp1_n400 = paired_ttest(c_pp1_n400_gram, c_pp1_n400_cv)
        c_pp1_p600 = paired_ttest(c_pp1_p600_gram, c_pp1_p600_cv)

        # Report
        self.log(f"\n  === Sub-experiment A: PP Depth 0 -> Test Depth 1 ===")
        self.log(f"    SVO baseline   N400 d={a_svo_n400['d']:.2f}  "
                 f"P600 d={a_svo_p600['d']:.2f}")
        self.log(f"    PP depth 0     N400 d={a_pp0_n400['d']:.2f}  "
                 f"P600 d={a_pp0_p600['d']:.2f}  (WITHIN)")
        self.log(f"    PP depth 1     N400 d={a_pp1_n400['d']:.2f}  "
                 f"P600 d={a_pp1_p600['d']:.2f}  (CROSS)")

        self.log(f"\n  === Sub-experiment B: No RC -> Test SRC ===")
        self.log(f"    SVO baseline   N400 d={b_svo_n400['d']:.2f}")
        self.log(f"    Main verb N400 after RC: {np.mean(b_rc_n400_main):.4f}")
        self.log(f"    Agent P600 (ROLE_AGENT):     {np.mean(b_rc_p600_agent):.4f}  (trained)")
        self.log(f"    Agent P600 (ROLE_REL_AGENT): {np.mean(b_rc_p600_rel_agent):.4f}  (untrained)")

        self.log(f"\n  === Sub-experiment C: Full Grammar (Positive Control) ===")
        self.log(f"    PP depth 0     N400 d={c_pp0_n400['d']:.2f}  "
                 f"P600 d={c_pp0_p600['d']:.2f}")
        self.log(f"    PP depth 1     N400 d={c_pp1_n400['d']:.2f}  "
                 f"P600 d={c_pp1_p600['d']:.2f}")

        # Hypotheses (use absolute d — gram < cv gives negative d)
        # H1: Within-depth effects are strong (|d| > 2.0)
        within_n400_d = a_pp0_n400["d"]
        within_p600_d = a_pp0_p600["d"]
        h1 = abs(within_n400_d) > 2.0 or abs(within_p600_d) > 2.0

        # H2: Cross-depth prediction degrades (|cross| < |within|)
        cross_n400_d = a_pp1_n400["d"]
        h2 = abs(cross_n400_d) < abs(within_n400_d)

        # H3: Binding generalizes better than prediction across depth
        # (cross P600 loses less effect size than cross N400)
        cross_p600_d = a_pp1_p600["d"]
        within_p600_for_h3 = abs(within_p600_d)
        cross_p600_for_h3 = abs(cross_p600_d)
        within_n400_for_h3 = abs(within_n400_d)
        cross_n400_for_h3 = abs(cross_n400_d)
        # Ratio of retained effect: cross/within (higher = better generalization)
        p600_retention = cross_p600_for_h3 / max(within_p600_for_h3, 0.01)
        n400_retention = cross_n400_for_h3 / max(within_n400_for_h3, 0.01)
        h3 = p600_retention > n400_retention

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Within-depth |d| > 2.0):       "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" (N400 |d|={abs(within_n400_d):.2f},"
                 f" P600 |d|={abs(within_p600_d):.2f})")
        self.log(f"    H2 (Cross |d| < within |d|):       "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" (cross |d|={abs(cross_n400_d):.2f}"
                 f" vs within |d|={abs(within_n400_d):.2f})")
        self.log(f"    H3 (P600 retains > N400):          "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (P600 retention={p600_retention:.2f},"
                 f" N400 retention={n400_retention:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "A_pp_depth": {
                "svo_n400": a_svo_n400, "svo_p600": a_svo_p600,
                "pp0_within_n400": a_pp0_n400, "pp0_within_p600": a_pp0_p600,
                "pp1_cross_n400": a_pp1_n400, "pp1_cross_p600": a_pp1_p600,
            },
            "B_rc_depth": {
                "svo_n400": b_svo_n400,
                "rc_main_verb_n400": summarize(b_rc_n400_main),
                "rc_p600_agent": summarize(b_rc_p600_agent),
                "rc_p600_rel_agent": summarize(b_rc_p600_rel_agent),
            },
            "C_full": {
                "pp0_n400": c_pp0_n400, "pp0_p600": c_pp0_p600,
                "pp1_n400": c_pp1_n400, "pp1_p600": c_pp1_p600,
            },
            "hypotheses": {
                "H1_within_depth_strong": h1,
                "H2_cross_depth_degrades": h2,
                "H3_p600_generalizes_better": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_train_sentences": cfg.n_train_sentences,
                "training_reps": cfg.training_reps,
                "n_test_items": cfg.n_test_items,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Depth Generalization Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = DepthGeneralizationExperiment(verbose=True)

    if args.quick:
        cfg = DepthGenConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2,
            n_test_items=4)
        n_seeds = args.seeds or 3
    else:
        cfg = DepthGenConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("DEPTH GENERALIZATION SUMMARY")
    print("=" * 70)
    print(f"\nH1 Within-depth strong:      {'PASS' if h['H1_within_depth_strong'] else 'FAIL'}")
    print(f"H2 Cross-depth degrades:     {'PASS' if h['H2_cross_depth_degrades'] else 'FAIL'}")
    print(f"H3 P600 generalizes better:  {'PASS' if h['H3_p600_generalizes_better'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
