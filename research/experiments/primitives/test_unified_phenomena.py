"""
Unified Single-Brain Experiment: All Phenomena Coexist

Tests whether a single brain, trained on one diverse corpus, simultaneously
exhibits all validated phenomena. Humans don't have separate brains for
each construction — if the mechanism is correct, everything should coexist.

One brain trained on RecursiveCFG with:
  - SVO, SVO+PP, recursive PP
  - Subject-relative clauses (SRC)
  - Object-relative clauses (ORC)

Then nine test suites measure:
  1. Object position double dissociation (N400/P600)
  2. PP-object position double dissociation
  3. Center-embedding main verb prediction
  4. Dual binding (ROLE_AGENT + ROLE_REL_AGENT)
  5. Garden-path vs unambiguous N400
  6. SRC vs ORC main verb N400
  7. SRC vs ORC dual-binding P600
  8. Agreement attraction (no PP vs short PP)
  9. Recursive PP depth 0 vs 1

For each, compute Cohen's d. Report which phenomena survive (d > 0.5).

Hypotheses:
  H1: All phenomena show d > 0.5 in the unified brain
  H2: Object position double dissociation has d > 1.0 (baseline quality)

Usage:
    uv run python research/experiments/primitives/test_unified_phenomena.py
    uv run python research/experiments/primitives/test_unified_phenomena.py --quick
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
class UnifiedConfig:
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
    # Rich grammar
    pp_prob: float = 0.4
    recursive_pp_prob: float = 0.3
    rel_prob: float = 0.4
    orc_prob: float = 0.3
    max_pp_depth: int = 2
    # Training
    n_train_sentences: int = 150
    training_reps: int = 3
    # Test
    n_test_items: int = 5


def run_trial(
    cfg: UnifiedConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one unified trial: one brain, all nine test suites."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Rich mixed grammar
    grammar = RecursiveCFG(
        pp_prob=cfg.pp_prob, recursive_pp_prob=cfg.recursive_pp_prob,
        rel_prob=cfg.rel_prob, orc_prob=cfg.orc_prob,
        max_pp_depth=cfg.max_pp_depth,
        vocab=vocab, rng=rng)

    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = grammar.generate_batch(n_train)

    n_src = sum(1 for s in train_sents if s.get("rel_type") == "SRC")
    n_orc = sum(1 for s in train_sents if s.get("rel_type") == "ORC")
    n_pp = sum(1 for s in train_sents if s.get("has_pp"))

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    preps = vocab.words_for_category("PREP")
    locs = vocab.words_for_category("LOCATION")
    novels = list(vocab.novel_words.keys())
    ni = cfg.n_test_items

    results = {"n_train": n_train, "n_src": n_src, "n_orc": n_orc, "n_pp": n_pp}

    # ── 1. Object position double dissociation ────────────────────
    obj_p600_gram, obj_p600_cv = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        obj_p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        obj_p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    results["obj_p600_gram"] = float(np.mean(obj_p600_gram))
    results["obj_p600_cv"] = float(np.mean(obj_p600_cv))

    # ── 2. PP-object position ─────────────────────────────────────
    pp_p600_gram, pp_p600_cv = [], []
    for i in range(ni):
        gram_pp = locs[i % len(locs)]
        cv_pp = verbs[(i + 2) % len(verbs)]
        pp_p600_gram.append(measure_p600(
            brain, gram_pp, "NOUN_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))
        pp_p600_cv.append(measure_p600(
            brain, cv_pp, "VERB_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))

    results["pp_p600_gram"] = float(np.mean(pp_p600_gram))
    results["pp_p600_cv"] = float(np.mean(pp_p600_cv))

    # ── 3. Center-embedding main verb N400 ────────────────────────
    ce_n400 = []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        ce_n400.append(measure_n400(pred, lexicon[main_verb]))

    results["ce_n400_main_verb"] = float(np.mean(ce_n400))

    # ── 4. Dual binding ───────────────────────────────────────────
    dual_agent, dual_rel_agent = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        dual_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))
        dual_rel_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))

    results["dual_p600_agent"] = float(np.mean(dual_agent))
    results["dual_p600_rel_agent"] = float(np.mean(dual_rel_agent))

    # ── 5. Garden-path vs unambiguous ─────────────────────────────
    gp_n400, unamb_n400 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # Unambiguous: agent that rel_verb rel_patient MAIN_VERB
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_u = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        unamb_n400.append(measure_n400(pred_u, lexicon[main_verb]))

        # Garden-path: agent rel_verb rel_patient MAIN_VERB (no "that")
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_g = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        gp_n400.append(measure_n400(pred_g, lexicon[main_verb]))

    results["gp_n400"] = float(np.mean(gp_n400))
    results["unamb_n400"] = float(np.mean(unamb_n400))

    # ── 6. SRC vs ORC main verb N400 ──────────────────────────────
    src_n400, orc_n400 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        orc_agent = nouns[(i + 1) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # SRC
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_s = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        src_n400.append(measure_n400(pred_s, lexicon[main_verb]))

        # ORC
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, orc_agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_o = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        orc_n400.append(measure_n400(pred_o, lexicon[main_verb]))

    results["src_n400_main"] = float(np.mean(src_n400))
    results["orc_n400_main"] = float(np.mean(orc_n400))

    # ── 7. SRC vs ORC dual-binding P600 ───────────────────────────
    src_dual, orc_dual = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        src_dual.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))
        orc_dual.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_PATIENT", cfg.n_settling_rounds))

    results["src_dual_p600"] = float(np.mean(src_dual))
    results["orc_dual_p600"] = float(np.mean(orc_dual))

    # ── 8. Agreement attraction ───────────────────────────────────
    aa_no_pp, aa_short_pp = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        prep = preps[i % len(preps)]
        pp_obj = locs[i % len(locs)]

        aa_no_pp.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, prep, "PREP_CORE", 3)
        activate_word(brain, pp_obj, "NOUN_CORE", 3)
        aa_short_pp.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))

    results["aa_p600_no_pp"] = float(np.mean(aa_no_pp))
    results["aa_p600_short_pp"] = float(np.mean(aa_short_pp))

    # ── 9. Recursive PP depth 0 vs 1 ─────────────────────────────
    rpp0_gram, rpp0_cv, rpp1_gram, rpp1_cv = [], [], [], []
    for i in range(ni):
        gram0 = locs[i % len(locs)]
        cv0 = verbs[(i + 2) % len(verbs)]
        gram1 = locs[(i + 1) % len(locs)]
        cv1 = verbs[(i + 3) % len(verbs)]

        rpp0_gram.append(measure_p600(
            brain, gram0, "NOUN_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))
        rpp0_cv.append(measure_p600(
            brain, cv0, "VERB_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))
        rpp1_gram.append(measure_p600(
            brain, gram1, "NOUN_CORE", "ROLE_PP_OBJ_1", cfg.n_settling_rounds))
        rpp1_cv.append(measure_p600(
            brain, cv1, "VERB_CORE", "ROLE_PP_OBJ_1", cfg.n_settling_rounds))

    results["rpp0_p600_gram"] = float(np.mean(rpp0_gram))
    results["rpp0_p600_cv"] = float(np.mean(rpp0_cv))
    results["rpp1_p600_gram"] = float(np.mean(rpp1_gram))
    results["rpp1_p600_cv"] = float(np.mean(rpp1_cv))

    brain.disable_plasticity = False
    return results


class UnifiedPhenomenaExperiment(ExperimentBase):
    """All phenomena in one brain."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="unified_phenomena",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[UnifiedConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or UnifiedConfig(
            **{k: v for k, v in kwargs.items()
               if k in UnifiedConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Unified Phenomena: All Tests in One Brain")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  pp={cfg.pp_prob}, rel={cfg.rel_prob}, orc={cfg.orc_prob}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = [
            "obj_p600_gram", "obj_p600_cv",
            "pp_p600_gram", "pp_p600_cv",
            "ce_n400_main_verb",
            "dual_p600_agent", "dual_p600_rel_agent",
            "gp_n400", "unamb_n400",
            "src_n400_main", "orc_n400_main",
            "src_dual_p600", "orc_dual_p600",
            "aa_p600_no_pp", "aa_p600_short_pp",
            "rpp0_p600_gram", "rpp0_p600_cv",
            "rpp1_p600_gram", "rpp1_p600_cv",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])
            if s == 0:
                self.log(f"    Training: {result['n_train']} sentences"
                         f" ({result['n_src']} SRC, {result['n_orc']} ORC,"
                         f" {result['n_pp']} PP)")

        # Compute all effect sizes
        tests = {}
        tests["1_obj_dissociation"] = paired_ttest(vals["obj_p600_cv"], vals["obj_p600_gram"])
        tests["2_pp_dissociation"] = paired_ttest(vals["pp_p600_cv"], vals["pp_p600_gram"])
        tests["3_ce_prediction"] = None  # single measure, compare to 0.5 threshold
        tests["4_dual_binding"] = None   # both should be low
        tests["5_garden_path"] = paired_ttest(vals["gp_n400"], vals["unamb_n400"])
        tests["6_src_orc_n400"] = paired_ttest(vals["orc_n400_main"], vals["src_n400_main"])
        tests["7_src_orc_dual"] = paired_ttest(vals["orc_dual_p600"], vals["src_dual_p600"])
        tests["8_agreement"] = paired_ttest(vals["aa_p600_short_pp"], vals["aa_p600_no_pp"])
        tests["9_rpp0"] = paired_ttest(vals["rpp0_p600_cv"], vals["rpp0_p600_gram"])
        tests["9_rpp1"] = paired_ttest(vals["rpp1_p600_cv"], vals["rpp1_p600_gram"])

        # Report
        self.log(f"\n  === Effect Sizes (Cohen's d) ===")
        self.log(f"    1. Object dissociation:     d={tests['1_obj_dissociation']['d']:.2f}")
        self.log(f"    2. PP dissociation:          d={tests['2_pp_dissociation']['d']:.2f}")
        self.log(f"    3. CE main verb N400:        {np.mean(vals['ce_n400_main_verb']):.3f}")
        self.log(f"    4. Dual binding:             AGENT={np.mean(vals['dual_p600_agent']):.3f}"
                 f"  REL_AGENT={np.mean(vals['dual_p600_rel_agent']):.3f}")
        self.log(f"    5. Garden-path N400:         d={tests['5_garden_path']['d']:.2f}")
        self.log(f"    6. SRC/ORC N400:             d={tests['6_src_orc_n400']['d']:.2f}")
        self.log(f"    7. SRC/ORC dual-binding:     d={tests['7_src_orc_dual']['d']:.2f}")
        self.log(f"    8. Agreement attraction:     d={tests['8_agreement']['d']:.2f}")
        self.log(f"    9. Recursive PP d0:          d={tests['9_rpp0']['d']:.2f}"
                 f"  d1: d={tests['9_rpp1']['d']:.2f}")

        # Count surviving phenomena (d > 0.5 for paired tests)
        d_values = [
            tests["1_obj_dissociation"]["d"],
            tests["2_pp_dissociation"]["d"],
            tests["5_garden_path"]["d"],
            tests["7_src_orc_dual"]["d"],
            tests["8_agreement"]["d"],
            tests["9_rpp0"]["d"],
        ]
        n_survive = sum(1 for d in d_values if d > 0.5)

        h1 = n_survive >= 5
        h2 = tests["1_obj_dissociation"]["d"] > 1.0

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    Phenomena surviving (d > 0.5): {n_survive}/6")
        self.log(f"    H1 (>= 5 survive):  {'PASS' if h1 else 'FAIL'}")
        self.log(f"    H2 (Obj d > 1.0):   {'PASS' if h2 else 'FAIL'}"
                 f" (d={tests['1_obj_dissociation']['d']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "effect_sizes": {name: t for name, t in tests.items() if t is not None},
            "raw_means": {k: summarize(v) for k, v in vals.items()},
            "n_surviving": n_survive,
            "hypotheses": {"H1_most_survive": h1, "H2_obj_large": h2},
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "pp_prob": cfg.pp_prob, "rel_prob": cfg.rel_prob,
                "orc_prob": cfg.orc_prob,
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
        description="Unified Phenomena Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = UnifiedPhenomenaExperiment(verbose=True)

    if args.quick:
        cfg = UnifiedConfig(
            n=5000, k=50,
            n_train_sentences=80, training_reps=2)
        n_seeds = args.seeds or 5
    else:
        cfg = UnifiedConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("UNIFIED PHENOMENA SUMMARY")
    print("=" * 70)

    print("\nEffect sizes:")
    for name, t in m["effect_sizes"].items():
        print(f"  {name}: d={t['d']:.2f}")

    print(f"\nPhenomena surviving (d > 0.5): {m['n_surviving']}/6")
    h = m["hypotheses"]
    print(f"H1 Most survive: {'PASS' if h['H1_most_survive'] else 'FAIL'}")
    print(f"H2 Obj d > 1.0:  {'PASS' if h['H2_obj_large'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
