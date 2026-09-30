"""
Recursive Structure: PP Depth and Center-Embedding

Tests whether Assembly Calculus can learn recursive context-free structure
from distributional evidence. Two phenomena:

1. Recursive PP: "dog chases cat in garden on hill"
   Can the prediction chain extend to arbitrary PP depth?
   Does P600 distinguish trained vs untrained bindings at each depth?

2. Center-embedding (subject-relative clauses):
   "dog that chases cat sees bird"
   Can the brain maintain the main-clause agent ("dog") across an
   intervening embedded clause and predict the main verb ("sees")?
   Does P600 correctly measure integration difficulty at both clause levels?

Center-embedding is the formal property that distinguishes context-free
languages from regular languages. If AC handles it, the mechanism can in
principle learn any context-free structure.

Architecture (11 areas, vocabulary-driven):
  NOUN_CORE, VERB_CORE, PREP_CORE, COMP_CORE  -- lexical assemblies
  PREDICTION                                     -- forward projection
  ROLE_AGENT, ROLE_PATIENT                       -- main clause roles
  ROLE_PP_OBJ, ROLE_PP_OBJ_1                    -- PP depth 0 and 1
  ROLE_REL_AGENT, ROLE_REL_PATIENT              -- relative clause roles

Grammar (RecursiveCFG):
  S  -> NP VP
  NP -> N | N "that" VP
  VP -> V NP (PP)?
  PP -> P NP (PP)?

Hypotheses:
  H1 (Recursive PP): N400/P600 double dissociation holds at PP depth 1
      (same pattern as depth 0)
  H2 (Center-embedding prediction): After "dog that chases cat", the brain
      can predict the main verb via the persisting agent assembly. N400
      at main verb position is lower for trained verbs.
  H3 (Dual binding): "dog" binds to both ROLE_AGENT (main) and
      ROLE_REL_AGENT (relative). P600 is low for both because noun->role
      pathways are trained at both clause levels.
  H4 (Cross-clause violation): A verb in the main-clause patient position
      after a relative clause shows high P600 (untrained binding to
      ROLE_PATIENT), demonstrating that clause-level role bindings are
      maintained across embedded material.

Usage:
    uv run python research/experiments/primitives/test_recursive_structure.py
    uv run python research/experiments/primitives/test_recursive_structure.py --quick
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
class RecursiveConfig:
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
    # Grammar parameters
    pp_prob: float = 0.5
    recursive_pp_prob: float = 0.5
    rel_prob: float = 0.4
    max_pp_depth: int = 2
    # Training
    n_train_sentences: int = 80
    training_reps: int = 3


def run_trial(
    cfg: RecursiveConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one recursive structure trial.

    Trains on RecursiveCFG-generated sentences with recursive PPs and
    relative clauses, then tests:
      - N400/P600 at main object position
      - N400/P600 at PP-object positions (depth 0 and 1)
      - N400/P600 at main verb position after relative clause
      - P600 for dual-binding (main agent + relative agent)
    """
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    # Create brain
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Generate training sentences
    grammar = RecursiveCFG(
        pp_prob=cfg.pp_prob,
        recursive_pp_prob=cfg.recursive_pp_prob,
        rel_prob=cfg.rel_prob,
        max_pp_depth=cfg.max_pp_depth,
        vocab=vocab, rng=rng)

    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = grammar.generate_batch(n_train)

    # Track what structures appeared in training
    n_rel = sum(1 for s in train_sents if s.get("has_rel"))
    n_pp = sum(1 for s in train_sents if s.get("has_pp"))
    n_deep_pp = sum(1 for s in train_sents if s.get("pp_depth", 0) >= 2)

    # Train
    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    # Build lexicon
    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    locs = vocab.words_for_category("LOCATION")
    novels = list(vocab.novel_words.keys())

    # ── Test 1: Main object position (same as composed ERP) ──────────
    obj_n400_gram, obj_n400_cv, obj_n400_novel = [], [], []
    obj_p600_gram, obj_p600_cv, obj_p600_novel = [], [], []

    for i in range(5):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]
        novel_obj = novels[i % len(novels)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        obj_n400_gram.append(measure_n400(predicted, lexicon[gram_obj]))
        obj_n400_cv.append(measure_n400(predicted, lexicon[cv_obj]))
        obj_n400_novel.append(measure_n400(predicted, lexicon[novel_obj]))

        obj_p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        obj_p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        obj_p600_novel.append(measure_p600(
            brain, novel_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    # ── Test 2: PP-object at depth 0 and depth 1 ─────────────────────
    pp0_n400_gram, pp0_n400_cv = [], []
    pp0_p600_gram, pp0_p600_cv = [], []
    pp1_n400_gram, pp1_n400_cv = [], []
    pp1_p600_gram, pp1_p600_cv = [], []

    preps = vocab.words_for_category("PREP")

    for i in range(5):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 1) % len(nouns)]
        prep0 = preps[i % len(preps)]
        gram_pp0 = locs[i % len(locs)]
        cv_pp0 = verbs[(i + 2) % len(verbs)]

        # Context: SVO + P (depth 0)
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)
        activate_word(brain, prep0, "PREP_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"PREP_CORE": ["PREDICTION"]})
        pred_pp0 = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        pp0_n400_gram.append(measure_n400(pred_pp0, lexicon[gram_pp0]))
        pp0_n400_cv.append(measure_n400(pred_pp0, lexicon[cv_pp0]))
        pp0_p600_gram.append(measure_p600(
            brain, gram_pp0, "NOUN_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))
        pp0_p600_cv.append(measure_p600(
            brain, cv_pp0, "VERB_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))

        # Context: SVO + P + NP + P (depth 1)
        prep1 = preps[(i + 1) % len(preps)]
        gram_pp1 = locs[(i + 1) % len(locs)]
        cv_pp1 = verbs[(i + 3) % len(verbs)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)
        activate_word(brain, prep0, "PREP_CORE", 3)
        activate_word(brain, gram_pp0, "NOUN_CORE", 3)
        activate_word(brain, prep1, "PREP_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"PREP_CORE": ["PREDICTION"]})
        pred_pp1 = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        pp1_n400_gram.append(measure_n400(pred_pp1, lexicon[gram_pp1]))
        pp1_n400_cv.append(measure_n400(pred_pp1, lexicon[cv_pp1]))
        pp1_p600_gram.append(measure_p600(
            brain, gram_pp1, "NOUN_CORE", "ROLE_PP_OBJ_1", cfg.n_settling_rounds))
        pp1_p600_cv.append(measure_p600(
            brain, cv_pp1, "VERB_CORE", "ROLE_PP_OBJ_1", cfg.n_settling_rounds))

    # ── Test 3: Center-embedding — main verb after relative clause ───
    # "dog that chases cat SEES bird"
    # The question: can the brain predict "sees" after "dog that chases cat"?
    # And does P600 work at the main-clause patient position?
    rel_n400_main_verb = []
    rel_p600_main_patient_gram = []
    rel_p600_main_patient_cv = []
    rel_p600_rel_patient_gram = []
    rel_p600_rel_patient_cv = []
    rel_dual_p600_agent = []      # P600 for agent in ROLE_AGENT
    rel_dual_p600_rel_agent = []  # P600 for agent in ROLE_REL_AGENT

    for i in range(5):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]
        main_patient_gram = nouns[(i + 3) % len(nouns)]
        main_patient_cv = verbs[(i + 2) % len(verbs)]

        # Process: agent -> "that" -> rel_verb -> rel_patient
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)

        # Now predict main verb: the agent's assembly in NOUN_CORE
        # should have been activated at the start and prediction bridges
        # should allow NOUN_CORE -> PREDICTION to evoke the main verb.
        # Re-activate agent to simulate the persisting main-clause context.
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_main_verb = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        rel_n400_main_verb.append(
            measure_n400(pred_main_verb, lexicon[main_verb]))

        # Process main verb and predict main object
        activate_word(brain, main_verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        pred_main_obj = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # P600 at main-clause patient position
        rel_p600_main_patient_gram.append(measure_p600(
            brain, main_patient_gram, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))
        rel_p600_main_patient_cv.append(measure_p600(
            brain, main_patient_cv, "VERB_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

        # P600 at relative-clause patient position
        rel_p600_rel_patient_gram.append(measure_p600(
            brain, rel_patient, "NOUN_CORE", "ROLE_REL_PATIENT",
            cfg.n_settling_rounds))
        rel_p600_rel_patient_cv.append(measure_p600(
            brain, main_patient_cv, "VERB_CORE", "ROLE_REL_PATIENT",
            cfg.n_settling_rounds))

        # Dual binding: agent -> ROLE_AGENT and ROLE_REL_AGENT
        rel_dual_p600_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT",
            cfg.n_settling_rounds))
        rel_dual_p600_rel_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT",
            cfg.n_settling_rounds))

    brain.disable_plasticity = False

    return {
        # Training stats
        "n_train": n_train,
        "n_rel": n_rel,
        "n_pp": n_pp,
        "n_deep_pp": n_deep_pp,
        # Object position
        "obj_n400_gram": float(np.mean(obj_n400_gram)),
        "obj_n400_cv": float(np.mean(obj_n400_cv)),
        "obj_n400_novel": float(np.mean(obj_n400_novel)),
        "obj_p600_gram": float(np.mean(obj_p600_gram)),
        "obj_p600_cv": float(np.mean(obj_p600_cv)),
        "obj_p600_novel": float(np.mean(obj_p600_novel)),
        # PP depth 0
        "pp0_n400_gram": float(np.mean(pp0_n400_gram)),
        "pp0_n400_cv": float(np.mean(pp0_n400_cv)),
        "pp0_p600_gram": float(np.mean(pp0_p600_gram)),
        "pp0_p600_cv": float(np.mean(pp0_p600_cv)),
        # PP depth 1
        "pp1_n400_gram": float(np.mean(pp1_n400_gram)),
        "pp1_n400_cv": float(np.mean(pp1_n400_cv)),
        "pp1_p600_gram": float(np.mean(pp1_p600_gram)),
        "pp1_p600_cv": float(np.mean(pp1_p600_cv)),
        # Center-embedding
        "rel_n400_main_verb": float(np.mean(rel_n400_main_verb)),
        "rel_p600_main_patient_gram": float(np.mean(rel_p600_main_patient_gram)),
        "rel_p600_main_patient_cv": float(np.mean(rel_p600_main_patient_cv)),
        "rel_p600_rel_patient_gram": float(np.mean(rel_p600_rel_patient_gram)),
        "rel_p600_rel_patient_cv": float(np.mean(rel_p600_rel_patient_cv)),
        "rel_dual_p600_agent": float(np.mean(rel_dual_p600_agent)),
        "rel_dual_p600_rel_agent": float(np.mean(rel_dual_p600_rel_agent)),
    }


class RecursiveStructureExperiment(ExperimentBase):
    """Recursive PP depth and center-embedding experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="recursive_structure",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[RecursiveConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or RecursiveConfig(
            **{k: v for k, v in kwargs.items()
               if k in RecursiveConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Recursive Structure: PP Depth + Center-Embedding")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  pp_prob={cfg.pp_prob}, recursive_pp={cfg.recursive_pp_prob},"
                 f" rel_prob={cfg.rel_prob}")
        self.log(f"  n_train_sentences={cfg.n_train_sentences},"
                 f" training_reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-seed values
        keys = [
            "obj_n400_gram", "obj_n400_cv", "obj_n400_novel",
            "obj_p600_gram", "obj_p600_cv", "obj_p600_novel",
            "pp0_n400_gram", "pp0_n400_cv", "pp0_p600_gram", "pp0_p600_cv",
            "pp1_n400_gram", "pp1_n400_cv", "pp1_p600_gram", "pp1_p600_cv",
            "rel_n400_main_verb",
            "rel_p600_main_patient_gram", "rel_p600_main_patient_cv",
            "rel_p600_rel_patient_gram", "rel_p600_rel_patient_cv",
            "rel_dual_p600_agent", "rel_dual_p600_rel_agent",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])

            if s == 0:
                self.log(f"    Training: {result['n_train']} sentences"
                         f" ({result['n_rel']} rel, {result['n_pp']} PP,"
                         f" {result['n_deep_pp']} deep PP)")

        # ── Report: Object position ──
        self.log(f"\n  === Object Position (baseline) ===")
        self.log(f"    N400 — g: {np.mean(vals['obj_n400_gram']):.3f}"
                 f"  cv: {np.mean(vals['obj_n400_cv']):.3f}"
                 f"  novel: {np.mean(vals['obj_n400_novel']):.3f}")
        self.log(f"    P600 — g: {np.mean(vals['obj_p600_gram']):.3f}"
                 f"  cv: {np.mean(vals['obj_p600_cv']):.3f}"
                 f"  novel: {np.mean(vals['obj_p600_novel']):.3f}")

        t_obj_p600 = paired_ttest(vals["obj_p600_cv"], vals["obj_p600_gram"])
        self.log(f"    P600 CatViol>Gram: d={t_obj_p600['d']:.2f}")

        # ── Report: PP depth 0 vs depth 1 ──
        t_pp0 = paired_ttest(vals["pp0_p600_cv"], vals["pp0_p600_gram"])
        t_pp1 = paired_ttest(vals["pp1_p600_cv"], vals["pp1_p600_gram"])

        self.log(f"\n  === Recursive PP ===")
        self.log(f"    Depth 0 — N400 g: {np.mean(vals['pp0_n400_gram']):.3f}"
                 f"  cv: {np.mean(vals['pp0_n400_cv']):.3f}"
                 f"  | P600 g: {np.mean(vals['pp0_p600_gram']):.3f}"
                 f"  cv: {np.mean(vals['pp0_p600_cv']):.3f}"
                 f"  d={t_pp0['d']:.2f}")
        self.log(f"    Depth 1 — N400 g: {np.mean(vals['pp1_n400_gram']):.3f}"
                 f"  cv: {np.mean(vals['pp1_n400_cv']):.3f}"
                 f"  | P600 g: {np.mean(vals['pp1_p600_gram']):.3f}"
                 f"  cv: {np.mean(vals['pp1_p600_cv']):.3f}"
                 f"  d={t_pp1['d']:.2f}")

        # ── Report: Center-embedding ──
        t_rel_main = paired_ttest(
            vals["rel_p600_main_patient_cv"], vals["rel_p600_main_patient_gram"])
        t_rel_rel = paired_ttest(
            vals["rel_p600_rel_patient_cv"], vals["rel_p600_rel_patient_gram"])

        self.log(f"\n  === Center-Embedding ===")
        self.log(f"    Main verb N400 (after rel clause): "
                 f"{np.mean(vals['rel_n400_main_verb']):.3f}")
        self.log(f"    Main patient P600 — g: "
                 f"{np.mean(vals['rel_p600_main_patient_gram']):.3f}"
                 f"  cv: {np.mean(vals['rel_p600_main_patient_cv']):.3f}"
                 f"  d={t_rel_main['d']:.2f}")
        self.log(f"    Rel patient P600  — g: "
                 f"{np.mean(vals['rel_p600_rel_patient_gram']):.3f}"
                 f"  cv: {np.mean(vals['rel_p600_rel_patient_cv']):.3f}"
                 f"  d={t_rel_rel['d']:.2f}")
        self.log(f"    Dual binding P600 — ROLE_AGENT: "
                 f"{np.mean(vals['rel_dual_p600_agent']):.3f}"
                 f"  ROLE_REL_AGENT: "
                 f"{np.mean(vals['rel_dual_p600_rel_agent']):.3f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "object_position": {
                "n400": {c: summarize(vals[f"obj_n400_{c}"]) for c in ["gram", "cv", "novel"]},
                "p600": {c: summarize(vals[f"obj_p600_{c}"]) for c in ["gram", "cv", "novel"]},
                "p600_test": t_obj_p600,
            },
            "recursive_pp": {
                "depth_0": {
                    "n400_gram": summarize(vals["pp0_n400_gram"]),
                    "n400_cv": summarize(vals["pp0_n400_cv"]),
                    "p600_gram": summarize(vals["pp0_p600_gram"]),
                    "p600_cv": summarize(vals["pp0_p600_cv"]),
                    "p600_test": t_pp0,
                },
                "depth_1": {
                    "n400_gram": summarize(vals["pp1_n400_gram"]),
                    "n400_cv": summarize(vals["pp1_n400_cv"]),
                    "p600_gram": summarize(vals["pp1_p600_gram"]),
                    "p600_cv": summarize(vals["pp1_p600_cv"]),
                    "p600_test": t_pp1,
                },
            },
            "center_embedding": {
                "main_verb_n400": summarize(vals["rel_n400_main_verb"]),
                "main_patient_p600_gram": summarize(vals["rel_p600_main_patient_gram"]),
                "main_patient_p600_cv": summarize(vals["rel_p600_main_patient_cv"]),
                "main_patient_test": t_rel_main,
                "rel_patient_p600_gram": summarize(vals["rel_p600_rel_patient_gram"]),
                "rel_patient_p600_cv": summarize(vals["rel_p600_rel_patient_cv"]),
                "rel_patient_test": t_rel_rel,
                "dual_binding": {
                    "agent_p600": summarize(vals["rel_dual_p600_agent"]),
                    "rel_agent_p600": summarize(vals["rel_dual_p600_rel_agent"]),
                },
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "pp_prob": cfg.pp_prob,
                "recursive_pp_prob": cfg.recursive_pp_prob,
                "rel_prob": cfg.rel_prob,
                "max_pp_depth": cfg.max_pp_depth,
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
        description="Recursive Structure Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = RecursiveStructureExperiment(verbose=True)

    if args.quick:
        cfg = RecursiveConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2)
        n_seeds = args.seeds or 5
    else:
        cfg = RecursiveConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("RECURSIVE STRUCTURE SUMMARY")
    print("=" * 70)

    print("\nObject position (baseline):")
    op = m["object_position"]
    print(f"  N400 — g: {op['n400']['gram']['mean']:.3f}"
          f"  cv: {op['n400']['cv']['mean']:.3f}"
          f"  novel: {op['n400']['novel']['mean']:.3f}")
    print(f"  P600 — g: {op['p600']['gram']['mean']:.3f}"
          f"  cv: {op['p600']['cv']['mean']:.3f}"
          f"  novel: {op['p600']['novel']['mean']:.3f}")
    print(f"  P600 CatViol>Gram: d={op['p600_test']['d']:.2f}")

    print("\nRecursive PP:")
    for d in [0, 1]:
        pp = m["recursive_pp"][f"depth_{d}"]
        print(f"  Depth {d} — N400 g: {pp['n400_gram']['mean']:.3f}"
              f"  cv: {pp['n400_cv']['mean']:.3f}"
              f"  | P600 g: {pp['p600_gram']['mean']:.3f}"
              f"  cv: {pp['p600_cv']['mean']:.3f}"
              f"  d={pp['p600_test']['d']:.2f}")

    print("\nCenter-embedding:")
    ce = m["center_embedding"]
    print(f"  Main verb N400: {ce['main_verb_n400']['mean']:.3f}")
    print(f"  Main patient P600 — g: {ce['main_patient_p600_gram']['mean']:.3f}"
          f"  cv: {ce['main_patient_p600_cv']['mean']:.3f}"
          f"  d={ce['main_patient_test']['d']:.2f}")
    print(f"  Rel patient P600  — g: {ce['rel_patient_p600_gram']['mean']:.3f}"
          f"  cv: {ce['rel_patient_p600_cv']['mean']:.3f}"
          f"  d={ce['rel_patient_test']['d']:.2f}")
    db = ce["dual_binding"]
    print(f"  Dual binding — ROLE_AGENT: {db['agent_p600']['mean']:.3f}"
          f"  ROLE_REL_AGENT: {db['rel_agent_p600']['mean']:.3f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
