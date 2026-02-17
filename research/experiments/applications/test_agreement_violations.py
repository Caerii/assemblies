"""
Agreement Violations — Graded P600 for Morphosyntactic Mismatch

Prediction 3.1 from IMPLICATIONS_AND_PREDICTIONS.md:

Tests whether P600 structural instability is graded:
  category violation > agreement violation > grammatical

RESULT (Feb 2026): NEGATIVE — the predicted ordering does NOT hold.
Actual ordering: P600(cat) > P600(gram) > P600(agree), d=-7.1, p=0.006.
Agreement violations produce LOWER instability than grammatical sentences.

==========================================================================
MECHANISTIC ANALYSIS OF THE NEGATIVE RESULT
==========================================================================

The failure traces to a CONTEXT CONTAMINATION effect in the core area,
not a failure of agreement detection per se. Here is the causal chain:

1. Context processing projects each word's phonological stimulus into its
   core area (test_p600_syntactic.py:146-150). After context ["the",
   "dog"/"dogs", "chases", "the"], NOUN_CORE retains a residual assembly
   from the subject ("dog" or "dogs"), because nothing else was projected
   into NOUN_CORE after it.

2. The critical word "cat" is then projected into NOUN_CORE with
   self-recurrence. The residual from the context subject INTERFERES with
   the incoming "cat" assembly. The amount of interference is proportional
   to the OVERLAP between the residual and the incoming assembly.

3. "dog" (SG) and "cat" (SG) share features: ANIMAL + SG.
   "dogs" (PL) and "cat" (SG) share features: ANIMAL only (differ on SG/PL).
   Therefore the "dog" residual has MORE overlap with "cat" than the "dogs"
   residual does.

4. More overlap → more context contamination → "cat" assembly in NOUN_CORE
   is more DIFFUSE (pulled toward the "dog" assembly). A diffuse core
   assembly projects less consistently into structural areas → HIGHER
   instability.

5. Less overlap ("dogs" context) → less contamination → "cat" assembly
   converges more cleanly → LOWER instability.

The P600 instability metric is therefore measuring a CONTEXT SIMILARITY
effect at the core-area level, not the predicted STRUCTURAL INTEGRATION
difficulty from agreement mismatch.

This is a genuine architectural property: the N400 mechanism (shared
features → lower energy → facilitation) and the P600 mechanism (shared
features → more diffuse assembly → higher instability) have OPPOSITE
relationships with context similarity. Facilitation helps semantic access
but hurts structural precision.

==========================================================================
DISAMBIGUATION EXPERIMENTS
==========================================================================

This file now runs 5 sub-experiments to test competing hypotheses:

  Exp A: Original object-position P600 (replication of negative result)
  Exp B: Verb-position P600 (measure at LOCUS of agreement violation)
  Exp C: Per-area P600 breakdown (which structural areas drive the effect)
  Exp D: N400 + core instability comparison (does agreement affect N400?)
  Exp E: Context-free baseline (P600 with no context, establishes floor)

Hypotheses being tested:

  H1 (Context Contamination): The effect is a core-area artifact. SG context
     contaminates the SG critical word more than PL context does, inflating
     grammatical P600 relative to agreement violation P600.
     Predictions: Verb-position should show agree > gram (if measured at
     the verb, "dogs" context differs from "dog" context differently).
     N400 should show agree > gram (PL context less facilitating for SG verb).

  H2 (Measurement Position): The agreement violation IS detected at the verb
     but the effect dissipates by the time we reach the object.
     Predictions: Verb-position P600 shows agree > gram. Object-position
     shows agree <= gram (because the effect has faded).

  H3 (Insufficient Feature Representation): SG/PL features encoded as
     grounding features don't create strong enough assembly differences to
     produce structural instability.
     Predictions: Verb-position P600 also shows agree <= gram. The SG/PL
     features don't produce measurable structural effects at any position.

  H4 (Area-Specific Effect): Agreement violations only affect specific
     structural areas (e.g., SUBJ/ROLE_AGENT) but the mean across all
     5 areas washes out the signal.
     Predictions: Per-area breakdown reveals some areas with agree > gram
     masked by others with agree < gram.

==========================================================================
DISAMBIGUATION RESULTS (Feb 2026, 3 seeds)
==========================================================================

Exp A (Object position): Replicates negative result.
  gram=1.566, agree=0.928, cat=1.645
  Context contamination confirmed (gram > agree, d=-7.1, p=0.006)

Exp B (Verb position): NO EFFECT at the locus of the violation.
  gram=0.979, agree=0.986, d=0.18, p=0.79
  Agreement mismatch does not create structural instability at any
  measurement position. This rules out H2 (position effect).

Exp C (Per-area breakdown):
  Object position: SUBJ and OBJ drive the effect (d=-10.9, -11.5).
  ROLE_AGENT, ROLE_PATIENT, VP are near zero.
  Verb position: Mixed directions, mostly non-significant.
  ROLE_AGENT shows marginal agree>gram (d=2.0, p=0.07).

Exp D (N400 + core instability): THE SMOKING GUN.
  Object N400: agree >> gram (d=88). PL context provides much less
    feature overlap with SG object -> higher N400 energy (less facilitation).
  Object core instability: agree << gram (d=-21). Less contamination from
    PL context -> cleaner core assembly -> lower Jaccard instability.
  This confirms the context contamination mechanism end-to-end.

Exp E (Context-free baseline): CONFIRMS CONTAMINATION DIRECTION.
  gram_with_ctx vs no_ctx: d=13.1, p=0.002 -> SG context ADDS instability
  agree_with_ctx vs no_ctx: d=0.78, p=0.31 -> PL context has NO effect
  The "grammatical advantage" is actually a "contamination penalty": SG
  subject residual in NOUN_CORE interferes with SG object assembly,
  while PL subject residual does not.

Hypothesis discrimination:
  H1 (Context Contamination): SUPPORTED
  H2 (Position Effect): NOT SUPPORTED
  H3 (Insufficient Representation): SUPPORTED
  H4 (Area-Specific Masking): SUPPORTED (verb-position only)

CONCLUSION: The Assembly Calculus model cannot produce P600 effects for
agreement violations because:

1. Number features (SG/PL) encoded as grounding features affect core-area
   assemblies but do NOT create different consolidation patterns in
   structural areas. Both SG and PL nouns get the same NOUN_CORE ->
   ROLE_AGENT consolidation. There are no number-specific structural
   pathways.

2. The current P600 measurement at the object position is confounded by
   context contamination: the SG/PL difference in the subject affects how
   the object assembles in NOUN_CORE, creating a spurious effect in the
   OPPOSITE direction from what was predicted.

3. To actually model agreement violations, the architecture would need:
   (a) An agreement-checking mechanism (e.g., inter-area inhibition
       between number-marked assemblies), OR
   (b) Number-specific structural pathways (SG nouns and PL nouns
       consolidated into different role sub-areas), OR
   (c) A mismatch detection mechanism that fires when number features
       in ROLE_AGENT conflict with number features in VP.

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
from typing import Dict, List, Any

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.infrastructure import (
    bootstrap_structural_connectivity,
    consolidate_role_connections,
    consolidate_vp_connections,
)
from research.experiments.applications.test_p600_syntactic import (
    _measure_critical_word,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP,
)


@dataclass
class AgreementConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 5
    consolidation_passes: int = 10


def _build_agreement_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary with singular/plural noun and verb forms.

    Number is encoded as a grounding feature (SG/PL) so that singular
    and plural forms of the same word share most features but differ
    in number marking.

    Feature structure per word:
      "dog"    → visual: [DOG, ANIMAL, SG]  → NOUN_CORE
      "dogs"   → visual: [DOG, ANIMAL, PL]  → NOUN_CORE
      "chases" → motor:  [CHASING, PURSUIT, SG] → VERB_CORE
      "chase"  → motor:  [CHASING, PURSUIT, PL] → VERB_CORE

    Each feature becomes a separate stimulus (e.g., visual_SG, visual_PL).
    During train_lexicon(), phon + grounding features are projected
    simultaneously, so assemblies encode both identity and number.

    "dog" and "dogs" share 2/3 grounding features (DOG, ANIMAL) but differ
    on SG vs PL. This creates overlapping but distinct assemblies in
    NOUN_CORE. The degree of overlap is what drives the context
    contamination effect observed in the negative result.
    """
    return {
        # Singular nouns
        "dog":    GroundingContext(visual=["DOG", "ANIMAL", "SG"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL", "SG"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL", "SG"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL", "SG"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL", "SG"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL", "SG"]),
        # Plural nouns
        "dogs":   GroundingContext(visual=["DOG", "ANIMAL", "PL"]),
        "cats":   GroundingContext(visual=["CAT", "ANIMAL", "PL"]),
        "birds":  GroundingContext(visual=["BIRD", "ANIMAL", "PL"]),
        "horses": GroundingContext(visual=["HORSE", "ANIMAL", "PL"]),
        # Untrained objects (for category violation baseline)
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        # Singular verbs (3sg)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT", "SG"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION", "SG"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION", "SG"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION", "SG"]),
        # Plural/bare verbs
        "chase":  GroundingContext(motor=["CHASING", "PURSUIT", "PL"]),
        "see":    GroundingContext(motor=["SEEING", "PERCEPTION", "PL"]),
        "find":   GroundingContext(motor=["FINDING", "PERCEPTION", "PL"]),
        "like":   GroundingContext(motor=["LIKING", "EMOTION", "PL"]),
        # Function word
        "the":    GroundingContext(),
    }


def _build_agreement_training(vocab):
    """Training on ONLY agreeing sentences (sg+sg, pl+pl).

    Establishes Hebbian connections for grammatical number agreement.
    The parser never sees disagreeing combinations during training.

    This creates the consolidation asymmetry:
    - SG_noun → ROLE_AGENT: strengthened (trained)
    - PL_noun → ROLE_AGENT: strengthened (trained)
    - VERB_CORE → VP: strengthened for both SG and PL verbs
    - But SG_noun + PL_verb combination was never seen in any role/VP path
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # Singular subject + singular verb (3x repetition for strength)
    sg_triples = [
        ("dog", "chases", "cat"), ("cat", "sees", "bird"),
        ("bird", "chases", "fish"), ("horse", "chases", "dog"),
        ("dog", "sees", "bird"), ("cat", "finds", "horse"),
    ]
    # Plural subject + plural verb
    pl_triples = [
        ("dogs", "chase", "cats"), ("cats", "see", "birds"),
        ("birds", "chase", "dogs"), ("horses", "chase", "cats"),
        ("dogs", "see", "birds"), ("cats", "find", "horses"),
    ]

    for triples in [sg_triples, pl_triples]:
        for _ in range(3):
            for subj, verb, obj in triples:
                sentences.append(GroundedSentence(
                    words=["the", subj, verb, "the", obj],
                    contexts=[ctx("the"), ctx(subj), ctx(verb),
                              ctx("the"), ctx(obj)],
                    roles=[None, "agent", "action", None, "patient"],
                ))

    return sentences


# =========================================================================
# Exp A: Object-position test (replicates original negative result)
# =========================================================================
# Measure P600 at object position ("cat") after different contexts.
# The agreement violation occurred earlier (at the verb), and by the
# object position we measure cumulative difficulty.
#
# Context contamination hypothesis: "dog" (SG) residual in NOUN_CORE
# has more feature overlap with "cat" (SG) than "dogs" (PL) does.
# More overlap → more diffuse "cat" assembly → higher instability.
# This predicts: gram > agree (OPPOSITE of the original prediction).
OBJECT_POSITION_TESTS = [
    {
        "label": "dog_chases_cat",
        "sg_context": ["the", "dog", "chases", "the"],
        "pl_context": ["the", "dogs", "chases", "the"],
        "grammatical_obj": "cat",
        "category_violation": "likes",
    },
    {
        "label": "cat_sees_bird",
        "sg_context": ["the", "cat", "sees", "the"],
        "pl_context": ["the", "cats", "sees", "the"],
        "grammatical_obj": "bird",
        "category_violation": "finds",
    },
    {
        "label": "bird_chases_fish",
        "sg_context": ["the", "bird", "chases", "the"],
        "pl_context": ["the", "birds", "chases", "the"],
        "grammatical_obj": "fish",
        "category_violation": "sees",
    },
    {
        "label": "horse_finds_mouse",
        "sg_context": ["the", "horse", "finds", "the"],
        "pl_context": ["the", "horses", "finds", "the"],
        "grammatical_obj": "mouse",
        "category_violation": "chases",
    },
]

# =========================================================================
# Exp B: Verb-position test (measure at the LOCUS of agreement violation)
# =========================================================================
# Context: ["the", subj], critical word: verb
# In the agreement violation, the PL subject assembly is in NOUN_CORE when
# the SG verb is projected into VERB_CORE. Since they're in DIFFERENT core
# areas, the context contamination effect should be minimal.
#
# If H1 (context contamination) is correct: verb-position measurement
# removes the contamination artifact, and we may see the predicted pattern.
#
# If H3 (insufficient representation) is correct: no effect at verb
# position either, because SG/PL features don't propagate to VP instability.
VERB_POSITION_TESTS = [
    {
        "label": "dogs_chases",
        "sg_context": ["the", "dog"],
        "pl_context": ["the", "dogs"],
        "verb": "chases",
    },
    {
        "label": "cats_sees",
        "sg_context": ["the", "cat"],
        "pl_context": ["the", "cats"],
        "verb": "sees",
    },
    {
        "label": "birds_chases",
        "sg_context": ["the", "bird"],
        "pl_context": ["the", "birds"],
        "verb": "chases",
    },
    {
        "label": "horses_finds",
        "sg_context": ["the", "horse"],
        "pl_context": ["the", "horses"],
        "verb": "finds",
    },
]

# =========================================================================
# Exp E: Context-free baseline
# =========================================================================
# No context at all — project the critical word directly and measure P600.
# This establishes the floor: any P600 differences in Exp A/B above this
# floor are context-driven.
CONTEXT_FREE_WORDS = [
    ("cat", "noun"), ("bird", "noun"), ("fish", "noun"), ("mouse", "noun"),
    ("chases", "verb"), ("sees", "verb"), ("finds", "verb"), ("likes", "verb"),
]


class AgreementViolationExperiment(ExperimentBase):
    """Test graded P600 for agreement violations with disambiguation.

    Runs 5 sub-experiments:
      A: Object-position P600 (replication of negative result)
      B: Verb-position P600 (at locus of violation)
      C: Per-area breakdown (included in A and B measurements)
      D: N400 + core instability (from _measure_critical_word output)
      E: Context-free baseline

    All sub-experiments share the same trained parser per seed, so they
    probe the same network under different measurement conditions.
    """

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="agreement_violations",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = AgreementConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_agreement_vocab()
        training = _build_agreement_training(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        # === Per-seed accumulators ===

        # Exp A: Object-position P600
        obj_gram_seeds = []
        obj_agree_seeds = []
        obj_cat_seeds = []

        # Exp A per-area: {area: [per_seed_means]}
        obj_gram_per_area = {a: [] for a in p600_areas}
        obj_agree_per_area = {a: [] for a in p600_areas}
        obj_cat_per_area = {a: [] for a in p600_areas}

        # Exp B: Verb-position P600
        verb_gram_seeds = []
        verb_agree_seeds = []

        # Exp B per-area
        verb_gram_per_area = {a: [] for a in p600_areas}
        verb_agree_per_area = {a: [] for a in p600_areas}

        # Exp D: N400 + core instability at object position
        n400_gram_seeds = []
        n400_agree_seeds = []
        n400_cat_seeds = []
        core_inst_gram_seeds = []
        core_inst_agree_seeds = []
        core_inst_cat_seeds = []

        # Exp D: N400 + core instability at verb position
        n400_verb_gram_seeds = []
        n400_verb_agree_seeds = []
        core_inst_verb_gram_seeds = []
        core_inst_verb_agree_seeds = []

        # Exp E: Context-free baseline
        ctxfree_noun_seeds = []
        ctxfree_verb_seeds = []

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            bootstrap_structural_connectivity(
                parser, p600_areas, log_fn=self.log)
            consolidate_role_connections(
                parser, training, n_passes=cfg.consolidation_passes,
                log_fn=self.log)
            consolidate_vp_connections(
                parser, training, n_passes=cfg.consolidation_passes,
                log_fn=self.log)

            # --- Exp A: Object-position measurement ---
            self.log("  Exp A: Object-position P600")
            gram_vals, agree_vals, cat_vals = [], [], []
            gram_areas = {a: [] for a in p600_areas}
            agree_areas = {a: [] for a in p600_areas}
            cat_areas = {a: [] for a in p600_areas}
            n400_g, n400_a, n400_c = [], [], []
            ci_g, ci_a, ci_c = [], [], []

            for test in OBJECT_POSITION_TESTS:
                # Grammatical: singular context + trained noun
                result_gram = _measure_critical_word(
                    parser, test["sg_context"], test["grammatical_obj"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                gram_vals.append(result_gram["p600_mean_instability"])
                n400_g.append(result_gram["n400_energy"])
                ci_g.append(result_gram["core_instability"])
                for a in p600_areas:
                    gram_areas[a].append(
                        result_gram["p600_instability"].get(a, 0.0))

                # Agreement violation: plural subject context + trained noun
                result_agree = _measure_critical_word(
                    parser, test["pl_context"], test["grammatical_obj"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                agree_vals.append(result_agree["p600_mean_instability"])
                n400_a.append(result_agree["n400_energy"])
                ci_a.append(result_agree["core_instability"])
                for a in p600_areas:
                    agree_areas[a].append(
                        result_agree["p600_instability"].get(a, 0.0))

                # Category violation: singular context + verb as noun
                result_cat = _measure_critical_word(
                    parser, test["sg_context"], test["category_violation"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                cat_vals.append(result_cat["p600_mean_instability"])
                n400_c.append(result_cat["n400_energy"])
                ci_c.append(result_cat["core_instability"])
                for a in p600_areas:
                    cat_areas[a].append(
                        result_cat["p600_instability"].get(a, 0.0))

            if gram_vals:
                obj_gram_seeds.append(float(np.mean(gram_vals)))
                obj_agree_seeds.append(float(np.mean(agree_vals)))
                obj_cat_seeds.append(float(np.mean(cat_vals)))
                n400_gram_seeds.append(float(np.mean(n400_g)))
                n400_agree_seeds.append(float(np.mean(n400_a)))
                n400_cat_seeds.append(float(np.mean(n400_c)))
                core_inst_gram_seeds.append(float(np.mean(ci_g)))
                core_inst_agree_seeds.append(float(np.mean(ci_a)))
                core_inst_cat_seeds.append(float(np.mean(ci_c)))
                for a in p600_areas:
                    obj_gram_per_area[a].append(
                        float(np.mean(gram_areas[a])))
                    obj_agree_per_area[a].append(
                        float(np.mean(agree_areas[a])))
                    obj_cat_per_area[a].append(
                        float(np.mean(cat_areas[a])))
                self.log(f"    gram={np.mean(gram_vals):.3f}  "
                         f"agree={np.mean(agree_vals):.3f}  "
                         f"cat={np.mean(cat_vals):.3f}")

            # --- Exp B: Verb-position measurement ---
            self.log("  Exp B: Verb-position P600")
            vg_vals, va_vals = [], []
            vg_areas = {a: [] for a in p600_areas}
            va_areas = {a: [] for a in p600_areas}
            n400_vg, n400_va = [], []
            ci_vg, ci_va = [], []

            for test in VERB_POSITION_TESTS:
                # Grammatical: sg subject context + verb
                result_vg = _measure_critical_word(
                    parser, test["sg_context"], test["verb"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                vg_vals.append(result_vg["p600_mean_instability"])
                n400_vg.append(result_vg["n400_energy"])
                ci_vg.append(result_vg["core_instability"])
                for a in p600_areas:
                    vg_areas[a].append(
                        result_vg["p600_instability"].get(a, 0.0))

                # Agreement violation: pl subject context + sg verb
                result_va = _measure_critical_word(
                    parser, test["pl_context"], test["verb"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                va_vals.append(result_va["p600_mean_instability"])
                n400_va.append(result_va["n400_energy"])
                ci_va.append(result_va["core_instability"])
                for a in p600_areas:
                    va_areas[a].append(
                        result_va["p600_instability"].get(a, 0.0))

            if vg_vals:
                verb_gram_seeds.append(float(np.mean(vg_vals)))
                verb_agree_seeds.append(float(np.mean(va_vals)))
                n400_verb_gram_seeds.append(float(np.mean(n400_vg)))
                n400_verb_agree_seeds.append(float(np.mean(n400_va)))
                core_inst_verb_gram_seeds.append(float(np.mean(ci_vg)))
                core_inst_verb_agree_seeds.append(float(np.mean(ci_va)))
                for a in p600_areas:
                    verb_gram_per_area[a].append(
                        float(np.mean(vg_areas[a])))
                    verb_agree_per_area[a].append(
                        float(np.mean(va_areas[a])))
                self.log(f"    gram={np.mean(vg_vals):.3f}  "
                         f"agree={np.mean(va_vals):.3f}")

            # --- Exp E: Context-free baseline ---
            self.log("  Exp E: Context-free baseline")
            noun_vals, verb_vals = [], []

            for word, wtype in CONTEXT_FREE_WORDS:
                result_cf = _measure_critical_word(
                    parser, [], word,
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                if wtype == "noun":
                    noun_vals.append(result_cf["p600_mean_instability"])
                else:
                    verb_vals.append(result_cf["p600_mean_instability"])

            if noun_vals:
                ctxfree_noun_seeds.append(float(np.mean(noun_vals)))
                ctxfree_verb_seeds.append(float(np.mean(verb_vals)))
                self.log(f"    noun={np.mean(noun_vals):.3f}  "
                         f"verb={np.mean(verb_vals):.3f}")

        # =====================================================================
        # ANALYSIS
        # =====================================================================
        self.log(f"\n{'='*70}")
        self.log("AGREEMENT VIOLATION RESULTS — DISAMBIGUATION ANALYSIS")
        self.log(f"{'='*70}")

        metrics = {}

        # --- Exp A: Object-position P600 ---
        self.log(f"\n--- Exp A: Object-Position P600 ---")
        self.log("Original prediction: P600(cat) > P600(agree) > P600(gram)")
        self.log("Context contamination predicts: P600(gram) > P600(agree)")

        if len(obj_gram_seeds) >= 2:
            gs = summarize(obj_gram_seeds)
            ags = summarize(obj_agree_seeds)
            cs = summarize(obj_cat_seeds)

            self.log(f"\n  Grammatical:         "
                     f"{gs['mean']:.4f} +/- {gs['sem']:.4f}")
            self.log(f"  Agreement violation: "
                     f"{ags['mean']:.4f} +/- {ags['sem']:.4f}")
            self.log(f"  Category violation:  "
                     f"{cs['mean']:.4f} +/- {cs['sem']:.4f}")

            metrics["obj_gram"] = gs
            metrics["obj_agreement"] = ags
            metrics["obj_category"] = cs

            ordering_correct = cs["mean"] > ags["mean"] > gs["mean"]
            contamination_pattern = gs["mean"] > ags["mean"]
            self.log(f"\n  Original ordering (cat>agree>gram): "
                     f"{'YES' if ordering_correct else 'NO'}")
            self.log(f"  Context contamination (gram>agree): "
                     f"{'YES' if contamination_pattern else 'NO'}")
            metrics["obj_ordering_correct"] = ordering_correct
            metrics["obj_contamination_pattern"] = contamination_pattern

            for label, a, b in [
                ("obj_cat_vs_gram", obj_cat_seeds, obj_gram_seeds),
                ("obj_agree_vs_gram", obj_agree_seeds, obj_gram_seeds),
                ("obj_cat_vs_agree", obj_cat_seeds, obj_agree_seeds),
            ]:
                stats = paired_ttest(a, b)
                direction = ("HIGHER" if np.mean(a) > np.mean(b)
                             else "LOWER")
                self.log(f"    {label:<25}: d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                metrics[label] = {"test": stats, "direction": direction}

        # --- Exp B: Verb-position P600 ---
        self.log(f"\n--- Exp B: Verb-Position P600 ---")
        self.log("H1 predicts: agree > gram (contamination removed)")
        self.log("H3 predicts: agree ~ gram (features too weak)")

        if len(verb_gram_seeds) >= 2:
            vgs = summarize(verb_gram_seeds)
            vas = summarize(verb_agree_seeds)

            self.log(f"\n  Grammatical (sg subj + sg verb): "
                     f"{vgs['mean']:.4f} +/- {vgs['sem']:.4f}")
            self.log(f"  Agreement viol (pl subj + sg verb): "
                     f"{vas['mean']:.4f} +/- {vas['sem']:.4f}")

            metrics["verb_gram"] = vgs
            metrics["verb_agreement"] = vas

            stats_verb = paired_ttest(verb_agree_seeds, verb_gram_seeds)
            verb_agree_higher = np.mean(verb_agree_seeds) > np.mean(verb_gram_seeds)
            self.log(f"\n  agree > gram: "
                     f"{'YES' if verb_agree_higher else 'NO'}  "
                     f"d={stats_verb['d']:.3f}  p={stats_verb['p']:.4f}")
            metrics["verb_agree_vs_gram"] = {
                "test": stats_verb,
                "agree_higher": verb_agree_higher,
            }

            # Hypothesis discrimination
            if verb_agree_higher and stats_verb["p"] < 0.1:
                self.log("  => Supports H1 (context contamination) or "
                         "H2 (position effect)")
            else:
                self.log("  => Supports H3 (insufficient representation)")

        # --- Exp C: Per-area P600 breakdown ---
        self.log(f"\n--- Exp C: Per-Area P600 Breakdown ---")
        self.log("H4 predicts: some areas show agree > gram, others don't")

        self.log(f"\n  Object-position (agree vs gram):")
        for a in p600_areas:
            if (len(obj_agree_per_area[a]) >= 2 and
                    len(obj_gram_per_area[a]) >= 2):
                stats = paired_ttest(
                    obj_agree_per_area[a], obj_gram_per_area[a])
                ag_mean = np.mean(obj_agree_per_area[a])
                gr_mean = np.mean(obj_gram_per_area[a])
                direction = "agree>gram" if ag_mean > gr_mean else "gram>agree"
                self.log(f"    {a:<15}: agree={ag_mean:.4f}  "
                         f"gram={gr_mean:.4f}  d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                metrics[f"obj_per_area_{a}"] = {
                    "agree": float(ag_mean), "gram": float(gr_mean),
                    "test": stats, "direction": direction,
                }

        self.log(f"\n  Verb-position (agree vs gram):")
        for a in p600_areas:
            if (len(verb_agree_per_area[a]) >= 2 and
                    len(verb_gram_per_area[a]) >= 2):
                stats = paired_ttest(
                    verb_agree_per_area[a], verb_gram_per_area[a])
                ag_mean = np.mean(verb_agree_per_area[a])
                gr_mean = np.mean(verb_gram_per_area[a])
                direction = "agree>gram" if ag_mean > gr_mean else "gram>agree"
                self.log(f"    {a:<15}: agree={ag_mean:.4f}  "
                         f"gram={gr_mean:.4f}  d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                metrics[f"verb_per_area_{a}"] = {
                    "agree": float(ag_mean), "gram": float(gr_mean),
                    "test": stats, "direction": direction,
                }

        # --- Exp D: N400 + core instability ---
        self.log(f"\n--- Exp D: N400 + Core Instability ---")
        self.log("If agreement affects N400 but not P600: core-area effect")
        self.log("If agreement affects P600 but not N400: structural effect")

        self.log(f"\n  Object-position N400:")
        if len(n400_gram_seeds) >= 2:
            for label, a, b in [
                ("agree_vs_gram", n400_agree_seeds, n400_gram_seeds),
                ("cat_vs_gram", n400_cat_seeds, n400_gram_seeds),
            ]:
                stats = paired_ttest(a, b)
                self.log(f"    {label}: a={np.mean(a):.1f}  "
                         f"b={np.mean(b):.1f}  d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}")
                metrics[f"n400_obj_{label}"] = {"test": stats}

        self.log(f"\n  Object-position core instability:")
        if len(core_inst_gram_seeds) >= 2:
            for label, a, b in [
                ("agree_vs_gram",
                 core_inst_agree_seeds, core_inst_gram_seeds),
                ("cat_vs_gram",
                 core_inst_cat_seeds, core_inst_gram_seeds),
            ]:
                stats = paired_ttest(a, b)
                self.log(f"    {label}: a={np.mean(a):.4f}  "
                         f"b={np.mean(b):.4f}  d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}")
                metrics[f"core_inst_obj_{label}"] = {"test": stats}

        self.log(f"\n  Verb-position N400:")
        if len(n400_verb_gram_seeds) >= 2:
            stats = paired_ttest(n400_verb_agree_seeds, n400_verb_gram_seeds)
            self.log(f"    agree_vs_gram: a={np.mean(n400_verb_agree_seeds):.1f}  "
                     f"b={np.mean(n400_verb_gram_seeds):.1f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}")
            metrics["n400_verb_agree_vs_gram"] = {"test": stats}

        self.log(f"\n  Verb-position core instability:")
        if len(core_inst_verb_gram_seeds) >= 2:
            stats = paired_ttest(
                core_inst_verb_agree_seeds, core_inst_verb_gram_seeds)
            self.log(f"    agree_vs_gram: "
                     f"a={np.mean(core_inst_verb_agree_seeds):.4f}  "
                     f"b={np.mean(core_inst_verb_gram_seeds):.4f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}")
            metrics["core_inst_verb_agree_vs_gram"] = {"test": stats}

        # --- Exp E: Context-free baseline ---
        self.log(f"\n--- Exp E: Context-Free Baseline ---")
        self.log("Establishes P600 floor without any context effects")

        if len(ctxfree_noun_seeds) >= 2:
            ns = summarize(ctxfree_noun_seeds)
            vs = summarize(ctxfree_verb_seeds)
            self.log(f"  Nouns (no context): "
                     f"{ns['mean']:.4f} +/- {ns['sem']:.4f}")
            self.log(f"  Verbs (no context): "
                     f"{vs['mean']:.4f} +/- {vs['sem']:.4f}")
            metrics["ctxfree_noun"] = ns
            metrics["ctxfree_verb"] = vs

            # Compare to with-context conditions
            if len(obj_gram_seeds) >= 2:
                stats = paired_ttest(obj_gram_seeds, ctxfree_noun_seeds)
                self.log(f"  gram_with_ctx vs no_ctx: "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics["context_effect_gram"] = {"test": stats}

                stats = paired_ttest(obj_agree_seeds, ctxfree_noun_seeds)
                self.log(f"  agree_with_ctx vs no_ctx: "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics["context_effect_agree"] = {"test": stats}

        # --- Hypothesis Summary ---
        self.log(f"\n{'='*70}")
        self.log("HYPOTHESIS DISCRIMINATION SUMMARY")
        self.log(f"{'='*70}")

        h1_supported = False
        h2_supported = False
        h3_supported = False
        h4_supported = False

        # H1: Context contamination
        if ("obj_agree_vs_gram" in metrics and
                "verb_agree_vs_gram" in metrics):
            obj_d = metrics["obj_agree_vs_gram"]["test"]["d"]
            verb_result = metrics["verb_agree_vs_gram"]
            verb_d = verb_result["test"]["d"]
            verb_higher = verb_result.get("agree_higher", False)

            # H1 confirmed if: obj shows gram>agree AND verb shows agree>gram
            h1_supported = (obj_d < 0 and verb_higher and verb_d > 0)
            self.log(f"\n  H1 (Context Contamination): "
                     f"{'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
            self.log(f"    Object-position d={obj_d:.3f} "
                     f"(need <0: {'OK' if obj_d < 0 else 'FAIL'})")
            self.log(f"    Verb-position d={verb_d:.3f} "
                     f"(need >0: {'OK' if verb_d > 0 else 'FAIL'})")

        # H2: Measurement position effect
        if "verb_agree_vs_gram" in metrics:
            verb_d = metrics["verb_agree_vs_gram"]["test"]["d"]
            verb_p = metrics["verb_agree_vs_gram"]["test"]["p"]
            h2_supported = (verb_d > 0 and verb_p < 0.1 and
                            "obj_agree_vs_gram" in metrics and
                            metrics["obj_agree_vs_gram"]["test"]["d"] < 0)
            self.log(f"\n  H2 (Position Effect): "
                     f"{'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")
            self.log(f"    Requires: verb agree>gram (d>0) AND "
                     f"obj agree<gram (d<0)")

        # H3: Insufficient representation
        if "verb_agree_vs_gram" in metrics:
            verb_d = metrics["verb_agree_vs_gram"]["test"]["d"]
            verb_p = metrics["verb_agree_vs_gram"]["test"]["p"]
            h3_supported = abs(verb_d) < 1.0 or verb_p > 0.2
            self.log(f"\n  H3 (Insufficient Representation): "
                     f"{'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}")
            self.log(f"    Requires: verb-position effect weak "
                     f"(|d|<1.0 or p>0.2)")
            self.log(f"    Verb d={verb_d:.3f}, p={verb_p:.4f}")

        # H4: Area-specific masking
        area_directions = []
        for a in p600_areas:
            key = f"verb_per_area_{a}"
            if key in metrics:
                area_directions.append(
                    (a, metrics[key]["direction"],
                     metrics[key]["test"]["d"]))
        if area_directions:
            agree_higher_areas = [a for a, d, _ in area_directions
                                  if d == "agree>gram"]
            gram_higher_areas = [a for a, d, _ in area_directions
                                 if d == "gram>agree"]
            h4_supported = (len(agree_higher_areas) > 0 and
                            len(gram_higher_areas) > 0)
            self.log(f"\n  H4 (Area-Specific Masking): "
                     f"{'SUPPORTED' if h4_supported else 'NOT SUPPORTED'}")
            self.log(f"    agree>gram in: {agree_higher_areas}")
            self.log(f"    gram>agree in: {gram_higher_areas}")

        metrics["hypotheses"] = {
            "H1_context_contamination": h1_supported,
            "H2_position_effect": h2_supported,
            "H3_insufficient_representation": h3_supported,
            "H4_area_specific_masking": h4_supported,
        }

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
        description="Agreement Violation Experiment (Prediction 3.1)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = AgreementViolationExperiment()
    exp.run(quick=args.quick)
