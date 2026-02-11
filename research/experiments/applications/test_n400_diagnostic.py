"""
N400 Priming Diagnostic Experiment

Deep investigation into why assembly calculus does NOT exhibit the N400
semantic priming effect, and what architectural changes would produce it.

Background — the anti-priming artifact:
    The original semantic_priming experiment reported "anti-priming":
    related=0.762, unrelated=0.924. But this was a measurement artifact.
    It compared DIFFERENT targets (cat vs table) that have different
    baseline self-reconstruction accuracies due to feature sharing:
    - Animals share visual_ANIMAL → entangled assemblies → baseline ~0.76
    - Objects have unique features → clean assemblies → baseline ~0.98

    When properly controlled (SAME target, different primes), there is
    NO priming and NO anti-priming. The prime has zero effect on the
    target's overlap with its lexicon entry.

The deeper problem — why no priming at all:
    k-WTA (winners-take-all) is BINARY: neurons are either in the winner
    set or out. There is no "partial pre-activation" that helps the target.
    In the brain, the N400 reflects GRADED activation: after processing
    "dog", related neurons have subthreshold pre-activation that helps
    "cat" reach threshold faster. k-WTA doesn't support this.

    However, there IS a tiny convergence speed effect at round 1:
    - No prime → cat round 1 overlap: 0.57
    - Dog prime → cat round 1 overlap: 0.63 (+0.06)
    This suggests the right N400 analogue is CONVERGENCE SPEED, not
    final assembly overlap.

This experiment systematically tests:

H1: Baseline asymmetry — Words sharing grounding features have lower
    self-reconstruction baselines, independent of any priming.

H2: Proper priming control — Same target with related vs unrelated
    prime shows no difference in final assembly overlap (confirming
    the artifact diagnosis).

H3: Convergence speed priming — The first-round overlap after a
    related prime is higher than after an unrelated prime, even though
    final overlaps are equal (the N400 analogue).

H4: Feature-level facilitation — Measuring overlap in the grounding
    INPUT areas (not core areas) shows that shared features ARE
    pre-activated by related primes.

H5: Semantic layer intervention — Adding a SEMANTIC area with larger
    k (where multiple concepts can coexist) produces genuine priming
    that matches the N400 pattern.

Statistical methodology:
- N_SEEDS independent random seeds.
- Within-subject design: same parser, same target, different primes.
- Paired t-tests for related vs unrelated conditions.
- Cohen's d effect sizes.

References:
- Kutas & Hillyard (1980). Reading senseless sentences.
- Kutas & Federmeier (2011). Thirty years and counting: N400.
- Meyer & Schvaneveldt (1971). Facilitation in recognizing pairs.
- Lau, Phillips & Poeppel (2008). A cortical network for semantics.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import NOUN_CORE, VERB_CORE, CORE_AREAS
from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.assembly import overlap as asm_overlap


@dataclass
class N400Config:
    """Configuration for the N400 diagnostic experiment."""
    n: int = 50000      # neurons per area (5x standard for finer resolution)
    k: int = 100        # assembly size (same as standard, sparser in larger n)
    n_seeds: int = 5    # independent random seeds
    p: float = 0.05     # connection probability (biologically realistic for cortex)
    beta: float = 0.05  # plasticity rate (lower to avoid over-fitting)
    rounds: int = 10    # projection rounds
    semantic_k: int = 300   # k for semantic area (3x standard)


# -- Vocabulary ----------------------------------------------------------------

def _build_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary with clear semantic clusters.

    Two groups of nouns with distinct feature sharing patterns:
    - Animals: all share visual_ANIMAL (high entanglement)
    - Objects: each has unique category feature (low entanglement)
    This creates the baseline asymmetry we want to diagnose.
    """
    return {
        # Animals — all share ANIMAL
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        # Objects — unique features
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "OBJECT"]),
        "car":    GroundingContext(visual=["CAR", "VEHICLE"]),
        # Verbs
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        # Function words
        "the":    GroundingContext(),
    }


def _build_training(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Training sentences with animal-animal co-occurrences."""
    def ctx(w):
        return vocab[w]

    sentences = []
    # Animal transitives
    for subj, verb, obj in [
        ("dog", "chases", "cat"), ("cat", "sees", "bird"),
        ("bird", "chases", "fish"), ("horse", "chases", "dog"),
        ("fish", "sees", "horse"), ("dog", "sees", "bird"),
        ("cat", "chases", "horse"), ("horse", "sees", "cat"),
        ("bird", "finds", "horse"), ("fish", "finds", "cat"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Object sentences (some coverage)
    for subj, verb, obj in [
        ("dog", "finds", "ball"), ("cat", "sees", "book"),
        ("bird", "finds", "car"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


# -- Measurement functions -----------------------------------------------------

def measure_baseline_overlap(
    parser: EmergentParser, word: str,
) -> float:
    """Measure a word's self-reconstruction overlap (no prime).

    Projects phon → core area → snapshot → compare with lexicon entry.
    This is the word's baseline: how well its assembly reproduces from
    phonological input alone (without grounding features).
    """
    core = parser._word_core_area(word)
    lex = parser.core_lexicons.get(core, {}).get(word)
    if lex is None:
        return 0.0
    parser.brain._engine.reset_area_connections(core)
    project(parser.brain, parser.stim_map[word], core, rounds=parser.rounds)
    asm = _snap(parser.brain, core)
    return float(asm_overlap(asm, lex))


def measure_neighbor_overlap(
    parser: EmergentParser, word: str,
) -> float:
    """Average overlap between word's lexicon entry and all neighbors in its area."""
    core = parser._word_core_area(word)
    lex = parser.core_lexicons.get(core, {}).get(word)
    if lex is None:
        return 0.0
    total = 0.0
    count = 0
    for other_word, other_lex in parser.core_lexicons.get(core, {}).items():
        if other_word != word:
            total += asm_overlap(lex, other_lex)
            count += 1
    return total / count if count > 0 else 0.0


def measure_primed_overlap(
    parser: EmergentParser, prime: str, target: str,
) -> float:
    """Project prime then target into same core area, measure target overlap."""
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return 0.0
    parser.brain._engine.reset_area_connections(core)
    # Project prime (if in same core area)
    if prime:
        project(parser.brain, parser.stim_map[prime], core, rounds=parser.rounds)
    # Project target
    project(parser.brain, parser.stim_map[target], core, rounds=parser.rounds)
    asm = _snap(parser.brain, core)
    return float(asm_overlap(asm, lex))


def measure_convergence_curve(
    parser: EmergentParser,
    prime: Optional[str],
    target: str,
    max_rounds: int = 15,
) -> List[float]:
    """Measure round-by-round convergence of target after optional prime.

    Returns list of overlaps, one per round, showing how quickly the
    target assembly converges toward its lexicon entry.

    This is the N400 analogue: faster convergence = easier processing
    = reduced N400 amplitude in the brain.
    """
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    parser.brain._engine.reset_area_connections(core)

    # Prime (if provided)
    if prime:
        project(parser.brain, parser.stim_map[prime], core,
                rounds=parser.rounds)

    # Target: one round at a time, measuring overlap at each step
    phon = parser.stim_map[target]
    overlaps = []
    # First projection (no recurrence yet)
    parser.brain.project({phon: [core]}, {})
    asm = _snap(parser.brain, core)
    overlaps.append(float(asm_overlap(asm, lex)))

    # Subsequent rounds with recurrence
    for _ in range(max_rounds - 1):
        parser.brain.project({phon: [core]}, {core: [core]})
        asm = _snap(parser.brain, core)
        overlaps.append(float(asm_overlap(asm, lex)))

    return overlaps


def measure_grounding_overlap(
    parser: EmergentParser,
    word_a: str, word_b: str,
) -> float:
    """Measure shared grounding features between two words.

    This is the feature-level similarity that SHOULD drive priming
    in the brain but doesn't affect k-WTA assembly competition.
    """
    ctx_a = parser.word_grounding.get(word_a, GroundingContext())
    ctx_b = parser.word_grounding.get(word_b, GroundingContext())

    features_a = set(parser._grounding_stim_names(ctx_a))
    features_b = set(parser._grounding_stim_names(ctx_b))

    if not features_a or not features_b:
        return 0.0

    intersection = len(features_a & features_b)
    union = len(features_a | features_b)
    return intersection / union if union > 0 else 0.0


def measure_activation_energy(
    parser: EmergentParser,
    prime: Optional[str],
    target: str,
) -> Dict[str, float]:
    """Measure how strongly the target activates its CORRECT core area
    vs other areas, after an optional prime.

    Higher selectivity = easier categorization = reduced N400.
    This captures whether the prime makes the target's category
    MORE obvious, even if the specific assembly doesn't change.
    """
    target_core = parser._word_core_area(target)
    phon = parser.stim_map.get(target)
    if phon is None:
        return {"target_area_score": 0.0, "selectivity": 0.0}

    # Prime in target's core area
    if prime:
        parser.brain._engine.reset_area_connections(target_core)
        project(parser.brain, parser.stim_map[prime], target_core,
                rounds=parser.rounds)

    # Project target into ALL core areas, measure readout in each
    from src.assembly_calculus.readout import readout_all

    scores = {}
    for core in CORE_AREAS:
        lex = parser.core_lexicons.get(core, {})
        if not lex:
            scores[core] = 0.0
            continue
        parser.brain._engine.reset_area_connections(core)
        parser.brain.project({phon: [core]}, {})
        if parser.rounds > 1:
            parser.brain.project_rounds(
                target=core,
                areas_by_stim={phon: [core]},
                dst_areas_by_src_area={core: [core]},
                rounds=parser.rounds - 1,
            )
        asm = _snap(parser.brain, core)
        overlaps = readout_all(asm, lex)
        scores[core] = overlaps[0][1] if overlaps else 0.0

    target_score = scores.get(target_core, 0.0)
    other_scores = [v for k, v in scores.items() if k != target_core]
    max_other = max(other_scores) if other_scores else 0.0

    # Selectivity: how much better is the correct area vs runner-up?
    selectivity = target_score - max_other

    return {
        "target_area_score": target_score,
        "max_other_score": max_other,
        "selectivity": selectivity,
    }


# -- Test pairs ----------------------------------------------------------------

# Priming test triplets: (target, related_prime, unrelated_prime)
# Related: same semantic cluster (shared features)
# Unrelated: different cluster (no shared features)
PRIMING_TESTS = [
    ("cat",   "dog",   "table"),
    ("bird",  "cat",   "chair"),
    ("horse", "fish",  "book"),
    ("fish",  "bird",  "car"),
    ("dog",   "horse", "ball"),
    ("table", "chair", "dog"),
    ("chair", "table", "cat"),
    ("book",  "ball",  "bird"),
]


# -- Experiment ----------------------------------------------------------------

class N400DiagnosticExperiment(ExperimentBase):
    """Diagnose anti-priming artifact and test N400 analogues."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="n400_diagnostic",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = N400Config()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training: {len(training)} sentences")
        self.log(f"Seeds: {cfg.n_seeds}")

        # ==============================================================
        # H1: Baseline asymmetry
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H1: Baseline asymmetry (feature sharing -> lower self-overlap)")
        self.log("=" * 60)

        # Group words by whether they share features with many neighbors
        animals = ["dog", "cat", "bird", "horse", "fish"]
        objects = ["table", "chair", "book", "ball", "car"]

        animal_baselines = []
        object_baselines = []
        animal_neighbor_ovs = []
        object_neighbor_ovs = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            a_base = [measure_baseline_overlap(parser, w) for w in animals]
            o_base = [measure_baseline_overlap(parser, w) for w in objects]
            a_neigh = [measure_neighbor_overlap(parser, w) for w in animals]
            o_neigh = [measure_neighbor_overlap(parser, w) for w in objects]

            animal_baselines.append(float(np.mean(a_base)))
            object_baselines.append(float(np.mean(o_base)))
            animal_neighbor_ovs.append(float(np.mean(a_neigh)))
            object_neighbor_ovs.append(float(np.mean(o_neigh)))

        h1_animal = summarize(animal_baselines)
        h1_object = summarize(object_baselines)
        h1_a_neigh = summarize(animal_neighbor_ovs)
        h1_o_neigh = summarize(object_neighbor_ovs)
        h1_test = paired_ttest(animal_baselines, object_baselines)

        self.log(f"  Animals: baseline={h1_animal['mean']:.3f} "
                 f"neighbor_ov={h1_a_neigh['mean']:.3f}")
        self.log(f"  Objects: baseline={h1_object['mean']:.3f} "
                 f"neighbor_ov={h1_o_neigh['mean']:.3f}")
        self.log(f"  Baseline gap: {h1_animal['mean'] - h1_object['mean']:+.3f}")
        self.log(f"  Test: t={h1_test['t']:.2f} p={h1_test['p']:.4f} "
                 f"d={h1_test['d']:.2f}")

        # ==============================================================
        # H2: Proper priming control (same target, different primes)
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H2: Same target, different primes (should be equal)")
        self.log("=" * 60)

        h2_related = []
        h2_unrelated = []
        h2_noprime = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_rel = []
            seed_unrel = []
            seed_none = []
            for target, related, unrelated in PRIMING_TESTS:
                r = measure_primed_overlap(parser, related, target)
                u = measure_primed_overlap(parser, unrelated, target)
                n = measure_baseline_overlap(parser, target)
                seed_rel.append(r)
                seed_unrel.append(u)
                seed_none.append(n)

            h2_related.append(float(np.mean(seed_rel)))
            h2_unrelated.append(float(np.mean(seed_unrel)))
            h2_noprime.append(float(np.mean(seed_none)))

        h2_r = summarize(h2_related)
        h2_u = summarize(h2_unrelated)
        h2_n = summarize(h2_noprime)
        h2_test = paired_ttest(h2_related, h2_unrelated)

        self.log(f"  No prime:        {h2_n['mean']:.3f} +/- {h2_n['sem']:.3f}")
        self.log(f"  Related prime:   {h2_r['mean']:.3f} +/- {h2_r['sem']:.3f}")
        self.log(f"  Unrelated prime: {h2_u['mean']:.3f} +/- {h2_u['sem']:.3f}")
        self.log(f"  Related vs unrelated: t={h2_test['t']:.2f} "
                 f"p={h2_test['p']:.4f}")

        # ==============================================================
        # H3: Convergence speed (the N400 analogue)
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H3: Convergence speed (round-by-round)")
        self.log("=" * 60)

        max_conv_rounds = 12
        # Collect per-round overlaps for each condition
        conv_related = {r: [] for r in range(max_conv_rounds)}
        conv_unrelated = {r: [] for r in range(max_conv_rounds)}
        conv_noprime = {r: [] for r in range(max_conv_rounds)}

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            for target, related, unrelated in PRIMING_TESTS:
                c_r = measure_convergence_curve(parser, related, target,
                                                max_conv_rounds)
                c_u = measure_convergence_curve(parser, unrelated, target,
                                                max_conv_rounds)
                c_n = measure_convergence_curve(parser, None, target,
                                                max_conv_rounds)
                for r in range(max_conv_rounds):
                    conv_related[r].append(c_r[r])
                    conv_unrelated[r].append(c_u[r])
                    conv_noprime[r].append(c_n[r])

        # Compute means and test at each round
        h3_by_round = []
        self.log(f"  {'Round':>5} {'NoPrime':>8} {'Related':>8} "
                 f"{'Unrelated':>10} {'R-U':>6} {'R-N':>6}")
        for r in range(min(max_conv_rounds, 6)):  # Show first 6 rounds
            mn = float(np.mean(conv_noprime[r]))
            mr = float(np.mean(conv_related[r]))
            mu = float(np.mean(conv_unrelated[r]))
            self.log(f"  {r:>5} {mn:>8.3f} {mr:>8.3f} {mu:>10.3f} "
                     f"{mr - mu:>+6.3f} {mr - mn:>+6.3f}")
            h3_by_round.append({
                "round": r,
                "no_prime": mn, "related": mr, "unrelated": mu,
                "related_minus_unrelated": mr - mu,
                "related_minus_noprime": mr - mn,
            })

        # Test at round 1 (earliest signal)
        h3_test_r1 = paired_ttest(conv_related[0], conv_unrelated[0])
        # Test at round 1: related vs no prime
        h3_test_r1_vs_none = paired_ttest(conv_related[0], conv_noprime[0])

        self.log(f"\n  Round 0 related vs unrelated: t={h3_test_r1['t']:.2f} "
                 f"p={h3_test_r1['p']:.4f} d={h3_test_r1['d']:.2f}")
        self.log(f"  Round 0 related vs no prime:  "
                 f"t={h3_test_r1_vs_none['t']:.2f} "
                 f"p={h3_test_r1_vs_none['p']:.4f} "
                 f"d={h3_test_r1_vs_none['d']:.2f}")

        # Convergence speed: how many rounds to reach 0.7 overlap?
        def rounds_to_threshold(curve, threshold=0.7):
            for i, v in enumerate(curve):
                if v >= threshold:
                    return i
            return len(curve)

        speed_related = []
        speed_unrelated = []
        speed_noprime = []
        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            for target, related, unrelated in PRIMING_TESTS:
                c_r = measure_convergence_curve(parser, related, target,
                                                max_conv_rounds)
                c_u = measure_convergence_curve(parser, unrelated, target,
                                                max_conv_rounds)
                c_n = measure_convergence_curve(parser, None, target,
                                                max_conv_rounds)
                speed_related.append(rounds_to_threshold(c_r))
                speed_unrelated.append(rounds_to_threshold(c_u))
                speed_noprime.append(rounds_to_threshold(c_n))

        self.log(f"\n  Rounds to 0.7 overlap:")
        self.log(f"    No prime:  {np.mean(speed_noprime):.2f}")
        self.log(f"    Related:   {np.mean(speed_related):.2f}")
        self.log(f"    Unrelated: {np.mean(speed_unrelated):.2f}")

        # ==============================================================
        # H4: Feature-level facilitation
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H4: Grounding-level feature overlap")
        self.log("=" * 60)

        # This is deterministic (no randomness), but validates the
        # architectural claim: related words share grounding features.
        related_feature_ov = []
        unrelated_feature_ov = []
        for target, related, unrelated in PRIMING_TESTS:
            parser_dummy = EmergentParser(
                n=cfg.n, k=cfg.k, seed=42, vocabulary=vocab)
            r_fov = measure_grounding_overlap(parser_dummy, target, related)
            u_fov = measure_grounding_overlap(parser_dummy, target, unrelated)
            related_feature_ov.append(r_fov)
            unrelated_feature_ov.append(u_fov)

        h4_r = summarize(related_feature_ov)
        h4_u = summarize(unrelated_feature_ov)

        self.log(f"  Related feature overlap:   {h4_r['mean']:.3f}")
        self.log(f"  Unrelated feature overlap: {h4_u['mean']:.3f}")
        self.log(f"  Delta: {h4_r['mean'] - h4_u['mean']:+.3f}")

        # ==============================================================
        # H5: Semantic layer with larger k
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H5: Semantic layer intervention")
        self.log("=" * 60)

        # Add a SEMANTIC area with k=300 (3x normal) where multiple
        # concepts can coexist via shared features.
        SEMANTIC = "SEMANTIC_HUB"

        h5_related = []
        h5_unrelated = []
        h5_noprime = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # Add semantic area with larger k for co-activation
            sem_k = min(cfg.semantic_k, cfg.n // 10)
            parser.brain.add_area(SEMANTIC, cfg.n, sem_k, cfg.beta)

            # Build semantic representations: project each word's core
            # assembly + grounding features into SEMANTIC with recurrence
            semantic_lexicon = {}
            for word in list(animals) + list(objects):
                core = parser._word_core_area(word)
                phon = parser.stim_map[word]
                grounding_stims = parser._grounding_stim_names(
                    parser.word_grounding.get(word, GroundingContext()))

                # Project phon + grounding → core → SEMANTIC
                stim_dict = {phon: [core]}
                for gs in grounding_stims:
                    stim_dict[gs] = [core]
                parser.brain.project(stim_dict, {})
                parser.brain.project(
                    {}, {core: [core, SEMANTIC], SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {core: [SEMANTIC], SEMANTIC: [SEMANTIC]})

                semantic_lexicon[word] = _snap(parser.brain, SEMANTIC)
                parser.brain._engine.reset_area_connections(SEMANTIC)
                parser.brain._engine.reset_area_connections(core)

            # Now measure priming in SEMANTIC area
            seed_rel = []
            seed_unrel = []
            seed_none = []

            for target, related, unrelated in PRIMING_TESTS:
                target_core = parser._word_core_area(target)
                target_sem = semantic_lexicon.get(target)
                if target_sem is None:
                    continue

                # No prime baseline
                parser.brain._engine.reset_area_connections(SEMANTIC)
                parser.brain._engine.reset_area_connections(target_core)
                phon = parser.stim_map[target]
                grounding = parser._grounding_stim_names(
                    parser.word_grounding.get(target, GroundingContext()))
                stim_dict = {phon: [target_core]}
                for gs in grounding:
                    stim_dict[gs] = [target_core]
                parser.brain.project(stim_dict, {})
                parser.brain.project(
                    {}, {target_core: [SEMANTIC], SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {target_core: [SEMANTIC],
                             SEMANTIC: [SEMANTIC]})
                none_asm = _snap(parser.brain, SEMANTIC)
                none_ov = float(asm_overlap(none_asm, target_sem))
                seed_none.append(none_ov)

                # Related prime → target
                parser.brain._engine.reset_area_connections(SEMANTIC)
                prime_core = parser._word_core_area(related)
                parser.brain._engine.reset_area_connections(prime_core)
                prime_phon = parser.stim_map[related]
                prime_ground = parser._grounding_stim_names(
                    parser.word_grounding.get(related, GroundingContext()))
                p_stim = {prime_phon: [prime_core]}
                for gs in prime_ground:
                    p_stim[gs] = [prime_core]
                parser.brain.project(p_stim, {})
                parser.brain.project(
                    {}, {prime_core: [SEMANTIC], SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {prime_core: [SEMANTIC],
                             SEMANTIC: [SEMANTIC]})
                # Now project target (WITHOUT resetting SEMANTIC)
                parser.brain._engine.reset_area_connections(target_core)
                parser.brain.project(stim_dict, {})
                parser.brain.project(
                    {}, {target_core: [SEMANTIC], SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {target_core: [SEMANTIC],
                             SEMANTIC: [SEMANTIC]})
                rel_asm = _snap(parser.brain, SEMANTIC)
                rel_ov = float(asm_overlap(rel_asm, target_sem))
                seed_rel.append(rel_ov)

                # Unrelated prime → target
                parser.brain._engine.reset_area_connections(SEMANTIC)
                uprime_core = parser._word_core_area(unrelated)
                parser.brain._engine.reset_area_connections(uprime_core)
                uprime_phon = parser.stim_map[unrelated]
                uprime_ground = parser._grounding_stim_names(
                    parser.word_grounding.get(unrelated, GroundingContext()))
                up_stim = {uprime_phon: [uprime_core]}
                for gs in uprime_ground:
                    up_stim[gs] = [uprime_core]
                parser.brain.project(up_stim, {})
                parser.brain.project(
                    {}, {uprime_core: [SEMANTIC], SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {uprime_core: [SEMANTIC],
                             SEMANTIC: [SEMANTIC]})
                # Target
                parser.brain._engine.reset_area_connections(target_core)
                parser.brain.project(stim_dict, {})
                parser.brain.project(
                    {}, {target_core: [SEMANTIC], SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {target_core: [SEMANTIC],
                             SEMANTIC: [SEMANTIC]})
                unrel_asm = _snap(parser.brain, SEMANTIC)
                unrel_ov = float(asm_overlap(unrel_asm, target_sem))
                seed_unrel.append(unrel_ov)

            h5_related.append(float(np.mean(seed_rel)))
            h5_unrelated.append(float(np.mean(seed_unrel)))
            h5_noprime.append(float(np.mean(seed_none)))

        h5_r = summarize(h5_related)
        h5_u = summarize(h5_unrelated)
        h5_n = summarize(h5_noprime)
        h5_test = paired_ttest(h5_related, h5_unrelated)
        h5_test_rn = paired_ttest(h5_related, h5_noprime)

        self.log(f"  Semantic layer (k={sem_k}):")
        self.log(f"    No prime:  {h5_n['mean']:.3f} +/- {h5_n['sem']:.3f}")
        self.log(f"    Related:   {h5_r['mean']:.3f} +/- {h5_r['sem']:.3f}")
        self.log(f"    Unrelated: {h5_u['mean']:.3f} +/- {h5_u['sem']:.3f}")
        self.log(f"    Related vs unrelated: t={h5_test['t']:.2f} "
                 f"p={h5_test['p']:.4f} d={h5_test['d']:.2f}")
        self.log(f"    Related vs no prime:  t={h5_test_rn['t']:.2f} "
                 f"p={h5_test_rn['p']:.4f} d={h5_test_rn['d']:.2f}")

        # Is the DIRECTION correct? Related > unrelated = N400 pattern
        n400_direction = h5_r["mean"] > h5_u["mean"]
        self.log(f"    N400 direction (related > unrelated): "
                 f"{'YES' if n400_direction else 'NO'}")

        # ==============================================================
        # Summary
        # ==============================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("N400 DIAGNOSTIC SUMMARY")
        self.log(f"  H1 (baseline asymmetry): "
                 f"animals={h1_animal['mean']:.3f} "
                 f"objects={h1_object['mean']:.3f} "
                 f"gap={h1_animal['mean'] - h1_object['mean']:+.3f}")
        self.log(f"  H2 (no real priming in core): "
                 f"rel={h2_r['mean']:.3f} unrel={h2_u['mean']:.3f} "
                 f"none={h2_n['mean']:.3f}")
        self.log(f"  H3 (convergence speed): "
                 f"round0 R-U={h3_by_round[0]['related_minus_unrelated']:+.3f} "
                 f"R-N={h3_by_round[0]['related_minus_noprime']:+.3f}")
        self.log(f"  H4 (feature overlap): "
                 f"rel={h4_r['mean']:.3f} unrel={h4_u['mean']:.3f}")
        self.log(f"  H5 (semantic layer): "
                 f"rel={h5_r['mean']:.3f} unrel={h5_u['mean']:.3f} "
                 f"{'N400 YES' if n400_direction else 'N400 NO'}")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "semantic_k": sem_k,
                "n_training": len(training),
            },
            metrics={
                "h1_baseline_asymmetry": {
                    "animal_baseline": h1_animal,
                    "object_baseline": h1_object,
                    "animal_neighbor_ov": h1_a_neigh,
                    "object_neighbor_ov": h1_o_neigh,
                    "test": h1_test,
                },
                "h2_proper_control": {
                    "no_prime": h2_n, "related": h2_r,
                    "unrelated": h2_u, "test": h2_test,
                },
                "h3_convergence": {
                    "by_round": h3_by_round,
                    "round0_test_rel_vs_unrel": h3_test_r1,
                    "round0_test_rel_vs_none": h3_test_r1_vs_none,
                    "speed_related": float(np.mean(speed_related)),
                    "speed_unrelated": float(np.mean(speed_unrelated)),
                    "speed_noprime": float(np.mean(speed_noprime)),
                },
                "h4_feature_overlap": {
                    "related": h4_r, "unrelated": h4_u,
                },
                "h5_semantic_layer": {
                    "no_prime": h5_n, "related": h5_r,
                    "unrelated": h5_u,
                    "test_rel_vs_unrel": h5_test,
                    "test_rel_vs_none": h5_test_rn,
                    "n400_direction": n400_direction,
                },
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="N400 priming diagnostic experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer seeds for faster iteration")
    args = parser.parse_args()

    exp = N400DiagnosticExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
