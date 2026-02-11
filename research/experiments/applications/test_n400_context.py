"""
N400 as Context Disruption Experiment

The N400 ERP component reflects the difficulty of integrating a new word
into the current semantic context.  In the brain, this integration happens
in temporal association cortex, NOT in primary sensory cortex.  Our previous
N400 experiments measured priming in the CORE area (lexical access level),
which is the wrong place — the N400 should be measured in the CONTEXT area
(semantic integration level).

Theoretical basis:
    The parser's CONTEXT area accumulates a running representation of all
    processed words via core_area -> CONTEXT with CONTEXT -> CONTEXT
    recurrence.  When a new word projects into CONTEXT, the k-WTA
    competition determines how much the context assembly changes:

    - Related word (shares features with context): its core -> CONTEXT
      connections overlap with already-strengthened connections from
      previous words.  The existing CONTEXT winners receive BOTH
      recurrent input AND new compatible input -> they survive k-WTA
      -> less disruption = smaller N400.

    - Unrelated word (no shared features): its core -> CONTEXT
      connections point to different neurons.  New neurons compete
      against old winners (which only have recurrent input) -> more
      displacement = more disruption = larger N400.

    This is neurobiologically principled:
    1. Measured in integration area (CONTEXT), not access area (core)
    2. Uses existing AC operations (no mechanism changes)
    3. Mediated by Hebbian weight traces and feature overlap
    4. Parallels temporal association cortex N400 generators

Hypotheses:

H1: Direct CONTEXT disruption — After projecting a prime word into
    CONTEXT, a related target word produces LESS context disruption
    than an unrelated target (measured as overlap between C_before
    and C_after the target).

H2: Sentence-level CONTEXT disruption — In full parsed sentences,
    a semantically congruent final word disrupts CONTEXT less than
    an incongruent one.

H3: CONTEXT convergence speed — Within the target word's CONTEXT
    projection rounds, related targets converge faster (reach stable
    overlap sooner) than unrelated targets.

H4: Feature-overlap scaling — Context disruption correlates with
    the degree of grounding-feature overlap between prime and target.

H5: N400 replication — In a 3-condition design (related, unrelated,
    no-prime), the pattern matches the classic N400:
    related < unrelated (facilitation from semantic relatedness).

Statistical methodology:
- N_SEEDS independent random seeds per condition.
- Within-subject design: same parser, same sentence frame, different targets.
- Paired t-tests for related vs unrelated conditions.
- Cohen's d effect sizes.
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
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import (
    NOUN_CORE, VERB_CORE, CONTEXT, CORE_AREAS,
)
from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.assembly import overlap as asm_overlap


@dataclass
class CtxConfig:
    """Configuration for the context-disruption N400 experiment."""
    n: int = 50000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10


# -- Vocabulary ---------------------------------------------------------------

def _build_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary with clear semantic clusters for priming.

    Animals all share ANIMAL -> high feature overlap.
    Objects have unique category features -> low overlap with animals.
    """
    return {
        # Animals (share ANIMAL feature)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Objects (unique features)
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "TOY"]),
        "car":    GroundingContext(visual=["CAR", "VEHICLE"]),
        "cup":    GroundingContext(visual=["CUP", "CONTAINER"]),
        # Verbs
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function words
        "the":    GroundingContext(),
    }


def _build_training(vocab) -> List[GroundedSentence]:
    """Training corpus with rich co-occurrence patterns."""
    def ctx(w):
        return vocab[w]
    sentences = []

    # Animal-animal transitive (build strong animal context traces)
    for subj, verb, obj in [
        ("dog", "chases", "cat"),    ("cat", "sees", "bird"),
        ("bird", "chases", "fish"),  ("horse", "chases", "dog"),
        ("fish", "sees", "horse"),   ("dog", "sees", "bird"),
        ("cat", "chases", "horse"),  ("horse", "sees", "cat"),
        ("bird", "finds", "horse"),  ("fish", "finds", "cat"),
        ("mouse", "chases", "fish"), ("dog", "finds", "mouse"),
        ("cat", "finds", "fish"),    ("horse", "sees", "bird"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Object sentences
    for subj, verb, obj in [
        ("dog", "finds", "ball"),    ("cat", "sees", "book"),
        ("bird", "finds", "car"),    ("horse", "sees", "table"),
        ("dog", "likes", "chair"),   ("cat", "likes", "cup"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


# -- Measurement functions ----------------------------------------------------

def measure_context_disruption_direct(
    parser: EmergentParser,
    prime: str,
    target: str,
) -> Dict[str, float]:
    """Direct CONTEXT disruption: project prime then target, measure change.

    1. Clear CONTEXT activation (keep trained weights!)
    2. Project prime: phon -> core, core -> CONTEXT (with recurrence)
    3. Snapshot C_prime
    4. Project target: phon -> core, core -> CONTEXT (with recurrence)
    5. Snapshot C_after
    6. disruption = 1 - overlap(C_prime, C_after)

    IMPORTANT: We do NOT reset CONTEXT area connections — the trained
    Hebbian weights from parser.train() encode which core neurons map
    to which CONTEXT neurons.  Resetting them would leave only random
    connections and destroy the learned structure.
    """
    prime_core = parser._word_core_area(prime)
    target_core = parser._word_core_area(target)

    # Clear CONTEXT activation only (keep trained Hebbian weights)
    parser.brain.inhibit_areas([CONTEXT])

    # Reset core area recurrence for clean word assembly formation
    parser.brain._engine.reset_area_connections(prime_core)

    # Project prime: phon -> core
    project(parser.brain, parser.stim_map[prime], prime_core,
            rounds=parser.rounds)

    # Prime: core -> CONTEXT with recurrence
    parser.brain.project(
        {}, {prime_core: [CONTEXT], CONTEXT: [CONTEXT]})
    for _ in range(parser.rounds - 1):
        parser.brain.project(
            {}, {prime_core: [CONTEXT], CONTEXT: [CONTEXT]})

    c_prime = _snap(parser.brain, CONTEXT)

    # Project target: phon -> core (reset core recurrence, not CONTEXT)
    if target_core != prime_core:
        parser.brain._engine.reset_area_connections(target_core)
    else:
        parser.brain._engine.reset_area_connections(target_core)
    project(parser.brain, parser.stim_map[target], target_core,
            rounds=parser.rounds)

    # Target: core -> CONTEXT with recurrence (CONTEXT keeps state)
    parser.brain.project(
        {}, {target_core: [CONTEXT], CONTEXT: [CONTEXT]})
    for _ in range(parser.rounds - 1):
        parser.brain.project(
            {}, {target_core: [CONTEXT], CONTEXT: [CONTEXT]})

    c_after = _snap(parser.brain, CONTEXT)

    ov = float(asm_overlap(c_prime, c_after))
    return {
        "overlap": ov,
        "disruption": 1.0 - ov,
    }


def measure_context_disruption_sentence(
    parser: EmergentParser,
    sentence: List[str],
    target_position: int,
) -> Dict[str, float]:
    """Sentence-level CONTEXT disruption at target word position.

    Uses parse_incremental which snapshots CONTEXT after each word.
    Measures overlap between CONTEXT before and after the target word.
    """
    result = parser.parse_incremental(sentence)
    steps = result["steps"]

    if target_position < 1 or target_position >= len(steps):
        return {"overlap": 0.0, "disruption": 1.0}

    c_before = steps[target_position - 1]["context_assembly"]
    c_after = steps[target_position]["context_assembly"]

    ov = float(asm_overlap(c_before, c_after))
    return {
        "overlap": ov,
        "disruption": 1.0 - ov,
        "target_word": steps[target_position]["word"],
        "context_before_word": steps[target_position - 1]["word"],
    }


def measure_context_convergence(
    parser: EmergentParser,
    prime: str,
    target: str,
    max_rounds: int = 12,
) -> List[float]:
    """Round-by-round CONTEXT convergence after target word.

    After projecting the prime into CONTEXT, project the target one
    round at a time and measure how CONTEXT changes each round.

    Returns list of overlap(C_before, C_round_i) values.
    """
    prime_core = parser._word_core_area(prime)
    target_core = parser._word_core_area(target)

    # Clear CONTEXT activation (keep trained weights)
    parser.brain.inhibit_areas([CONTEXT])
    parser.brain._engine.reset_area_connections(prime_core)

    project(parser.brain, parser.stim_map[prime], prime_core,
            rounds=parser.rounds)
    parser.brain.project(
        {}, {prime_core: [CONTEXT], CONTEXT: [CONTEXT]})
    for _ in range(parser.rounds - 1):
        parser.brain.project(
            {}, {prime_core: [CONTEXT], CONTEXT: [CONTEXT]})

    c_prime = _snap(parser.brain, CONTEXT)

    # Project target into core
    parser.brain._engine.reset_area_connections(target_core)
    project(parser.brain, parser.stim_map[target], target_core,
            rounds=parser.rounds)

    # Now project target core -> CONTEXT one round at a time
    overlaps = []
    for _ in range(max_rounds):
        parser.brain.project(
            {}, {target_core: [CONTEXT], CONTEXT: [CONTEXT]})
        c_now = _snap(parser.brain, CONTEXT)
        overlaps.append(float(asm_overlap(c_prime, c_now)))

    return overlaps


def build_context_lexicon(
    parser: EmergentParser,
    words: List[str],
) -> Dict[str, Any]:
    """Build CONTEXT lexicon: each word's CONTEXT assembly when projected alone.

    For each word, project phon -> core -> CONTEXT in isolation and
    snapshot.  This represents what CONTEXT "looks like" when processing
    that word.  Trained core -> CONTEXT connections shape the result.
    """
    lexicon = {}
    for word in words:
        core = parser._word_core_area(word)
        parser.brain.inhibit_areas([CONTEXT])
        parser.brain._engine.reset_area_connections(core)
        project(parser.brain, parser.stim_map[word], core,
                rounds=parser.rounds)
        parser.brain.project(
            {}, {core: [CONTEXT], CONTEXT: [CONTEXT]})
        for _ in range(parser.rounds - 1):
            parser.brain.project(
                {}, {core: [CONTEXT], CONTEXT: [CONTEXT]})
        lexicon[word] = _snap(parser.brain, CONTEXT)
    return lexicon


def measure_context_prediction(
    parser: EmergentParser,
    sentence_prefix: List[str],
    target: str,
    ctx_lexicon: Dict[str, Any],
) -> float:
    """Measure how well sentence context predicts a target word.

    Parse the sentence prefix to build CONTEXT, then measure overlap
    between the CONTEXT assembly and the target's CONTEXT lexicon entry.
    Higher overlap = context better predicts the target = smaller N400.
    """
    # Parse prefix to build CONTEXT
    parser.parse_incremental(sentence_prefix)
    c_ctx = _snap(parser.brain, CONTEXT)

    # Measure prediction: overlap with target's CONTEXT entry
    target_entry = ctx_lexicon.get(target)
    if target_entry is None:
        return 0.0
    return float(asm_overlap(c_ctx, target_entry))


def measure_topdown_convergence(
    parser: EmergentParser,
    sentence_prefix: List[str],
    target: str,
    max_rounds: int = 10,
) -> List[float]:
    """Measure target convergence in core area after top-down feedback.

    1. Parse sentence prefix (builds CONTEXT)
    2. Project CONTEXT -> target's core area (top-down prediction)
    3. Project target's phon -> core area round by round
    4. Measure overlap with target's lexicon entry at each round

    The top-down projection pre-activates neurons in the core area.
    If these neurons overlap with the target's assembly, convergence
    is faster = smaller N400.
    """
    target_core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(target_core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    # Parse prefix to build CONTEXT
    parser.parse_incremental(sentence_prefix)

    # Top-down feedback: CONTEXT -> core area (one round)
    # Uses whatever CONTEXT->core connections exist (random if untrained)
    parser.brain._engine.reset_area_connections(target_core)
    parser.brain.project({}, {CONTEXT: [target_core]})

    # Now project target phon -> core, measuring convergence
    phon = parser.stim_map[target]
    overlaps = []
    parser.brain.project({phon: [target_core]}, {})
    overlaps.append(float(asm_overlap(
        _snap(parser.brain, target_core), lex)))
    for _ in range(max_rounds - 1):
        parser.brain.project(
            {phon: [target_core]}, {target_core: [target_core]})
        overlaps.append(float(asm_overlap(
            _snap(parser.brain, target_core), lex)))
    return overlaps


def train_feedback_connections(
    parser: EmergentParser,
    sentences: List[GroundedSentence],
    n_passes: int = 3,
):
    """Train CONTEXT -> core feedback connections via Hebbian learning.

    For each training sentence, parse it normally, then at each word,
    project CONTEXT -> core area.  This establishes Hebbian traces:
    the context state at each word position gets connected to that
    word's core assembly.

    At test time, similar context states will pre-activate related
    word assemblies in core areas through these feedback connections.
    """
    for _ in range(n_passes):
        for sent in sentences:
            words = sent.words
            # Parse normally first to build CONTEXT
            parser.brain.inhibit_areas([CONTEXT])
            for word in words:
                core = parser._word_core_area(word)
                parser.brain._engine.reset_area_connections(core)
                phon = parser.stim_map.get(word)
                if phon:
                    project(parser.brain, phon, core,
                            rounds=parser.rounds)
                # Forward: core -> CONTEXT
                parser.brain.project(
                    {}, {core: [CONTEXT], CONTEXT: [CONTEXT]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {core: [CONTEXT], CONTEXT: [CONTEXT]})

                # FEEDBACK: CONTEXT -> core (trains feedback connections)
                parser.brain.project(
                    {}, {CONTEXT: [core]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {CONTEXT: [core], core: [core]})


# -- Test configurations ------------------------------------------------------

# Direct priming pairs: (prime, related_target, unrelated_target)
DIRECT_PAIRS = [
    ("dog",   "cat",   "table"),
    ("cat",   "bird",  "chair"),
    ("horse", "fish",  "book"),
    ("fish",  "bird",  "car"),
    ("dog",   "horse", "ball"),
    ("bird",  "mouse", "cup"),
    ("mouse", "dog",   "table"),
    ("horse", "cat",   "chair"),
]

# Sentence frames: "the [SUBJECT] sees the [TARGET]"
# Target position = 4 (0-indexed)
SENTENCE_PAIRS = [
    # (subject, related_target, unrelated_target)
    ("dog",   "cat",   "table"),
    ("cat",   "bird",  "chair"),
    ("horse", "fish",  "book"),
    ("fish",  "mouse", "car"),
    ("bird",  "dog",   "cup"),
    ("mouse", "horse", "ball"),
]


# -- Experiment ---------------------------------------------------------------

class N400ContextExperiment(ExperimentBase):
    """Test N400 as context disruption in the CONTEXT area."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="n400_context",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = CtxConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training: {len(training)} sentences")
        self.log(f"Seeds: {cfg.n_seeds}")
        self.log(f"Params: n={cfg.n}, k={cfg.k}, p={cfg.p}, "
                 f"beta={cfg.beta}, rounds={cfg.rounds}")

        # ==============================================================
        # H1: Direct CONTEXT disruption
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H1: Direct CONTEXT disruption (prime -> target)")
        self.log("=" * 60)

        h1_related = []
        h1_unrelated = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_rel = []
            seed_unrel = []
            for prime, rel_target, unrel_target in DIRECT_PAIRS:
                r = measure_context_disruption_direct(
                    parser, prime, rel_target)
                u = measure_context_disruption_direct(
                    parser, prime, unrel_target)
                seed_rel.append(r["disruption"])
                seed_unrel.append(u["disruption"])

            h1_related.append(float(np.mean(seed_rel)))
            h1_unrelated.append(float(np.mean(seed_unrel)))

        h1_r = summarize(h1_related)
        h1_u = summarize(h1_unrelated)
        h1_test = paired_ttest(h1_related, h1_unrelated)

        self.log(f"  Related disruption:   {h1_r['mean']:.4f} "
                 f"+/- {h1_r['sem']:.4f}")
        self.log(f"  Unrelated disruption: {h1_u['mean']:.4f} "
                 f"+/- {h1_u['sem']:.4f}")
        self.log(f"  Delta (unrel - rel):  "
                 f"{h1_u['mean'] - h1_r['mean']:+.4f}")
        self.log(f"  Test: t={h1_test['t']:.2f} p={h1_test['p']:.4f} "
                 f"d={h1_test['d']:.2f} "
                 f"{'*' if h1_test['significant'] else ''}")
        n400_h1 = h1_r["mean"] < h1_u["mean"]
        self.log(f"  N400 pattern (related < unrelated): "
                 f"{'YES' if n400_h1 else 'NO'}")

        # Per-pair details for seed 0
        parser0 = EmergentParser(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
            seed=self.seed, rounds=cfg.rounds, vocabulary=vocab,
        )
        parser0.train(sentences=training)
        self.log(f"\n  Per-pair details (seed 0):")
        self.log(f"  {'Prime':<8} {'RelTgt':<8} {'UnrelTgt':<8} "
                 f"{'RelDisr':>8} {'UnrDisr':>8} {'Delta':>8}")
        for prime, rel_target, unrel_target in DIRECT_PAIRS:
            r = measure_context_disruption_direct(
                parser0, prime, rel_target)
            u = measure_context_disruption_direct(
                parser0, prime, unrel_target)
            self.log(f"  {prime:<8} {rel_target:<8} {unrel_target:<8} "
                     f"{r['disruption']:>8.4f} {u['disruption']:>8.4f} "
                     f"{u['disruption'] - r['disruption']:>+8.4f}")

        # ==============================================================
        # H2: Sentence-level CONTEXT disruption
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H2: Sentence-level CONTEXT disruption")
        self.log("=" * 60)
        self.log("  Frame: 'the [SUBJ] sees the [TARGET]'")
        self.log("  Target position: 4 (0-indexed)")

        h2_related = []
        h2_unrelated = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_rel = []
            seed_unrel = []
            for subj, rel_target, unrel_target in SENTENCE_PAIRS:
                rel_sent = ["the", subj, "sees", "the", rel_target]
                unrel_sent = ["the", subj, "sees", "the", unrel_target]
                r = measure_context_disruption_sentence(
                    parser, rel_sent, target_position=4)
                u = measure_context_disruption_sentence(
                    parser, unrel_sent, target_position=4)
                seed_rel.append(r["disruption"])
                seed_unrel.append(u["disruption"])

            h2_related.append(float(np.mean(seed_rel)))
            h2_unrelated.append(float(np.mean(seed_unrel)))

        h2_r = summarize(h2_related)
        h2_u = summarize(h2_unrelated)
        h2_test = paired_ttest(h2_related, h2_unrelated)

        self.log(f"  Related disruption:   {h2_r['mean']:.4f} "
                 f"+/- {h2_r['sem']:.4f}")
        self.log(f"  Unrelated disruption: {h2_u['mean']:.4f} "
                 f"+/- {h2_u['sem']:.4f}")
        self.log(f"  Delta (unrel - rel):  "
                 f"{h2_u['mean'] - h2_r['mean']:+.4f}")
        self.log(f"  Test: t={h2_test['t']:.2f} p={h2_test['p']:.4f} "
                 f"d={h2_test['d']:.2f} "
                 f"{'*' if h2_test['significant'] else ''}")
        n400_h2 = h2_r["mean"] < h2_u["mean"]
        self.log(f"  N400 pattern (related < unrelated): "
                 f"{'YES' if n400_h2 else 'NO'}")

        # Per-pair details
        parser0 = EmergentParser(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
            seed=self.seed, rounds=cfg.rounds, vocabulary=vocab,
        )
        parser0.train(sentences=training)
        self.log(f"\n  Per-pair details (seed 0):")
        self.log(f"  {'Subj':<8} {'RelTgt':<8} {'UnrelTgt':<8} "
                 f"{'RelDisr':>8} {'UnrDisr':>8} {'Delta':>8}")
        for subj, rel_target, unrel_target in SENTENCE_PAIRS:
            rel_sent = ["the", subj, "sees", "the", rel_target]
            unrel_sent = ["the", subj, "sees", "the", unrel_target]
            r = measure_context_disruption_sentence(
                parser0, rel_sent, target_position=4)
            u = measure_context_disruption_sentence(
                parser0, unrel_sent, target_position=4)
            self.log(f"  {subj:<8} {rel_target:<8} {unrel_target:<8} "
                     f"{r['disruption']:>8.4f} {u['disruption']:>8.4f} "
                     f"{u['disruption'] - r['disruption']:>+8.4f}")

        # ==============================================================
        # H3: CONTEXT convergence speed
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H3: CONTEXT convergence speed (round-by-round)")
        self.log("=" * 60)

        max_conv_rounds = 10
        # overlap = how much of prime context survives each round
        conv_related = {r: [] for r in range(max_conv_rounds)}
        conv_unrelated = {r: [] for r in range(max_conv_rounds)}

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            for prime, rel_target, unrel_target in DIRECT_PAIRS:
                c_r = measure_context_convergence(
                    parser, prime, rel_target, max_conv_rounds)
                c_u = measure_context_convergence(
                    parser, prime, unrel_target, max_conv_rounds)
                for r in range(max_conv_rounds):
                    conv_related[r].append(c_r[r])
                    conv_unrelated[r].append(c_u[r])

        self.log(f"  {'Round':>5} {'Related':>8} {'Unrel':>8} "
                 f"{'Rel-Unr':>8} (overlap with pre-target context)")
        h3_by_round = []
        for r in range(max_conv_rounds):
            mr = float(np.mean(conv_related[r]))
            mu = float(np.mean(conv_unrelated[r]))
            self.log(f"  {r:>5} {mr:>8.4f} {mu:>8.4f} "
                     f"{mr - mu:>+8.4f}")
            h3_by_round.append({
                "round": r,
                "related": mr, "unrelated": mu,
                "delta": mr - mu,
            })

        # Test at each round
        self.log(f"\n  Statistical tests by round:")
        h3_tests = {}
        for r in range(min(max_conv_rounds, 6)):
            t = paired_ttest(conv_related[r], conv_unrelated[r])
            sig = "*" if t["significant"] else ""
            self.log(f"    Round {r}: t={t['t']:.2f} p={t['p']:.4f} "
                     f"d={t['d']:.2f} {sig}")
            h3_tests[f"round_{r}"] = t

        # ==============================================================
        # H4: Feature-overlap scaling
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H4: Feature-overlap scaling")
        self.log("=" * 60)

        # Use seed 0 parser, measure disruption for pairs with varying
        # feature overlap
        parser0 = EmergentParser(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
            seed=self.seed, rounds=cfg.rounds, vocabulary=vocab,
        )
        parser0.train(sentences=training)

        # Categorize pairs by feature overlap level
        overlap_levels = {
            "same_category_animal": [
                ("dog", "cat"), ("cat", "bird"), ("horse", "fish"),
                ("bird", "mouse"), ("fish", "dog"), ("mouse", "horse"),
            ],
            "same_category_object": [
                ("table", "chair"), ("book", "ball"), ("car", "cup"),
            ],
            "cross_category": [
                ("dog", "table"), ("cat", "chair"), ("bird", "book"),
                ("horse", "car"), ("fish", "cup"), ("mouse", "ball"),
            ],
        }

        self.log(f"  {'Level':<25} {'Disruption':>10} {'N':>4}")
        h4_data = {}
        for level, pairs in overlap_levels.items():
            disrs = []
            for prime, target in pairs:
                r = measure_context_disruption_direct(
                    parser0, prime, target)
                disrs.append(r["disruption"])
            mean_d = float(np.mean(disrs))
            self.log(f"  {level:<25} {mean_d:>10.4f} {len(pairs):>4}")
            h4_data[level] = {
                "mean_disruption": mean_d,
                "n_pairs": len(pairs),
                "disruptions": disrs,
            }

        # Test: same_category < cross_category?
        h4_test = paired_ttest(
            h4_data["same_category_animal"]["disruptions"]
            + h4_data["same_category_object"]["disruptions"],
            h4_data["cross_category"]["disruptions"],
        ) if (len(h4_data["same_category_animal"]["disruptions"])
              + len(h4_data["same_category_object"]["disruptions"])
              == len(h4_data["cross_category"]["disruptions"])) else None

        if h4_test:
            self.log(f"\n  Same-cat vs cross-cat: t={h4_test['t']:.2f} "
                     f"p={h4_test['p']:.4f} d={h4_test['d']:.2f}")

        # ==============================================================
        # H5: Classic 3-condition N400 replication
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H5: Classic N400 (3-condition sentence design)")
        self.log("=" * 60)
        self.log("  Sentences: 'the [ANIMAL] sees the [TARGET]'")
        self.log("  Related: TARGET = animal, Unrelated: TARGET = object")

        # Use more sentence pairs for power
        n400_pairs = [
            ("dog",   "cat",   "table"),
            ("cat",   "bird",  "chair"),
            ("horse", "fish",  "book"),
            ("fish",  "mouse", "car"),
            ("bird",  "dog",   "cup"),
            ("mouse", "horse", "ball"),
            ("dog",   "bird",  "car"),
            ("cat",   "horse", "table"),
        ]

        h5_related = []
        h5_unrelated = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_rel = []
            seed_unrel = []
            for subj, rel_target, unrel_target in n400_pairs:
                rel_sent = ["the", subj, "sees", "the", rel_target]
                unrel_sent = ["the", subj, "sees", "the", unrel_target]
                r = measure_context_disruption_sentence(
                    parser, rel_sent, target_position=4)
                u = measure_context_disruption_sentence(
                    parser, unrel_sent, target_position=4)
                seed_rel.append(r["disruption"])
                seed_unrel.append(u["disruption"])

            h5_related.append(float(np.mean(seed_rel)))
            h5_unrelated.append(float(np.mean(seed_unrel)))

        h5_r = summarize(h5_related)
        h5_u = summarize(h5_unrelated)
        h5_test = paired_ttest(h5_related, h5_unrelated)

        self.log(f"\n  Related N400:   {h5_r['mean']:.4f} "
                 f"+/- {h5_r['sem']:.4f}")
        self.log(f"  Unrelated N400: {h5_u['mean']:.4f} "
                 f"+/- {h5_u['sem']:.4f}")
        self.log(f"  N400 effect (unrel - rel): "
                 f"{h5_u['mean'] - h5_r['mean']:+.4f}")
        self.log(f"  Test: t={h5_test['t']:.2f} p={h5_test['p']:.4f} "
                 f"d={h5_test['d']:.2f} "
                 f"{'*' if h5_test['significant'] else ''}")
        n400_h5 = h5_r["mean"] < h5_u["mean"]
        self.log(f"  Classic N400 pattern: "
                 f"{'REPLICATED' if n400_h5 and h5_test['significant'] else 'NOT REPLICATED'}")

        # ==============================================================
        # H6: Context prediction (does context predict related better?)
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H6: Context prediction (overlap with CONTEXT lexicon)")
        self.log("=" * 60)
        self.log("  Build CONTEXT lexicon for each word, then measure")
        self.log("  how well sentence context matches the target's entry.")

        animals = ["dog", "cat", "bird", "horse", "fish", "mouse"]
        objects = ["table", "chair", "book", "ball", "car", "cup"]

        h6_related = []
        h6_unrelated = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # Build CONTEXT lexicon for all nouns
            ctx_lex = build_context_lexicon(parser, animals + objects)

            seed_rel = []
            seed_unrel = []
            for subj, rel_target, unrel_target in n400_pairs:
                prefix = ["the", subj, "sees", "the"]
                r = measure_context_prediction(
                    parser, prefix, rel_target, ctx_lex)
                u = measure_context_prediction(
                    parser, prefix, unrel_target, ctx_lex)
                seed_rel.append(r)
                seed_unrel.append(u)

            h6_related.append(float(np.mean(seed_rel)))
            h6_unrelated.append(float(np.mean(seed_unrel)))

        h6_r = summarize(h6_related)
        h6_u = summarize(h6_unrelated)
        h6_test = paired_ttest(h6_related, h6_unrelated)

        self.log(f"  Related prediction:   {h6_r['mean']:.4f} "
                 f"+/- {h6_r['sem']:.4f}")
        self.log(f"  Unrelated prediction: {h6_u['mean']:.4f} "
                 f"+/- {h6_u['sem']:.4f}")
        self.log(f"  Delta (rel - unrel):  "
                 f"{h6_r['mean'] - h6_u['mean']:+.4f}")
        self.log(f"  Test: t={h6_test['t']:.2f} p={h6_test['p']:.4f} "
                 f"d={h6_test['d']:.2f} "
                 f"{'*' if h6_test['significant'] else ''}")
        n400_h6 = h6_r["mean"] > h6_u["mean"]
        self.log(f"  N400 pattern (related > unrelated): "
                 f"{'YES' if n400_h6 else 'NO'}")

        # Per-pair details
        parser0 = EmergentParser(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
            seed=self.seed, rounds=cfg.rounds, vocabulary=vocab,
        )
        parser0.train(sentences=training)
        ctx_lex0 = build_context_lexicon(parser0, animals + objects)
        self.log(f"\n  Per-pair details (seed 0):")
        self.log(f"  {'Subj':<8} {'RelTgt':<8} {'UnrelTgt':<8} "
                 f"{'RelPred':>8} {'UnrPred':>8} {'Delta':>8}")
        for subj, rel_target, unrel_target in n400_pairs:
            prefix = ["the", subj, "sees", "the"]
            r = measure_context_prediction(
                parser0, prefix, rel_target, ctx_lex0)
            u = measure_context_prediction(
                parser0, prefix, unrel_target, ctx_lex0)
            self.log(f"  {subj:<8} {rel_target:<8} {unrel_target:<8} "
                     f"{r:>8.4f} {u:>8.4f} "
                     f"{r - u:>+8.4f}")

        # ==============================================================
        # H7: Top-down feedback (CONTEXT -> core pre-activation)
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H7: Top-down feedback (trained CONTEXT -> core)")
        self.log("=" * 60)
        self.log("  Train feedback connections, then measure if context")
        self.log("  pre-activates related target assemblies in core.")

        h7_related = []
        h7_unrelated = []
        h7_baseline = []  # no context (just phon -> core)

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # Train feedback connections
            train_feedback_connections(parser, training, n_passes=2)

            seed_rel = []
            seed_unrel = []
            seed_base = []
            for subj, rel_target, unrel_target in n400_pairs:
                prefix = ["the", subj, "sees", "the"]

                # Related: convergence with feedback
                c_r = measure_topdown_convergence(
                    parser, prefix, rel_target, max_rounds=8)
                seed_rel.append(c_r[0])  # round 0 overlap

                # Unrelated: convergence with feedback
                c_u = measure_topdown_convergence(
                    parser, prefix, unrel_target, max_rounds=8)
                seed_unrel.append(c_u[0])

                # Baseline: no context, just phon -> core
                target_core = parser._word_core_area(rel_target)
                lex_r = parser.core_lexicons.get(
                    target_core, {}).get(rel_target)
                if lex_r is not None:
                    parser.brain._engine.reset_area_connections(
                        target_core)
                    phon = parser.stim_map[rel_target]
                    parser.brain.project({phon: [target_core]}, {})
                    asm = _snap(parser.brain, target_core)
                    seed_base.append(float(asm_overlap(asm, lex_r)))

            h7_related.append(float(np.mean(seed_rel)))
            h7_unrelated.append(float(np.mean(seed_unrel)))
            h7_baseline.append(float(np.mean(seed_base)))

        h7_r = summarize(h7_related)
        h7_u = summarize(h7_unrelated)
        h7_b = summarize(h7_baseline)
        h7_test = paired_ttest(h7_related, h7_unrelated)

        self.log(f"  Baseline (no context): {h7_b['mean']:.4f}")
        self.log(f"  Related (with ctx):    {h7_r['mean']:.4f} "
                 f"+/- {h7_r['sem']:.4f}")
        self.log(f"  Unrelated (with ctx):  {h7_u['mean']:.4f} "
                 f"+/- {h7_u['sem']:.4f}")
        self.log(f"  Delta (rel - unrel):   "
                 f"{h7_r['mean'] - h7_u['mean']:+.4f}")
        self.log(f"  Test: t={h7_test['t']:.2f} p={h7_test['p']:.4f} "
                 f"d={h7_test['d']:.2f} "
                 f"{'*' if h7_test['significant'] else ''}")
        n400_h7 = h7_r["mean"] > h7_u["mean"]
        self.log(f"  N400 pattern (related > unrelated convergence): "
                 f"{'YES' if n400_h7 else 'NO'}")

        # ==============================================================
        # Summary
        # ==============================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("N400 CONTEXT DISRUPTION SUMMARY")
        self.log("=" * 60)
        self.log(f"  H1 (direct disruption):   rel={h1_r['mean']:.4f} "
                 f"unrel={h1_u['mean']:.4f} d={h1_test['d']:.2f} "
                 f"{'SUPPORTED' if h1_test['significant'] and n400_h1 else 'NOT SUPPORTED'}")
        self.log(f"  H2 (sentence disruption):  rel={h2_r['mean']:.4f} "
                 f"unrel={h2_u['mean']:.4f} d={h2_test['d']:.2f} "
                 f"{'SUPPORTED' if h2_test['significant'] and n400_h2 else 'NOT SUPPORTED'}")
        self.log(f"  H3 (convergence speed):   "
                 f"{'SUPPORTED' if any(h3_tests[f'round_{r}']['significant'] for r in range(min(max_conv_rounds, 6))) else 'NOT SUPPORTED'}")
        self.log(f"  H4 (feature scaling):     "
                 f"same_cat={np.mean(h4_data['same_category_animal']['disruptions'] + h4_data['same_category_object']['disruptions']):.4f} "
                 f"cross={np.mean(h4_data['cross_category']['disruptions']):.4f}")
        self.log(f"  H5 (classic N400):        "
                 f"{'REPLICATED' if n400_h5 and h5_test['significant'] else 'NOT REPLICATED'} "
                 f"d={h5_test['d']:.2f}")
        self.log(f"  H6 (context prediction):  "
                 f"rel={h6_r['mean']:.4f} unrel={h6_u['mean']:.4f} "
                 f"d={h6_test['d']:.2f} "
                 f"{'SUPPORTED' if h6_test['significant'] and n400_h6 else 'NOT SUPPORTED'}")
        self.log(f"  H7 (top-down feedback):   "
                 f"rel={h7_r['mean']:.4f} unrel={h7_u['mean']:.4f} "
                 f"d={h7_test['d']:.2f} "
                 f"{'SUPPORTED' if h7_test['significant'] and n400_h7 else 'NOT SUPPORTED'}")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_training": len(training),
            },
            metrics={
                "h1_direct": {
                    "related": h1_r, "unrelated": h1_u,
                    "test": h1_test, "n400_pattern": n400_h1,
                },
                "h2_sentence": {
                    "related": h2_r, "unrelated": h2_u,
                    "test": h2_test, "n400_pattern": n400_h2,
                },
                "h3_convergence": {
                    "by_round": h3_by_round,
                    "tests": h3_tests,
                },
                "h4_scaling": h4_data,
                "h5_classic": {
                    "related": h5_r, "unrelated": h5_u,
                    "test": h5_test, "n400_pattern": n400_h5,
                },
                "h6_prediction": {
                    "related": h6_r, "unrelated": h6_u,
                    "test": h6_test, "n400_pattern": n400_h6,
                },
                "h7_topdown": {
                    "related": h7_r, "unrelated": h7_u,
                    "baseline": h7_b,
                    "test": h7_test, "n400_pattern": n400_h7,
                },
            },
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="N400 context disruption experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer seeds for faster iteration")
    args = parser.parse_args()

    exp = N400ContextExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")
    for h in ["h1_direct", "h2_sentence", "h5_classic"]:
        d = result.metrics[h]["test"]["d"]
        p = result.metrics[h]["test"]["p"]
        pat = result.metrics[h]["n400_pattern"]
        sig = result.metrics[h]["test"]["significant"]
        status = "SUPPORTED" if pat and sig else "not supported"
        print(f"  {h}: d={d:.2f} p={p:.4f} {status}")


if __name__ == "__main__":
    main()
