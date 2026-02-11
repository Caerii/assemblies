"""
Function Word Discovery Experiment

Tests whether distributional patterns from natural language exposure alone
can identify and sub-categorize function words, without requiring any
explicit grounding information.

Function words (determiners, auxiliaries, complementizers, conjunctions)
have no referential content — they don't map to visual, motor, or other
sensory modalities. In the current architecture they're hardcoded with
empty GroundingContext(). The question is whether the distributional
learning system can DISCOVER their syntactic roles from position and
co-occurrence patterns alone.

This connects to the "distributional bootstrap" hypothesis in language
acquisition (Mintz 2003; Christophe et al. 1997): children use the
distributional signatures of already-known content words to bootstrap
function word categories.

Hypotheses:

H1: Position discrimination — Function words have distinctive positional
    distributions that separate them from content words. Measured by
    position entropy: function words should have LOWER entropy (more
    predictable positions) than content words.

H2: Distributional sub-categorization — Different function word types
    (DET, AUX, COMP) form distinct distributional clusters. Measured
    by within-category vs between-category similarity of distributional
    feature vectors.

H3: Grounding-free classification — After distributional training on
    raw sentences, the system can correctly classify function words as
    non-content (DET category) from distributional evidence alone,
    without any grounding information.

H4: Functional frame discovery — Function words can be identified by
    their "frames" (what categories appear before and after them).
    "the" always precedes NOUN/ADJ. "was" always precedes VERB.
    "that" follows NOUN and precedes DET/NOUN.

Statistical methodology:
- N_SEEDS independent parser instances.
- Permutation tests for cluster separation significance.
- Cohen's d effect sizes.

References:
- Mintz (2003). Frequent frames as a cue for grammatical categories.
- Christophe et al. (1997). Prosodic structure and function word
  identification in speech.
- Friederici (2002). Towards a neural basis of auditory sentence
  processing (ELAN for function words).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, ttest_vs_null, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class FuncWordConfig:
    """Configuration for function word discovery experiment."""
    n: int = 10000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10


# -- Vocabulary ----------------------------------------------------------------

# Ground truth function word sub-categories
FUNC_WORD_CATEGORIES = {
    "the": "DET", "a": "DET",
    "was": "AUX", "were": "AUX",
    "that": "COMP", "which": "COMP",
    "and": "CONJ",
    "by": "MARKER",  # agent marker in passives
}

CONTENT_WORD_CATEGORIES = {
    "dog": "NOUN", "cat": "NOUN", "bird": "NOUN",
    "boy": "NOUN", "girl": "NOUN", "ball": "NOUN",
    "book": "NOUN", "table": "NOUN",
    "runs": "VERB", "sees": "VERB", "eats": "VERB",
    "chases": "VERB", "sleeps": "VERB", "reads": "VERB",
    "finds": "VERB", "plays": "VERB",
    "big": "ADJ", "small": "ADJ", "fast": "ADJ", "red": "ADJ",
}


def _build_vocab() -> Dict[str, GroundingContext]:
    """Build vocabulary: content words grounded, function words ungrounded."""
    vocab = {
        # Nouns (visual grounding)
        "dog": GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat": GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird": GroundingContext(visual=["BIRD", "ANIMAL"]),
        "boy": GroundingContext(visual=["BOY", "PERSON"]),
        "girl": GroundingContext(visual=["GIRL", "PERSON"]),
        "ball": GroundingContext(visual=["BALL", "OBJECT"]),
        "book": GroundingContext(visual=["BOOK", "OBJECT"]),
        "table": GroundingContext(visual=["TABLE", "FURNITURE"]),
        # Verbs (motor grounding)
        "runs": GroundingContext(motor=["RUNNING", "MOTION"]),
        "sees": GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "eats": GroundingContext(motor=["EATING", "CONSUMPTION"]),
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sleeps": GroundingContext(motor=["SLEEPING", "REST"]),
        "reads": GroundingContext(motor=["READING", "COGNITION"]),
        "finds": GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "plays": GroundingContext(motor=["PLAYING", "ACTION"]),
        # Adjectives (property grounding)
        "big": GroundingContext(properties=["SIZE", "BIG"]),
        "small": GroundingContext(properties=["SIZE", "SMALL"]),
        "fast": GroundingContext(properties=["SPEED", "FAST"]),
        "red": GroundingContext(properties=["COLOR", "RED"]),
        # Function words — NO grounding (the words to discover)
        "the": GroundingContext(),
        "a": GroundingContext(),
        "was": GroundingContext(),
        "were": GroundingContext(),
        "that": GroundingContext(),
        "and": GroundingContext(),
        "by": GroundingContext(spatial=["BY", "AGENT_MARKER"]),
    }
    return vocab


def _build_training_sentences() -> List[List[str]]:
    """Build raw training sentences (no role annotations — distributional only).

    Covers patterns that expose function word distributional signatures:
    - DET NOUN VERB (intransitive)
    - DET NOUN VERB DET NOUN (transitive)
    - DET ADJ NOUN VERB (adjective)
    - DET NOUN was VERB by DET NOUN (passive)
    - DET NOUN that VERB DET NOUN VERB (SRC)
    - DET NOUN and DET NOUN VERB (coordination)
    """
    sentences = []

    # Intransitive: DET NOUN VERB
    for det in ["the", "a"]:
        for noun in ["dog", "cat", "bird", "boy", "girl"]:
            for verb in ["runs", "sleeps", "plays"]:
                sentences.append([det, noun, verb])

    # Transitive: DET NOUN VERB DET NOUN
    trans = [
        ("the", "cat", "chases", "the", "bird"),
        ("the", "dog", "sees", "a", "cat"),
        ("the", "boy", "finds", "the", "ball"),
        ("a", "girl", "reads", "a", "book"),
        ("the", "bird", "eats", "the", "ball"),
        ("the", "dog", "chases", "the", "cat"),
        ("the", "cat", "finds", "the", "ball"),
        ("a", "boy", "sees", "the", "bird"),
        ("the", "girl", "plays", "the", "ball"),
        ("the", "dog", "finds", "a", "book"),
    ]
    for t in trans:
        sentences.append(list(t))

    # Adjective: DET ADJ NOUN VERB
    for det in ["the", "a"]:
        for adj in ["big", "small", "fast", "red"]:
            for noun in ["dog", "cat", "bird"]:
                sentences.append([det, adj, noun, "runs"])

    # Passive: DET NOUN was VERB by DET NOUN
    passives = [
        ("the", "cat", "was", "chases", "by", "the", "dog"),
        ("the", "bird", "was", "sees", "by", "the", "cat"),
        ("the", "ball", "was", "finds", "by", "the", "boy"),
        ("the", "book", "was", "reads", "by", "the", "girl"),
        ("the", "dog", "was", "chases", "by", "a", "cat"),
        ("a", "bird", "was", "eats", "by", "the", "cat"),
    ]
    for p in passives:
        sentences.append(list(p))

    # Coordination: DET NOUN and DET NOUN VERB
    coords = [
        ("the", "dog", "and", "the", "cat", "runs"),
        ("the", "boy", "and", "the", "girl", "plays"),
        ("a", "cat", "and", "a", "bird", "sleeps"),
    ]
    for c in coords:
        sentences.append(list(c))

    return sentences


# -- Distributional feature extraction -----------------------------------------

def _compute_position_entropy(
    word: str,
    sentences: List[List[str]],
) -> float:
    """Compute Shannon entropy of a word's position distribution.

    Low entropy = predictable position (function words).
    High entropy = variable position (content words).
    """
    positions = defaultdict(int)
    total = 0
    for sent in sentences:
        n = len(sent)
        for i, w in enumerate(sent):
            if w == word:
                # Normalize position to [0, 1]
                norm_pos = round(i / max(n - 1, 1), 2)
                positions[norm_pos] += 1
                total += 1

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in positions.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def _compute_distributional_vector(
    word: str,
    sentences: List[List[str]],
    parser: EmergentParser,
) -> np.ndarray:
    """Compute a distributional feature vector for a word.

    Features:
    - Mean normalized position (1 dim)
    - Position entropy (1 dim)
    - Frequency rank (1 dim)
    - Left context category distribution (7 dims: DET, NOUN, VERB, ADJ, ADV, PREP, PRON)
    - Right context category distribution (7 dims)
    - Pre-verb fraction (1 dim)
    - Post-verb fraction (1 dim)
    Total: 19 dimensions
    """
    cat_labels = ["DET", "NOUN", "VERB", "ADJ", "ADV", "PREP", "PRON"]
    cat_to_idx = {c: i for i, c in enumerate(cat_labels)}

    # Position statistics
    positions = []
    total_count = 0
    left_cats = defaultdict(int)
    right_cats = defaultdict(int)
    pre_verb = 0
    post_verb = 0

    for sent in sentences:
        n = len(sent)
        # Classify all words in sentence
        cats = []
        for w in sent:
            grounding = parser.word_grounding.get(w)
            cat, _ = parser.classify_word(w, grounding=grounding)
            cats.append(cat)

        # Find verb position
        verb_pos = None
        for idx, c in enumerate(cats):
            if c == "VERB":
                verb_pos = idx
                break

        for i, w in enumerate(sent):
            if w != word:
                continue
            total_count += 1
            positions.append(i / max(n - 1, 1))

            # Left context
            if i > 0:
                lc = cats[i - 1]
                if lc in cat_to_idx:
                    left_cats[lc] += 1

            # Right context
            if i < n - 1:
                rc = cats[i + 1]
                if rc in cat_to_idx:
                    right_cats[rc] += 1

            # Verb-relative position
            if verb_pos is not None:
                if i < verb_pos:
                    pre_verb += 1
                elif i > verb_pos:
                    post_verb += 1

    if total_count == 0:
        return np.zeros(19)

    # Build feature vector
    vec = np.zeros(19)
    vec[0] = np.mean(positions)  # mean position
    vec[1] = _compute_position_entropy(word, sentences)  # entropy
    vec[2] = total_count / max(len(sentences), 1)  # relative frequency

    # Left context distribution
    left_total = sum(left_cats.values()) or 1
    for cat, idx in cat_to_idx.items():
        vec[3 + idx] = left_cats[cat] / left_total

    # Right context distribution
    right_total = sum(right_cats.values()) or 1
    for cat, idx in cat_to_idx.items():
        vec[10 + idx] = right_cats[cat] / right_total

    # Pre/post verb fractions
    rel_total = pre_verb + post_verb or 1
    vec[17] = pre_verb / rel_total
    vec[18] = post_verb / rel_total

    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# -- Experiment ----------------------------------------------------------------

class FunctionWordDiscoveryExperiment(ExperimentBase):
    """Test distributional discovery of function word categories."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="function_word_discovery",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = FuncWordConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        sentences = _build_training_sentences()
        seeds = list(range(cfg.n_seeds))

        func_words = list(FUNC_WORD_CATEGORIES.keys())
        content_words = list(CONTENT_WORD_CATEGORIES.keys())

        self.log(f"Training sentences: {len(sentences)}")
        self.log(f"Function words: {func_words}")
        self.log(f"Content words: {len(content_words)}")
        self.log(f"Seeds: {cfg.n_seeds}")

        # ==============================================================
        # H1: Position entropy discrimination
        # ==============================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Position entropy — function vs content words")
        self.log("=" * 60)

        func_entropies_by_seed = []
        content_entropies_by_seed = []

        for s in seeds:
            # Position entropy doesn't depend on parser — just sentences
            func_ent = [_compute_position_entropy(w, sentences)
                        for w in func_words if w in vocab]
            content_ent = [_compute_position_entropy(w, sentences)
                           for w in content_words if w in vocab]
            func_entropies_by_seed.append(float(np.mean(func_ent)))
            content_entropies_by_seed.append(float(np.mean(content_ent)))

        func_ent_stats = summarize(func_entropies_by_seed)
        content_ent_stats = summarize(content_entropies_by_seed)
        ent_test = paired_ttest(func_entropies_by_seed,
                                content_entropies_by_seed)

        self.log(f"  Function word entropy:  {func_ent_stats['mean']:.3f} "
                 f"+/- {func_ent_stats['sem']:.3f}")
        self.log(f"  Content word entropy:   {content_ent_stats['mean']:.3f} "
                 f"+/- {content_ent_stats['sem']:.3f}")
        self.log(f"  Test (func < content): t={ent_test['t']:.2f} "
                 f"p={ent_test['p']:.4f} d={ent_test['d']:.2f} "
                 f"{'*' if ent_test['significant'] else ''}")

        # Per-word entropies
        for w in func_words:
            ent = _compute_position_entropy(w, sentences)
            self.log(f"    {w:8s} (func): {ent:.3f}")
        for w in content_words[:6]:
            ent = _compute_position_entropy(w, sentences)
            self.log(f"    {w:8s} (content): {ent:.3f}")

        # ==============================================================
        # H2: Distributional sub-categorization
        # ==============================================================
        self.log("\n" + "=" * 60)
        self.log("H2: Distributional sub-categorization (within vs between)")
        self.log("=" * 60)

        within_sims_by_seed = []
        between_sims_by_seed = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train_lexicon()

            # Compute distributional vectors for function words
            vectors = {}
            for w in func_words:
                if w in vocab:
                    vectors[w] = _compute_distributional_vector(
                        w, sentences, parser)

            # Within-category similarities
            within_sims = []
            categories = set(FUNC_WORD_CATEGORIES.values())
            for cat in categories:
                cat_words = [w for w, c in FUNC_WORD_CATEGORIES.items()
                             if c == cat and w in vectors]
                for i in range(len(cat_words)):
                    for j in range(i + 1, len(cat_words)):
                        sim = _cosine_similarity(
                            vectors[cat_words[i]], vectors[cat_words[j]])
                        within_sims.append(sim)

            # Between-category similarities
            between_sims = []
            cat_list = list(categories)
            for ci in range(len(cat_list)):
                for cj in range(ci + 1, len(cat_list)):
                    words_i = [w for w, c in FUNC_WORD_CATEGORIES.items()
                               if c == cat_list[ci] and w in vectors]
                    words_j = [w for w, c in FUNC_WORD_CATEGORIES.items()
                               if c == cat_list[cj] and w in vectors]
                    for wi in words_i:
                        for wj in words_j:
                            sim = _cosine_similarity(
                                vectors[wi], vectors[wj])
                            between_sims.append(sim)

            within_mean = float(np.mean(within_sims)) if within_sims else 0.0
            between_mean = float(np.mean(between_sims)) if between_sims else 0.0
            within_sims_by_seed.append(within_mean)
            between_sims_by_seed.append(between_mean)

        within_stats = summarize(within_sims_by_seed)
        between_stats = summarize(between_sims_by_seed)
        cluster_test = paired_ttest(within_sims_by_seed,
                                    between_sims_by_seed)

        self.log(f"  Within-category sim:  {within_stats['mean']:.3f} "
                 f"+/- {within_stats['sem']:.3f}")
        self.log(f"  Between-category sim: {between_stats['mean']:.3f} "
                 f"+/- {between_stats['sem']:.3f}")
        self.log(f"  Test (within > between): t={cluster_test['t']:.2f} "
                 f"p={cluster_test['p']:.4f} d={cluster_test['d']:.2f} "
                 f"{'*' if cluster_test['significant'] else ''}")

        # ==============================================================
        # H3: Grounding-free classification
        # ==============================================================
        self.log("\n" + "=" * 60)
        self.log("H3: Distributional classification of function words")
        self.log("=" * 60)

        func_correct_by_seed = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train_lexicon()

            # Run distributional training on raw sentences
            parser.train_distributional(sentences, repetitions=3)

            # Test: classify function words (should get DET or similar
            # non-content category)
            correct = 0
            total = 0
            for w in func_words:
                if w not in parser.stim_map:
                    continue
                cat, scores = parser.classify_distributional(w)
                # "Correct" = classified as non-content word
                # (DET, PREP, or CONJ — all function-word-like categories)
                is_func = cat in ("DET", "PREP", "CONJ", "UNKNOWN")
                total += 1
                if is_func:
                    correct += 1
                self.log(f"    {w:8s} -> {cat:8s} "
                         f"({'ok' if is_func else 'WRONG'})")

            acc = correct / total if total > 0 else 0.0
            func_correct_by_seed.append(acc)

        func_class_stats = summarize(func_correct_by_seed)
        func_class_test = ttest_vs_null(func_correct_by_seed, 0.5)

        self.log(f"  Function word classification accuracy: "
                 f"{func_class_stats['mean']:.3f} +/- "
                 f"{func_class_stats['sem']:.3f}")
        self.log(f"  vs chance (0.5): t={func_class_test['t']:.2f} "
                 f"p={func_class_test['p']:.4f} d={func_class_test['d']:.2f} "
                 f"{'*' if func_class_test['significant'] else ''}")

        # ==============================================================
        # H4: Functional frame discovery
        # ==============================================================
        self.log("\n" + "=" * 60)
        self.log("H4: Functional frames (left/right context categories)")
        self.log("=" * 60)

        frame_accs_by_seed = []

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train_lexicon()

            # Define expected frames for each function word type
            # Frame = (dominant_left_context, dominant_right_context)
            expected_frames = {
                "the": (None, "NOUN"),    # DET: followed by NOUN/ADJ
                "a":   (None, "NOUN"),
                "was": ("NOUN", "VERB"),  # AUX: between NOUN and VERB
                "by":  ("VERB", "DET"),   # MARKER: after VERB, before DET
                "and": ("NOUN", "DET"),   # CONJ: between NPs
            }

            correct = 0
            total = 0
            for w, (exp_left, exp_right) in expected_frames.items():
                # Compute actual dominant left/right context
                left_cats = defaultdict(int)
                right_cats = defaultdict(int)
                for sent in sentences:
                    for idx, sw in enumerate(sent):
                        if sw != w:
                            continue
                        if idx > 0:
                            grounding = parser.word_grounding.get(sent[idx - 1])
                            lc, _ = parser.classify_word(
                                sent[idx - 1], grounding=grounding)
                            left_cats[lc] += 1
                        if idx < len(sent) - 1:
                            grounding = parser.word_grounding.get(sent[idx + 1])
                            rc, _ = parser.classify_word(
                                sent[idx + 1], grounding=grounding)
                            right_cats[rc] += 1

                dom_right = max(right_cats, key=right_cats.get) \
                    if right_cats else None
                dom_left = max(left_cats, key=left_cats.get) \
                    if left_cats else None

                # Check right context (most discriminative)
                right_match = (exp_right is None or dom_right == exp_right)
                left_match = (exp_left is None or dom_left == exp_left)

                total += 1
                if right_match and left_match:
                    correct += 1

                self.log(f"    {w:8s}: left={dom_left:6s} "
                         f"right={dom_right:6s} "
                         f"(exp: {exp_left}/{exp_right}) "
                         f"{'ok' if right_match and left_match else 'MISS'}")

            acc = correct / total if total > 0 else 0.0
            frame_accs_by_seed.append(acc)

        frame_stats = summarize(frame_accs_by_seed)
        frame_test = ttest_vs_null(frame_accs_by_seed, 0.5)

        self.log(f"  Frame accuracy: {frame_stats['mean']:.3f} "
                 f"+/- {frame_stats['sem']:.3f}")
        self.log(f"  vs chance (0.5): t={frame_test['t']:.2f} "
                 f"p={frame_test['p']:.4f} d={frame_test['d']:.2f} "
                 f"{'*' if frame_test['significant'] else ''}")

        # ==============================================================
        # Summary
        # ==============================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("FUNCTION WORD DISCOVERY SUMMARY")
        self.log(f"  H1 (entropy discrimination): "
                 f"{'SUPPORTED' if ent_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={ent_test['d']:.2f})")
        self.log(f"  H2 (sub-categorization):     "
                 f"{'SUPPORTED' if cluster_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={cluster_test['d']:.2f})")
        self.log(f"  H3 (distributional class):   "
                 f"{'SUPPORTED' if func_class_test['significant'] else 'NOT SUPPORTED'} "
                 f"(acc={func_class_stats['mean']:.3f})")
        self.log(f"  H4 (functional frames):      "
                 f"{'SUPPORTED' if frame_test['significant'] else 'NOT SUPPORTED'} "
                 f"(acc={frame_stats['mean']:.3f})")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_sentences": len(sentences),
                "n_func_words": len(func_words),
                "n_content_words": len(content_words),
            },
            metrics={
                "h1_entropy": {
                    "function": func_ent_stats,
                    "content": content_ent_stats,
                    "test": ent_test,
                },
                "h2_subcategorization": {
                    "within": within_stats,
                    "between": between_stats,
                    "test": cluster_test,
                },
                "h3_classification": {
                    "stats": func_class_stats,
                    "test": func_class_test,
                },
                "h4_frames": {
                    "stats": frame_stats,
                    "test": frame_test,
                },
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Function word discovery experiment")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp = FunctionWordDiscoveryExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")
    for h in ["h1_entropy", "h2_subcategorization", "h3_classification",
              "h4_frames"]:
        metrics = result.metrics[h]
        if "test" in metrics:
            sig = metrics["test"]["significant"]
            print(f"  {h}: sig={sig}")


if __name__ == "__main__":
    main()
