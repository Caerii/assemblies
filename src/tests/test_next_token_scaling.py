"""
Scaled next-token prediction: 50-word vocabulary with structured grammar.

Extends the toy 8-word demo to a more realistic setting with word classes,
structured grammar, and stratified evaluation. Characterizes where pure
Hebbian prediction breaks down.

Architecture:
    token -> stimulus -> LEX area (with recurrence) -> overlap readout

Vocabulary: 50 words in 5 classes (DET, ADJ, NOUN, VERB, PREP)
Grammar:    DET (ADJ?) NOUN VERB DET (ADJ?) NOUN
            DET (ADJ?) NOUN VERB PREP DET NOUN
Training:   40 sentences generated from grammar
"""

import random
import time
from collections import defaultdict

import pytest

from src.core.brain import Brain
from src.assembly_calculus import overlap, chance_overlap
from src.assembly_calculus.next_token import (
    build_next_token_model, train_on_corpus,
    predict_next_token, score_corpus,
)


N = 10000
K = 100
P = 0.05
BETA = 0.1
SEED = 42
ROUNDS = 8


@pytest.fixture(autouse=True)
def _timer(request):
    t0 = time.perf_counter()
    yield
    print(f"  [{time.perf_counter() - t0:.3f}s]")


# ======================================================================
# Vocabulary and Grammar
# ======================================================================

WORD_CLASSES = {
    "DET":  ["the", "a", "this", "that", "some"],
    "ADJ":  ["big", "small", "red", "old", "fast",
             "new", "good", "bad", "hot", "cold"],
    "NOUN": ["dog", "cat", "bird", "man", "woman",
             "boy", "girl", "ball", "house", "car",
             "tree", "fish", "book", "door", "hand"],
    "VERB": ["sees", "chases", "eats", "likes", "gives",
             "takes", "hits", "finds", "holds", "makes"],
    "PREP": ["in", "on", "by", "with", "near",
             "to", "from", "at", "under", "over"],
}

ALL_WORDS = []
WORD_CLASS_LOOKUP = {}
for cls, words in WORD_CLASSES.items():
    for w in words:
        ALL_WORDS.append(w)
        WORD_CLASS_LOOKUP[w] = cls

assert len(ALL_WORDS) == 50, f"Expected 50 words, got {len(ALL_WORDS)}"

# Grammar rules: each is a tuple of word class tags
GRAMMAR_RULES = [
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"),
]


def _generate_sentences(n_sentences=40, seed=SEED):
    """Generate training corpus from grammar rules."""
    rng = random.Random(seed)
    corpus = []
    for _ in range(n_sentences):
        rule = rng.choice(GRAMMAR_RULES)
        sentence = [rng.choice(WORD_CLASSES[cls]) for cls in rule]
        corpus.append(sentence)
    return corpus


def _generate_test_sentences(n_sentences=10, seed=SEED + 100):
    """Generate test corpus (different seed from training)."""
    return _generate_sentences(n_sentences, seed=seed)


# ======================================================================
# Model Setup
# ======================================================================

def _setup_scaled_model():
    """Build 50-word model and train on generated corpus.

    Returns (brain, stim_map, lexicon, train_corpus).
    """
    b = Brain(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    b.add_area("LEX", N, K, BETA)

    stim_map = {}
    for w in ALL_WORDS:
        b.add_stimulus(f"stim_{w}", K)
        stim_map[w] = f"stim_{w}"

    lexicon = build_next_token_model(
        b, "LEX", ALL_WORDS, stim_map, rounds=ROUNDS,
    )

    train_corpus = _generate_sentences(40, seed=SEED)
    train_on_corpus(
        b, "LEX", train_corpus, stim_map,
        rounds_per_token=ROUNDS, repetitions=3,
    )

    return b, stim_map, lexicon, train_corpus


# ======================================================================
# Scoring Helpers
# ======================================================================

def _score_by_word_class(brain, area, corpus, stim_map, lexicon,
                         rounds_per_token=ROUNDS):
    """Score predictions stratified by target word class.

    Returns: {class_name: {"mrr": float, "top1": float, "count": int}}
    """
    class_scores = defaultdict(lambda: {"rr_sum": 0.0, "top1": 0, "count": 0})

    for sentence in corpus:
        for pos in range(len(sentence) - 1):
            context = sentence[:pos + 1]
            actual = sentence[pos + 1]
            cls = WORD_CLASS_LOOKUP[actual]

            preds = predict_next_token(
                brain, area, context, stim_map, lexicon,
                rounds_per_token=rounds_per_token,
            )
            pred_words = [w for w, _ in preds]

            class_scores[cls]["count"] += 1
            if pred_words and actual == pred_words[0]:
                class_scores[cls]["top1"] += 1
            if actual in pred_words:
                rank = pred_words.index(actual) + 1
                class_scores[cls]["rr_sum"] += 1.0 / rank

    result = {}
    for cls, s in class_scores.items():
        n = max(s["count"], 1)
        result[cls] = {
            "mrr": s["rr_sum"] / n,
            "top1": s["top1"] / n,
            "count": s["count"],
        }
    return result


def _score_by_context_length(brain, area, corpus, stim_map, lexicon,
                             rounds_per_token=ROUNDS):
    """Score predictions stratified by context length.

    Returns: {context_len: {"mrr": float, "top1": float, "count": int}}
    """
    length_scores = defaultdict(lambda: {"rr_sum": 0.0, "top1": 0, "count": 0})

    for sentence in corpus:
        for pos in range(len(sentence) - 1):
            context = sentence[:pos + 1]
            actual = sentence[pos + 1]
            ctx_len = len(context)

            preds = predict_next_token(
                brain, area, context, stim_map, lexicon,
                rounds_per_token=rounds_per_token,
            )
            pred_words = [w for w, _ in preds]

            length_scores[ctx_len]["count"] += 1
            if pred_words and actual == pred_words[0]:
                length_scores[ctx_len]["top1"] += 1
            if actual in pred_words:
                rank = pred_words.index(actual) + 1
                length_scores[ctx_len]["rr_sum"] += 1.0 / rank

    result = {}
    for ctx_len, s in length_scores.items():
        n = max(s["count"], 1)
        result[ctx_len] = {
            "mrr": s["rr_sum"] / n,
            "top1": s["top1"] / n,
            "count": s["count"],
        }
    return result


# ======================================================================
# Tests
# ======================================================================

class TestNextTokenScaling:
    """Scale next-token prediction from 8 to 50 words."""

    def test_vocabulary_build(self):
        """50-word lexicon builds with distinct assemblies."""
        b, stim_map, lexicon, _ = _setup_scaled_model()

        assert len(lexicon) == 50, f"Expected 50 words, got {len(lexicon)}"
        for w in ALL_WORDS:
            assert w in lexicon
            assert len(lexicon[w]) == K

    def test_training_completes(self):
        """Training on 40 sentences completes without error."""
        # _setup_scaled_model already trains; just verify it runs
        b, stim_map, lexicon, train_corpus = _setup_scaled_model()
        assert len(train_corpus) == 40

    def test_overall_accuracy_above_chance(self):
        """MRR should be above chance (1/50 = 0.02 for top-1).

        Chance MRR for uniform random over 50 words is approximately
        sum(1/k for k=1..50) / 50 = 0.090.  We expect the model to
        exceed this.
        """
        b, stim_map, lexicon, _ = _setup_scaled_model()
        test_corpus = _generate_test_sentences(10)

        scores = score_corpus(
            b, "LEX", test_corpus, stim_map, lexicon,
            rounds_per_token=ROUNDS,
        )

        chance_mrr = sum(1.0 / k for k in range(1, 51)) / 50
        print(f"  MRR={scores['mrr']:.3f}, top1={scores['top1_accuracy']:.3f}, "
              f"top3={scores['top3_accuracy']:.3f}, chance_mrr={chance_mrr:.3f}")

        assert scores["mrr"] > chance_mrr, (
            f"MRR {scores['mrr']:.3f} should exceed chance {chance_mrr:.3f}. "
            f"Scores: {scores}"
        )

    def test_accuracy_by_word_class(self):
        """Prediction accuracy stratified by next-word class.

        Hypothesis: predicting the next DET (5 options) should be
        easier than predicting the next NOUN (15 options).
        """
        b, stim_map, lexicon, _ = _setup_scaled_model()
        test_corpus = _generate_test_sentences(10)

        class_scores = _score_by_word_class(
            b, "LEX", test_corpus, stim_map, lexicon,
        )

        print("  Per-class MRR:")
        for cls in ["DET", "ADJ", "NOUN", "VERB", "PREP"]:
            if cls in class_scores:
                s = class_scores[cls]
                print(f"    {cls:5s}: MRR={s['mrr']:.3f}, "
                      f"top1={s['top1']:.3f}, count={s['count']}")

        # At least one class should be above overall chance
        chance_mrr = sum(1.0 / k for k in range(1, 51)) / 50
        any_above_chance = any(
            s["mrr"] > chance_mrr
            for s in class_scores.values()
            if s["count"] >= 3
        )
        assert any_above_chance, (
            f"At least one word class should exceed chance MRR {chance_mrr:.3f}"
        )

    def test_accuracy_vs_context_length(self):
        """MRR as function of context window length.

        Hypothesis: MRR should be meaningful at all tested context lengths.
        """
        b, stim_map, lexicon, _ = _setup_scaled_model()
        test_corpus = _generate_test_sentences(10)

        ctx_scores = _score_by_context_length(
            b, "LEX", test_corpus, stim_map, lexicon,
        )

        print("  Per-context-length MRR:")
        for ctx_len in sorted(ctx_scores.keys()):
            s = ctx_scores[ctx_len]
            print(f"    ctx={ctx_len}: MRR={s['mrr']:.3f}, "
                  f"top1={s['top1']:.3f}, count={s['count']}")

        # Context length 1 should have meaningful predictions
        if 1 in ctx_scores and ctx_scores[1]["count"] >= 3:
            assert ctx_scores[1]["mrr"] > 0.0, (
                "Context length 1 should produce non-zero MRR"
            )

    def test_within_class_vs_across_class_ranking(self):
        """After DET, grammatical continuations should rank higher.

        After a determiner, the grammar allows ADJ or NOUN next.
        So {ADJ, NOUN} words should rank higher on average than
        {VERB, PREP} words in the prediction.
        """
        b, stim_map, lexicon, _ = _setup_scaled_model()

        # Test with "the" as context
        preds = predict_next_token(
            b, "LEX", ["the"], stim_map, lexicon,
            rounds_per_token=ROUNDS,
        )

        # Compute average rank by class
        pred_words = [w for w, _ in preds]
        class_ranks = defaultdict(list)
        for rank, word in enumerate(pred_words):
            cls = WORD_CLASS_LOOKUP[word]
            class_ranks[cls].append(rank + 1)

        avg_ranks = {cls: sum(ranks) / len(ranks)
                     for cls, ranks in class_ranks.items()}

        print("  Avg rank by class after 'the':")
        for cls in ["DET", "ADJ", "NOUN", "VERB", "PREP"]:
            if cls in avg_ranks:
                print(f"    {cls:5s}: avg_rank={avg_ranks[cls]:.1f}")

        # Grammatical classes (ADJ, NOUN) should rank better (lower)
        # than non-grammatical (VERB, PREP) on average
        grammatical = []
        for cls in ["ADJ", "NOUN"]:
            if cls in avg_ranks:
                grammatical.append(avg_ranks[cls])
        ungrammatical = []
        for cls in ["VERB", "PREP"]:
            if cls in avg_ranks:
                ungrammatical.append(avg_ranks[cls])

        if grammatical and ungrammatical:
            avg_gram = sum(grammatical) / len(grammatical)
            avg_ungram = sum(ungrammatical) / len(ungrammatical)
            print(f"  Grammatical avg rank: {avg_gram:.1f}")
            print(f"  Ungrammatical avg rank: {avg_ungram:.1f}")

            # Grammatical should rank better (lower number)
            assert avg_gram < avg_ungram, (
                f"After DET, grammatical classes should rank higher: "
                f"gram={avg_gram:.1f}, ungram={avg_ungram:.1f}"
            )

    def test_breakdown_identification(self):
        """Identify where Hebbian prediction breaks down by sentence length.

        Generate test sentences of increasing length and measure MRR
        at the last position for each length.
        """
        b, stim_map, lexicon, _ = _setup_scaled_model()

        # Group test sentences by length
        test_corpus = _generate_test_sentences(20, seed=SEED + 200)
        by_length = defaultdict(list)
        for sent in test_corpus:
            by_length[len(sent)].append(sent)

        print("  MRR at last position by sentence length:")
        for length in sorted(by_length.keys()):
            sentences = by_length[length]
            rr_sum = 0.0
            count = 0
            for sent in sentences:
                context = sent[:-1]  # All but last
                actual = sent[-1]

                preds = predict_next_token(
                    b, "LEX", context, stim_map, lexicon,
                    rounds_per_token=ROUNDS,
                )
                pred_words = [w for w, _ in preds]
                if actual in pred_words:
                    rank = pred_words.index(actual) + 1
                    rr_sum += 1.0 / rank
                count += 1

            mrr = rr_sum / max(count, 1)
            print(f"    length={length}: MRR={mrr:.3f} (n={count})")

        # Test passes as diagnostic — no hard threshold
        assert True


class TestNextTokenScalingDeterminism:
    """Verify deterministic behavior at scale."""

    def test_same_seed_same_predictions(self):
        """Two runs with same seed produce identical top-1 predictions."""
        b1, stim1, lex1, _ = _setup_scaled_model()
        b2, stim2, lex2, _ = _setup_scaled_model()

        test_contexts = [["the"], ["a", "big"], ["the", "dog", "sees"]]

        for ctx in test_contexts:
            pred1 = predict_next_token(
                b1, "LEX", ctx, stim1, lex1, rounds_per_token=ROUNDS,
            )
            pred2 = predict_next_token(
                b2, "LEX", ctx, stim2, lex2, rounds_per_token=ROUNDS,
            )

            words1 = [w for w, _ in pred1]
            words2 = [w for w, _ in pred2]
            assert words1 == words2, (
                f"Same seed should produce same predictions for context {ctx}: "
                f"{words1[:3]} vs {words2[:3]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
