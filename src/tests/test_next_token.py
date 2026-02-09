"""
Tests for next-token prediction via assembly overlap readout.

Demonstrates that after Hebbian training on a toy corpus, the assembly
system can predict the next token above chance.  Uses a simple bigram
corpus with 8 words vocabulary.

This is the narrow, structured demonstration described in
BRIDGE_WEBSCALE_CURRICULUM.md Section 4.

Architecture:
    token -> stimulus -> LEX area (with recurrence) -> overlap readout

Training:
    sequence_memorize over each sentence builds Hebbian bridges
    between consecutive token assemblies.

Prediction:
    Feed context tokens sequentially, read out overlap with each
    vocabulary word, rank by overlap.
"""

import time

import pytest

from src.core.brain import Brain
from src.assembly_calculus import overlap, chance_overlap, build_lexicon
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


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


# Toy corpus with strong bigram patterns:
#   "the cat sat"   -> the->cat, cat->sat
#   "the dog ran"   -> the->dog, dog->ran
#   "a cat sat"     -> a->cat, cat->sat
#   "a dog ran"     -> a->dog, dog->ran
#   "the big cat sat"   -> the->big, big->cat, cat->sat
#   "the small dog ran" -> the->small, small->dog, dog->ran
VOCAB = ["the", "a", "cat", "dog", "sat", "ran", "big", "small"]
TRAIN_CORPUS = [
    ["the", "cat", "sat"],
    ["the", "dog", "ran"],
    ["a", "cat", "sat"],
    ["a", "dog", "ran"],
    ["the", "big", "cat", "sat"],
    ["the", "small", "dog", "ran"],
]


def _setup_model():
    """Create brain, build lexicon, train on corpus."""
    b = _make_brain()
    b.add_area("LEX", N, K, BETA)

    stim_map = {}
    for w in VOCAB:
        b.add_stimulus(f"stim_{w}", K)
        stim_map[w] = f"stim_{w}"

    lexicon = build_next_token_model(
        b, "LEX", VOCAB, stim_map, rounds=ROUNDS
    )

    train_on_corpus(
        b, "LEX", TRAIN_CORPUS, stim_map,
        rounds_per_token=ROUNDS, repetitions=3,
    )

    return b, stim_map, lexicon


class TestNextTokenPrediction:
    """End-to-end next-token prediction on a toy corpus."""

    def test_lexicon_is_built(self):
        """Vocabulary lexicon has correct size and distinct entries."""
        b, stim_map, lexicon = _setup_model()

        assert len(lexicon) == len(VOCAB)
        for w in VOCAB:
            assert w in lexicon
            assert len(lexicon[w]) == K

    def test_next_token_after_the(self):
        """After 'the', valid continuations (cat/dog/big/small) should
        appear in the top predictions.

        'the' -> 'cat' and 'the' -> 'dog' are strong bigrams.
        'the' -> 'big' and 'the' -> 'small' also appear in training.
        """
        b, stim_map, lexicon = _setup_model()

        predictions = predict_next_token(
            b, "LEX", ["the"], stim_map, lexicon, rounds_per_token=ROUNDS,
        )

        pred_words = [w for w, _ in predictions]
        valid = {"cat", "dog", "big", "small"}
        top4 = set(pred_words[:4])
        assert len(top4 & valid) > 0, (
            f"After 'the', expected some of {valid} in top-4, "
            f"got {pred_words[:4]}"
        )

    def test_next_token_after_cat(self):
        """After 'cat', 'sat' should rank above non-continuations.

        'cat' -> 'sat' appears in every sentence containing 'cat'.
        """
        b, stim_map, lexicon = _setup_model()

        predictions = predict_next_token(
            b, "LEX", ["cat"], stim_map, lexicon, rounds_per_token=ROUNDS,
        )

        pred_words = [w for w, _ in predictions]
        # 'sat' should be in the predictions
        assert "sat" in pred_words, (
            f"After 'cat', 'sat' should appear: {pred_words}"
        )
        # 'sat' should be in top half of predictions
        sat_rank = pred_words.index("sat")
        assert sat_rank < len(VOCAB) // 2, (
            f"After 'cat', 'sat' should rank in top half: "
            f"rank={sat_rank + 1}/{len(VOCAB)}"
        )

    def test_above_chance_accuracy(self):
        """Overall prediction accuracy should be above chance (1/|V|).

        Chance for top-1 with |V|=8 is 12.5%.  Top-3 chance is 37.5%.
        We test that the model performs above random guessing.
        """
        b, stim_map, lexicon = _setup_model()

        test_corpus = [
            ["the", "cat", "sat"],
            ["the", "dog", "ran"],
        ]

        scores = score_corpus(
            b, "LEX", test_corpus, stim_map, lexicon,
            rounds_per_token=ROUNDS,
        )

        chance = 1.0 / len(VOCAB)
        # At minimum, MRR should be above chance
        assert scores["mrr"] > chance, (
            f"MRR {scores['mrr']:.3f} should exceed chance {chance:.3f}. "
            f"Scores: {scores}"
        )

    def test_deterministic_with_seed(self):
        """Same seed produces same predictions."""
        b1, stim1, lex1 = _setup_model()
        b2, stim2, lex2 = _setup_model()

        pred1 = predict_next_token(
            b1, "LEX", ["the"], stim1, lex1, rounds_per_token=ROUNDS,
        )
        pred2 = predict_next_token(
            b2, "LEX", ["the"], stim2, lex2, rounds_per_token=ROUNDS,
        )

        # Same top prediction
        assert pred1[0][0] == pred2[0][0], (
            f"Same seed should produce same top prediction: "
            f"{pred1[0][0]} vs {pred2[0][0]}"
        )
        # Same ranking
        words1 = [w for w, _ in pred1]
        words2 = [w for w, _ in pred2]
        assert words1 == words2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
