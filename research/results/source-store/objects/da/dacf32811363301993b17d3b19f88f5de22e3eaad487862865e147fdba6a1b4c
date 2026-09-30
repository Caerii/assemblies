"""Tests for large-corpus training and novel sentence generation."""

import os


os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.session.novel_chat import (
    build_chat_corpus,
    register_corpus_memory,
    train_for_novel_chat,
)
from neural_assemblies.assembly_calculus.emergent.session.interactive import EmergentSession

N, K = 3000, 30


class TestNovelGeneration:
    def test_continue_sentence_extends_prefix(self):
        parser = EmergentParser(n=N, k=K, seed=42, fast_training=True)
        parser.train(train_prediction=True)
        parser._ensure_prediction_lexicon()

        out = parser.continue_sentence(["the", "dog"], max_words=8, min_words=3)
        assert out[:2] == ["the", "dog"]
        assert len(out) >= 3

    def test_sentence_novelty_known_vs_unknown(self):
        parser = EmergentParser(n=N, k=K, seed=43, fast_training=True)
        corpus = build_chat_corpus(build_vocabulary_preset("core"), n_sentences=20)
        register_corpus_memory(parser, corpus)
        known = list(corpus[0].words)
        assert parser.sentence_novelty(known) == 0.0
        assert parser.sentence_novelty(["xyz", "runs", "away"]) == 1.0

    def test_generate_novel_sentence_returns_words(self):
        parser = EmergentParser(n=N, k=K, seed=44, fast_training=True)
        parser.train(train_prediction=True)
        parser._ensure_prediction_lexicon()
        sents = build_chat_corpus(parser.word_grounding, n_sentences=30)
        register_corpus_memory(parser, sents)

        novel = parser.generate_novel_sentence(
            seed_prefix=["the"], min_words=3, max_words=7, max_attempts=3,
        )
        assert len(novel) >= 3
        assert all(w in parser.stim_map for w in novel)


class TestNovelChatTraining:
    def test_train_for_novel_chat_smoke(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=45, vocabulary=vocab, fast_training=True,
        )
        summary = train_for_novel_chat(
            parser,
            n_corpus_sentences=40,
            max_stage="DIALOGUE",
            skip_early_curriculum=True,
        )
        assert summary["corpus_sentences"] >= 40
        assert summary["vocab_size"] >= len(vocab)
        assert hasattr(parser, "prediction_lexicon")
        assert len(parser.prediction_lexicon) > 0

    def test_bootstrap_novel_chat_session(self):
        session = EmergentSession.bootstrap_novel_chat(
            preset="core",
            n_corpus_sentences=35,
            max_stage="DIALOGUE",
            n=N,
            k=K,
            seed=46,
        )
        reply = session.interact("the dog runs")
        assert isinstance(reply, str)
        assert len(reply.split()) >= 1
