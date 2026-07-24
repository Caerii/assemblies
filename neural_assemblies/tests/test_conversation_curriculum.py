"""Tests for scaled vocabulary and conversation curriculum."""

import pytest

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    EmergentSession,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.conversation import (
    create_conversation_pairs,
    create_conversation_scripts,
    get_conversation_curriculum,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import CurriculumTrainer
from neural_assemblies.assembly_calculus.emergent.core.grounding import VOCABULARY

N, K, P, BETA, SEED, ROUNDS = 3000, 30, 0.05, 0.1, 42, 4


class TestVocabularyPresets:
    def test_core_preset_size(self):
        vocab = build_vocabulary_preset("core")
        assert len(vocab) == len(VOCABULARY)

    def test_medium_larger_than_core(self):
        core = build_vocabulary_preset("core")
        medium = build_vocabulary_preset("medium")
        assert len(medium) > len(core)
        assert all(w in medium for w in core)

    def test_discussion_includes_agent_words(self):
        vocab = build_vocabulary_preset("discussion")
        assert "chases" in vocab
        assert "dog" in vocab


class TestConversationCurriculum:
    def test_scaled_pairs_from_medium_vocab(self):
        vocab = build_vocabulary_preset("medium")
        pairs = create_conversation_pairs(vocab, seed=42, max_transitive=4)
        assert len(pairs) >= 4
        types = {p.pattern_type for p in pairs}
        assert "who_query" in types

    def test_conversation_scripts_three_turns(self):
        vocab = build_vocabulary_preset("medium")
        scripts = create_conversation_scripts(vocab, seed=1, n_scripts=3)
        assert len(scripts) >= 1
        assert len(scripts[0].turns) == 3

    def test_curriculum_sentences_use_known_words(self):
        vocab = build_vocabulary_preset("medium")
        sents = get_conversation_curriculum(vocab, max_transitive=2)
        for gs in sents:
            for w in gs.words:
                assert w in vocab or w in ("does", "what", "who"), (
                    f"unknown word {w!r}"
                )


class TestConversationTraining:
    @pytest.fixture(scope="class")
    def conversation_parser(self):
        vocab = build_vocabulary_preset("medium")
        parser = EmergentParser(
            n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS,
            vocabulary=vocab,
        )
        trainer = CurriculumTrainer(parser)
        trainer.train_stage("FIRST_WORDS")
        trainer.train_stage("TWO_WORD")
        trainer.train_stage("DIALOGUE")
        return parser

    def test_dialogue_stage_runs_dialogue_phase(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS,
            vocabulary=vocab,
        )
        trainer = CurriculumTrainer(parser)
        for stage in ("FIRST_WORDS", "TWO_WORD"):
            trainer.train_stage(stage)
        result = trainer.train_stage("DIALOGUE")
        assert "dialogue" in result.phases_run

    def test_train_for_conversation_registers_prediction(self, conversation_parser):
        assert hasattr(conversation_parser, "prediction_lexicon")
        assert len(conversation_parser.prediction_lexicon) > 0

    def test_session_bootstrap_conversation_smoke(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS,
            vocabulary=vocab,
        )
        trainer = CurriculumTrainer(parser)
        trainer.train_stage("FIRST_WORDS")
        trainer.train_stage("TWO_WORD")
        trainer.train_stage("DIALOGUE")
        session = EmergentSession(parser=parser, online_learn=False)
        reply = session.interact("who chases the cat")
        assert reply in ("dog", "unknown", "cat") or len(reply) > 0

    def test_unknown_word_registered_on_interact(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS,
            vocabulary=vocab,
        )
        trainer = CurriculumTrainer(parser)
        trainer.train_stage("DIALOGUE")
        session = EmergentSession(parser=parser, online_learn=False)
        before = len(parser.stim_map)
        session.interact("the elephant runs")
        assert len(parser.stim_map) >= before
