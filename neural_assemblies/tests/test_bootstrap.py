"""Tests for emergent POS bootstrap (multi-signal fusion, exposure-driven stats)."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.acquisition import (
    infer_holdout_categories,
    classify_word_bootstrapped,
    decompose_holdout_classification,
    grounding_evidence_scores,
    ingest_holdout_sentence_stats,
    record_exposure_sentence,
    sentences_from_exposure_log,
    sentences_from_transition_paths,
    signal_confidence,
)
from neural_assemblies.assembly_calculus.emergent.core.grounding import GroundingContext

N, K = 3000, 30


class TestGroundingEvidence:
    def test_multi_modality_scores_not_dominant_only(self):
        ctx = GroundingContext(
            visual=["BIRD"],
            motor=["FINDING"],
        )
        scores = grounding_evidence_scores(ctx)
        assert "NOUN" in scores
        assert "VERB" in scores
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_properties_weight_adjective(self):
        ctx = GroundingContext(properties=["SIZE", "SMALL"])
        scores = grounding_evidence_scores(ctx)
        assert scores.get("ADJ", 0) > scores.get("NOUN", 0)


class TestEmergentFusion:
    def test_signal_confidence_uses_margin(self):
        tight = signal_confidence({"NOUN": 0.9, "VERB": 0.85})
        wide = signal_confidence({"NOUN": 0.9, "VERB": 0.1})
        assert wide > tight

    def test_exposure_boosts_distributional_weight(self):
        base = signal_confidence({"VERB": 0.6, "NOUN": 0.2}, exposure=0)
        boosted = signal_confidence({"VERB": 0.6, "NOUN": 0.2}, exposure=20)
        assert boosted > base


class TestExposureBootstrap:
    def test_exposure_log_captures_sentences(self):
        parser = EmergentParser(n=N, k=K, seed=1, fast_training=True)
        parser.register_word("dog")
        parser.register_word("runs")
        parser.ingest_raw_sentence(["the", "dog", "runs"])
        sents = sentences_from_exposure_log(parser, {"dog"})
        assert ["the", "dog", "runs"] in sents

    def test_transition_paths_reconstruct_context(self):
        parser = EmergentParser(n=N, k=K, seed=2, fast_training=True)
        for w in ("the", "small", "dog", "runs"):
            parser.register_word(w)
        parser.ingest_raw_sentence(["the", "small", "dog"])
        parser.ingest_raw_sentence(["the", "dog", "runs"])
        paths = sentences_from_transition_paths(parser, {"small"})
        assert any("small" in p for p in paths)

    def test_ingest_holdout_prefers_observed_exposure(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=3,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", "small", "bird", "runs"):
            parser.register_word(w)
        record_exposure_sentence(parser, ["the", "small", "bird"])
        record_exposure_sentence(parser, ["the", "bird", "runs"])
        n = ingest_holdout_sentence_stats(
            parser,
            {"small", "bird"},
            allow_canonical_fallback=False,
        )
        assert n >= 2
        assert parser.dist_stats.word_count.get("small", 0) > 0


class TestHoldoutBootstrap:
    def test_bootstrapped_small_adj_from_exposure(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=4,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", "small", "dog", "runs", "bird"):
            parser.register_word(w)
        parser.ingest_raw_sentence(["the", "small", "dog"])
        parser.ingest_raw_sentence(["the", "small", "bird"])
        parser.ingest_raw_sentence(["the", "bird", "runs"])
        assigned = infer_holdout_categories(parser, {"small", "bird", "finds"})
        assert assigned.get("small") == "ADJ"
        cat, _ = classify_word_bootstrapped(parser, "small")
        assert cat == "ADJ"

    def test_sentences_depth_holdout_bootstrap_floor(self, forked_parser):
        parser = forked_parser("SENTENCES", seed=42)
        decomp = decompose_holdout_classification(parser)
        assert decomp["accuracy_bootstrapped"] >= 1.0, decomp
