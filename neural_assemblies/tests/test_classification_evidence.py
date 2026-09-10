"""Fallbacks must retain their source and never masquerade as neural evidence."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference import (
    classify_word_bootstrapped, decompose_word_classification,
)
from neural_assemblies.assembly_calculus.emergent.core.classification import ClassificationEvidence
from neural_assemblies.assembly_calculus.emergent.core.grounding import GroundingContext
from neural_assemblies.assembly_calculus.emergent.parser_mixins.classify import CategoryClassificationMixin
from neural_assemblies.assembly_calculus.emergent.parser_mixins.distributional import DistributionalMixin


class FallbackParser(CategoryClassificationMixin):
    """No neural cue or lexicon drive; real classifier dispatch must fall back."""
    def __init__(self):
        self.stim_map = {}
        self.core_lexicons = {}
        self.word_grounding = {"word": GroundingContext(visual=["ANIMAL"])}
        self.dist_stats = SimpleNamespace(
            word_count={"word": 4}, word_as_pre_verb={}, word_as_post_verb={}, word_as_action={})
        self._grounding_stim_names_set = set()
        self.brain = SimpleNamespace(read_only=nullcontext)

    def _grounding_stim_names(self, ctx):
        return []

    def classify_distributional(self, word):
        return "VERB", {"VERB": 0.8, "NOUN": 0.2}


def test_lexicon_fallback_keeps_category_scores_and_real_source():
    parser = FallbackParser()
    parser.core_lexicons = {"VERB_CORE": {"word": object()}}
    category, scores = classify_word_bootstrapped(parser, "word")
    assert category == "VERB"
    assert scores["_source"] == "distributional"
    assert scores["VERB"] == 0.8
    assert scores["_confidence"] > 0


def test_fallback_is_not_fused_or_reported_as_neural_evidence():
    parser = FallbackParser()
    _, scores = classify_word_bootstrapped(parser, "word")
    assert "_weight_distributional" in scores
    assert "_weight_neural" not in scores
    assert scores["_neural"] == "UNKNOWN"
    report = decompose_word_classification(parser, "word", "VERB")
    assert report["classification_source"] == "distributional"
    assert report["neural_readout"] == "UNKNOWN"
    assert report["neural_by_category"] == {}
    assert not report["correct_neural"]
    assert report["correct_distributional"]


def test_strong_wrong_neural_signal_is_not_diagnosed_as_weak(monkeypatch):
    parser = FallbackParser()
    parser.classify_word_evidence = lambda *a, **k: ClassificationEvidence(
        "VERB", "neural", {"VERB_CORE": 0.8})
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference.classify_word_bootstrapped",
        lambda *a, **k: ("VERB", {"VERB": 1.0}))
    report = decompose_word_classification(parser, "word", "NOUN")
    assert report["failure_mode"] == "both_signals_wrong"


@pytest.mark.parametrize("source,scores", [
    ("neural", {"NOUN": 1.0}), ("distributional", {"NOUN_CORE": 1.0}),
    ("neural", {"NOUN_CORE": float("nan")}), ("distributional", {"NOUN": -1.0}),
    ("invented", {}), ("none", {"NOUN": 1.0}),
])
def test_wrong_domain_or_invalid_scores_are_rejected(source, scores):
    with pytest.raises(ValueError):
        ClassificationEvidence("UNKNOWN", source, scores)


def test_source_conversion_is_explicit_and_does_not_alias_inputs():
    raw = {"NOUN_CORE": 0.7}
    evidence = ClassificationEvidence("NOUN", "neural", raw)
    raw.clear()
    assert evidence.category_scores() == {"NOUN": 0.7}
    assert evidence.source == "neural"
    _, legacy = evidence.as_legacy_tuple()
    legacy.clear()
    assert evidence.scores["NOUN_CORE"] == 0.7


def test_legacy_classifier_retains_its_fallback_tuple():
    assert FallbackParser().classify_word("word") == ("VERB", {"VERB": 0.8, "NOUN": 0.2})


@pytest.mark.parametrize("grounded,confidence", [(True, 0.8), (False, 0.2), (False, 0.8)])
def test_frame_subcategory_is_converted_before_distributional_scoring(grounded, confidence):
    parser = FallbackParser()
    if not grounded:
        parser.word_grounding.clear()
    parser.classify_by_frame = lambda word: ("AUX", confidence)
    parser.dist_stats.position_counts = {}
    parser.dist_stats.transitions = {}
    category, scores = DistributionalMixin.classify_distributional(parser, "word")
    assert category == "DET"
    assert set(scores) == {"DET"}



def test_cue_provenance_is_immutable_and_legacy_view_stays_compatible():
    cues = ['phon_word']
    evidence = ClassificationEvidence('NOUN', 'neural', {'NOUN_CORE': .5},
                                      cue_mode='phon_only', cues=cues)
    cues.clear()
    assert evidence.cues == ('phon_word',)
    assert evidence.as_legacy_tuple() == ('NOUN', {'NOUN_CORE': .5})


@pytest.mark.parametrize('mode,cues', [(None, ['phon_word']), ('typo', []),
                                      ('combined', ['s', 's']), ('combined', 's')])
def test_malformed_cue_provenance_is_rejected(mode, cues):
    with pytest.raises(ValueError):
        ClassificationEvidence('UNKNOWN', 'neural', {}, cue_mode=mode, cues=cues)
