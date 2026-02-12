"""
Training data builders for N400/P600 experiments.

Two training paradigms:
- build_priming_pairs(): SVO sentences for word-pair priming experiments
- build_svo_sentences(): SVO sentences with role annotations for parser training
"""

from typing import Dict, List

from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


def build_priming_pairs(vocab: Dict[str, GroundingContext]) -> List[GroundedSentence]:
    """SVO training sentences for word-pair priming experiments.

    Creates sentences with animals as subjects/objects and occasional
    cross-category objects (animals + tools/objects). Establishes Hebbian
    connections between co-occurring assemblies.

    Used by: test_n400_pre_kwta, test_n400_controls, test_n400_parameter_sweep
    """
    def ctx(w):
        return vocab[w]

    sentences = []
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


def build_svo_sentences(vocab: Dict[str, GroundingContext]) -> List[GroundedSentence]:
    """SVO sentences with role annotations for parser syntactic training.

    Creates sentences with animal-only subjects and objects, repeated 3x for
    the core set plus additional diversity sentences. All objects are animals
    (no furniture/tools), so these nouns get trained in structural pathways.

    Used by: test_p600_syntactic
    """
    def ctx(w):
        return vocab[w]

    sentences = []
    for _ in range(3):
        for subj, verb, obj in [
            ("dog", "chases", "cat"),
            ("cat", "chases", "bird"),
            ("bird", "sees", "fish"),
            ("horse", "chases", "dog"),
            ("dog", "sees", "bird"),
            ("cat", "sees", "horse"),
        ]:
            sentences.append(GroundedSentence(
                words=["the", subj, verb, "the", obj],
                contexts=[ctx("the"), ctx(subj), ctx(verb),
                          ctx("the"), ctx(obj)],
                roles=[None, "agent", "action", None, "patient"],
            ))

    for subj, verb, obj in [
        ("fish", "sees", "mouse"),
        ("horse", "finds", "cat"),
        ("dog", "finds", "mouse"),
        ("bird", "chases", "horse"),
        ("mouse", "sees", "dog"),
        ("cat", "finds", "fish"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences
