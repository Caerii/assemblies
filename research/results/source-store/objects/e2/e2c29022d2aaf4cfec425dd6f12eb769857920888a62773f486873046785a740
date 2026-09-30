"""Shared generators for primitive language experiments.

The SVO protocol is identical across the primitive ERP studies; vocabularies
remain caller-owned so each study keeps its declared stimulus inventory.
"""

import numpy as np
from typing import Sequence

def generate_svo_sentences(n_sentences: int, rng: np.random.Generator,
                           nouns: Sequence[str], verbs: Sequence[str]) -> list[tuple[str, str, str]]:
    """Draw ``(agent, verb, patient)`` triples without self-patient pairs."""
    if n_sentences < 0:
        raise ValueError("n_sentences must be nonnegative")
    if len(nouns) < 2 or not verbs:
        raise ValueError("SVO generation requires at least two nouns and one verb")
    sentences = []
    for _ in range(n_sentences):
        agent = str(rng.choice(nouns))
        verb = str(rng.choice(verbs))
        eligible_patients = [noun for noun in nouns if noun != agent]
        patient = str(rng.choice(eligible_patients))
        sentences.append((agent, verb, patient))
    return sentences
