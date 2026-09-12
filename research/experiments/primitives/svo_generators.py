"""Shared generators for primitive language experiments.

The SVO protocol is identical across the primitive ERP studies; vocabularies
remain caller-owned so each study keeps its declared stimulus inventory.
"""

from typing import Sequence
import numpy as np

def generate_svo_sentences(n_sentences: int, rng: np.random.Generator,
                           nouns: Sequence[str], verbs: Sequence[str]) -> list[tuple[str, str, str]]:
    """Draw ``(agent, verb, patient)`` triples without self-patient pairs."""
    if n_sentences < 0:
        raise ValueError("n_sentences must be nonnegative")
    if len(nouns) < 2 or not verbs:
        raise ValueError("SVO generation requires at least two nouns and one verb")
    return [
        (agent := str(rng.choice(nouns)),
         str(rng.choice(verbs)),
         str(rng.choice([noun for noun in nouns if noun != agent])))
        for _ in range(n_sentences)
    ]
