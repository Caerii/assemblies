"""Matched ERP stimulus constructors shared by primitive studies."""

from collections.abc import Sequence
import numpy as np

def generate_test_triples(
    rng: np.random.Generator,
    n_triples: int,
    nouns: Sequence[str],
    verbs: Sequence[str],
    novel_nouns: Sequence[str],
) -> list[tuple[str, str, str, str, str]]:
    """Construct deterministic matched triples for a shared context.

    ``rng`` remains in the signature for compatibility with preregistered
    callers; the protocol deliberately indexes vocabularies, so seed changes
    do not reorder matched conditions.
    """
    del rng
    if isinstance(n_triples, bool) or not isinstance(n_triples, int) or n_triples < 0:
        raise ValueError("n_triples must be a nonnegative integer")
    if len(nouns) < 2 or not verbs or not novel_nouns:
        raise ValueError("matched triples require two nouns, one verb, and one novel noun")
    triples = []
    for index in range(n_triples):
        agent = nouns[index % len(nouns)]
        verb = verbs[index % len(verbs)]
        eligible_patients = [noun for noun in nouns if noun != agent]
        grammatical_object = eligible_patients[index % len(eligible_patients)]
        category_violation = verbs[(index + 1) % len(verbs)]
        novel_object = novel_nouns[index % len(novel_nouns)]
        triples.append((agent, verb, grammatical_object,
                        category_violation, novel_object))
    return triples
