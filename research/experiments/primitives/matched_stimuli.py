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
    return [
        (
            nouns[i % len(nouns)],
            verbs[i % len(verbs)],
            [noun for noun in nouns if noun != nouns[i % len(nouns)]][i % (len(nouns) - 1)],
            verbs[(i + 1) % len(verbs)],
            novel_nouns[i % len(novel_nouns)],
        )
        for i in range(n_triples)
    ]
