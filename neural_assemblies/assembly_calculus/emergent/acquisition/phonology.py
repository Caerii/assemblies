"""Surface-form phonology: babbling syllables and fuzzy spelling variants."""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence, Set

# Canonical early lemmas we expose with child-like surface noise.
EARLY_FUZZY_LEMMAS: Sequence[str] = (
    "dog", "cat", "mama", "dada", "ball", "go", "no", "hi",
)

_BABBLE_SYLLABLES = (
    "ba", "da", "ga", "ma", "pa", "ta", "ka", "la",
    "bi", "di", "gi", "mi", "bo", "do", "mo",
)

_DROP_VOWEL = re.compile(r"[aeiou]")


def generate_babble_forms(n: int = 24, *, seed: int = 0) -> List[str]:
    """Syllable-like forms for the pre-lexical babbling stage."""
    rng = random.Random(seed)
    pool = list(_BABBLE_SYLLABLES)
    rng.shuffle(pool)
    forms: List[str] = []
    while len(forms) < n:
        syls = rng.randint(1, 3)
        utter = "".join(rng.choice(pool) for _ in range(syls))
        if utter not in forms:
            forms.append(utter)
    return forms


def generate_babble_utterances(
    forms: Sequence[str],
    *,
    n: int = 20,
    seed: int = 0,
) -> List[List[str]]:
    """Short babbling sequences (one or two syllable forms)."""
    rng = random.Random(seed)
    out: List[List[str]] = []
    for _ in range(n):
        length = rng.randint(1, 2)
        out.append([rng.choice(forms) for _ in range(length)])
    return out


def fuzzy_variants(canonical: str, *, max_variants: int = 4) -> List[str]:
    """Child-like misspellings / reduced forms for a canonical lemma."""
    if not canonical or len(canonical) < 2:
        return []

    variants: Set[str] = {canonical}
    base = canonical.lower()

    if len(base) > 3:
        variants.add(base[:-1])
        variants.add(base[:3])
    if len(base) > 4:
        variants.add(base[:-2])

    doubled = base + base[-1]
    variants.add(doubled)

    consonant_skeleton = _DROP_VOWEL.sub("", base)
    if len(consonant_skeleton) >= 2:
        variants.add(consonant_skeleton)

    if base.endswith("g"):
        variants.add(base + "g")
    if base.endswith("s") and len(base) > 3:
        variants.add(base[:-1])

    variants.discard(canonical)
    ordered = sorted(variants, key=lambda v: (len(v), v))
    return ordered[:max_variants]


def normalize_surface_form(
    surface: str,
    surface_to_canonical: Dict[str, str],
) -> str:
    """Map a fuzzy surface form to its canonical lemma when known."""
    if surface in surface_to_canonical:
        return surface_to_canonical[surface]
    return surface
