"""
Build scaled vocabularies from lexicon data for the emergent NEMO parser.

Converts the rich lexicon entries (with domains, features, freq, AoA) into
GroundingContext objects suitable for the EmergentParser.

Domain → modality mapping:
    PERSON, ANIMAL, BODY_PART, FOOD, OBJECT, PLACE, PLANT,
    FURNITURE, NATURE, ABSTRACT        → visual
    MOTION, PERCEPTION, CONSUMPTION, CREATION, ACTION,
    COMMUNICATION, POSSESSION, COGNITION, STATE → motor
    QUALITY                             → properties
    SPACE, LOCATION                     → spatial
    SOCIAL                              → social
    TIME                                → temporal
    EMOTION                             → emotional
    FUNCTION_WORD (or no domain)        → none (DET_CORE)
"""

from typing import Dict, List, Optional, Tuple

from .core.grounding import GroundingContext

# ---- Domain → modality mapping ----

DOMAIN_TO_MODALITY = {
    # Visual (objects, beings, places)
    "PERSON": "visual",
    "ANIMAL": "visual",
    "BODY_PART": "visual",
    "FOOD": "visual",
    "PLANT": "visual",
    "OBJECT": "visual",
    "PLACE": "visual",
    "FURNITURE": "visual",
    "NATURE": "visual",
    "ABSTRACT": "visual",
    # Motor (actions, processes)
    "MOTION": "motor",
    "PERCEPTION": "motor",
    "CONSUMPTION": "motor",
    "CREATION": "motor",
    "ACTION": "motor",
    "COMMUNICATION": "motor",
    "POSSESSION": "motor",
    "COGNITION": "motor",
    "STATE": "motor",
    # Properties
    "QUALITY": "properties",
    # Spatial
    "SPACE": "spatial",
    "LOCATION": "spatial",
    # Social
    "SOCIAL": "social",
    # Temporal
    "TIME": "temporal",
    # Emotional
    "EMOTION": "emotional",
    # Function words
    "FUNCTION_WORD": "none",
}


_POS_TO_MODALITY = {
    "NOUN": "visual",
    "VERB": "motor",
    "ADJ": "properties",
    "ADV": "temporal",
    "PREP": "spatial",
    "PRON": "social",
    "DET": "none",
    "CONJ": "none",
}


def entry_to_grounding(entry: dict, pos: str) -> GroundingContext:
    """Convert a lexicon entry to a GroundingContext.

    The POS determines the primary modality (which core area the word
    assembles in).  Domains and feature keys become the grounding feature
    list within that modality — these shared features enable generalization
    across words of the same type.

    Args:
        entry: Lexicon entry dict with 'lemma', 'domains', 'features'.
        pos: Part-of-speech label (determines modality).
    """
    domains = entry.get("domains", [])
    features = entry.get("features", {})

    modality = _POS_TO_MODALITY.get(pos, "none")

    # Build grounding features from domains + feature keys
    grounding_features: List[str] = []
    for domain in domains:
        grounding_features.append(domain)
    for feat_name, feat_val in features.items():
        if feat_val is True:
            grounding_features.append(feat_name.upper())

    # Assign to the correct modality field
    if modality != "none" and grounding_features:
        return GroundingContext(**{modality: grounding_features})
    return GroundingContext()


def _select_top_entries(entries: list, max_count: Optional[int],
                        min_freq: float = 0.0) -> list:
    """Select top entries sorted by frequency (desc) then AoA (asc).

    Filters by minimum frequency and returns at most max_count entries.
    """
    filtered = [e for e in entries if e.get("freq", 0) >= min_freq]
    filtered.sort(key=lambda e: (-e.get("freq", 0), e.get("aoa", 10)))
    if max_count is not None:
        filtered = filtered[:max_count]
    return filtered


def build_vocabulary(
    max_nouns: int = 55,
    max_verbs: int = 40,
    max_adj: int = 35,
    max_adv: int = 20,
    max_prep: int = 15,
    max_pron: int = 10,
    max_det: int = 8,
    max_conj: int = 5,
) -> Dict[str, GroundingContext]:
    """Build a scaled vocabulary from the lexicon data.

    Selects the highest-frequency, lowest-AoA words from each POS category.
    Returns a dict mapping lemmas to GroundingContext, suitable for
    passing to EmergentParser(vocabulary=...).

    Args:
        max_nouns: Maximum nouns to include (default 55).
        max_verbs: Maximum verbs (default 40).
        max_adj: Maximum adjectives (default 35).
        max_adv: Maximum adverbs (default 20).
        max_prep: Maximum prepositions (default 15).
        max_pron: Maximum pronouns (default 10).
        max_det: Maximum determiners (default 8).
        max_conj: Maximum conjunctions (default 5).

    Returns:
        Dict mapping word lemmas to GroundingContext objects.
    """
    from neural_assemblies.lexicon.data import (
        NOUNS, VERBS, ADJECTIVES, ADVERBS,
        PREPOSITIONS, PRONOUNS, DETERMINERS, CONJUNCTIONS,
    )

    vocab: Dict[str, GroundingContext] = {}

    categories = [
        (NOUNS, max_nouns, "NOUN"),
        (VERBS, max_verbs, "VERB"),
        (ADJECTIVES, max_adj, "ADJ"),
        (ADVERBS, max_adv, "ADV"),
        (PREPOSITIONS, max_prep, "PREP"),
        (PRONOUNS, max_pron, "PRON"),
        (DETERMINERS, max_det, "DET"),
        (CONJUNCTIONS, max_conj, "CONJ"),
    ]

    for entries, max_count, pos in categories:
        selected = _select_top_entries(entries, max_count)
        for entry in selected:
            lemma = entry["lemma"]
            if lemma not in vocab:  # Avoid duplicates across categories
                vocab[lemma] = entry_to_grounding(entry, pos)

    return vocab


# ---- Preset vocabulary sizes for curriculum / chat ----

VOCAB_PRESETS = {
    "core": {
        "max_nouns": 0,
        "max_verbs": 0,
        "max_adj": 0,
        "max_adv": 0,
        "max_prep": 0,
        "max_pron": 0,
        "max_det": 0,
        "max_conj": 0,
    },
    "medium": {
        "max_nouns": 70,
        "max_verbs": 50,
        "max_adj": 40,
        "max_adv": 22,
        "max_prep": 15,
        "max_pron": 12,
        "max_det": 8,
        "max_conj": 6,
    },
    "large": {
        "max_nouns": 110,
        "max_verbs": 80,
        "max_adj": 60,
        "max_adv": 35,
        "max_prep": 20,
        "max_pron": 15,
        "max_det": 10,
        "max_conj": 8,
    },
    "discussion": {
        "max_nouns": 90,
        "max_verbs": 65,
        "max_adj": 50,
        "max_adv": 28,
        "max_prep": 18,
        "max_pron": 14,
        "max_det": 10,
        "max_conj": 7,
    },
}


def build_vocabulary_preset(
    name: str = "medium",
    *,
    merge_core: bool = True,
) -> Dict[str, GroundingContext]:
    """Build a named vocabulary preset for scaled training and chat.

    Presets:
        core       — closed toy ``VOCABULARY`` (~45 words)
        medium     — ~200 high-frequency lexicon words + core
        large      — ~350 words + core
        discussion — conversation-tuned subset + core

    Args:
        name: One of ``VOCAB_PRESETS`` keys.
        merge_core: When True, always include toy ``VOCABULARY`` agent words.

    Returns:
        lemma → GroundingContext
    """
    from .core.grounding import VOCABULARY

    if name not in VOCAB_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}; choose from {list(VOCAB_PRESETS)}"
        )

    if name == "core":
        return dict(VOCABULARY)

    scaled = build_vocabulary(**VOCAB_PRESETS[name])
    if not merge_core:
        return scaled

    merged = dict(VOCABULARY)
    merged.update(scaled)
    return merged


def verb_surface_form(lemma: str) -> str:
    """Return a training surface form for a verb lemma (3sg if available).

    Filters to VERB entries whose lemma IS the argument: the POS filter
    alone would resolve "saw" to see.past and return "sees", and no filter
    resolved homographs like "love" to the noun and returned the bare
    lemma, silently breaking subject agreement in generated corpora.
    """
    for entry, _pos in lookup_lexicon_entries(lemma, pos="VERB"):
        if entry["lemma"] == lemma:
            forms = entry.get("forms", {})
            return forms.get("3sg") or forms.get("present") or lemma
    return lemma


def words_by_modality(
    vocab: Dict[str, GroundingContext],
) -> Dict[str, List[str]]:
    """Partition vocabulary lemmas by dominant grounding modality."""
    buckets: Dict[str, List[str]] = {
        "visual": [],
        "motor": [],
        "properties": [],
        "spatial": [],
        "social": [],
        "temporal": [],
        "none": [],
    }
    for word, ctx in vocab.items():
        mod = ctx.dominant_modality
        if mod not in buckets:
            mod = "none"
        buckets[mod].append(word)
    return buckets

_LEXICON_INDEX: Optional[Dict[str, List[Tuple[dict, str]]]] = None


def _build_lexicon_index() -> Dict[str, List[Tuple[dict, str]]]:
    """Build a flat word → [(entry, pos), ...] index over all lexicon data.

    Indexes both lemmas and inflected forms, keeping EVERY entry that claims
    a surface form.  90 surfaces in the lexicon are claimed more than once
    ("loves" is both the plural of the noun "love" and the 3sg of the verb
    "love"; "lives" is life.plural AND live.3sg; "thought" is a noun AND
    think.past), so a single-slot index necessarily hides one reading — the
    old first-wins index made every noun-verb homograph's verbhood invisible.
    Candidates are stored in category order (NOUN, VERB, ADJ, ...), so
    ``candidates[0]`` reproduces the old first-wins winner exactly.
    """
    from neural_assemblies.lexicon.data import (
        NOUNS, VERBS, ADJECTIVES, ADVERBS,
        PREPOSITIONS, PRONOUNS, DETERMINERS, CONJUNCTIONS,
    )

    index: Dict[str, List[Tuple[dict, str]]] = {}

    categories = [
        (NOUNS, "NOUN"),
        (VERBS, "VERB"),
        (ADJECTIVES, "ADJ"),
        (ADVERBS, "ADV"),
        (PREPOSITIONS, "PREP"),
        (PRONOUNS, "PRON"),
        (DETERMINERS, "DET"),
        (CONJUNCTIONS, "CONJ"),
    ]

    for entries, pos in categories:
        for entry in entries:
            # Lemma plus inflected forms, deduped within this entry (a
            # zero-derivation form like put.past can equal the lemma).
            surfaces = [entry["lemma"]]
            for form_val in entry.get("forms", {}).values():
                if isinstance(form_val, str) and form_val:
                    surfaces.append(form_val)
            for surface in dict.fromkeys(surfaces):
                index.setdefault(surface, []).append((entry, pos))

    return index


def lookup_lexicon_entries(
    word: str, pos: Optional[str] = None,
) -> List[Tuple[dict, str]]:
    """Look up ALL lexicon entries claiming a surface form.

    This is the homograph-safe lookup: call sites that know the expected
    POS should use it with the ``pos`` filter instead of
    ``lookup_lexicon_entry``, which silently returns only the
    category-priority winner.

    Args:
        word: Word string (case-sensitive).
        pos: Optional POS label filter ("NOUN", "VERB", ...).

    Returns:
        List of (entry_dict, pos_label) candidates in category-priority
        order (NOUN before VERB before ADJ, ...); empty if unknown.
    """
    global _LEXICON_INDEX
    if _LEXICON_INDEX is None:
        _LEXICON_INDEX = _build_lexicon_index()
    candidates = _LEXICON_INDEX.get(word, [])
    if pos is not None:
        return [c for c in candidates if c[1] == pos]
    return list(candidates)


def lookup_verb_form(word: str) -> Optional[Tuple[str, str]]:
    """Look up whether a word is a verb form and identify its tense.

    Filters candidates to VERB entries, so noun-verb homographs
    ("loves", "lives", "thought") resolve to their verb reading here even
    though ``lookup_lexicon_entry`` returns the noun.

    Args:
        word: Word string to check.

    Returns:
        (lemma, tense_label) if the word is a verb form, else None.
        tense_label is one of: "PRESENT", "PAST", "PROGRESSIVE", "PERFECT".
    """
    candidates = lookup_lexicon_entries(word, pos="VERB")
    if not candidates:
        return None
    entry, _ = candidates[0]

    lemma = entry["lemma"]
    forms = entry.get("forms", {})

    # Check which form this word matches
    if word == forms.get("past"):
        return (lemma, "PAST")
    if word == forms.get("ppart"):
        return (lemma, "PERFECT")
    if word == forms.get("prog"):
        return (lemma, "PROGRESSIVE")
    # Default: present tense (lemma or 3sg)
    return (lemma, "PRESENT")


def lookup_lexicon_entry(word: str) -> Optional[Tuple[dict, str]]:
    """Look up a word in all lexicon data files.

    Checks both lemmas and inflected forms (e.g., "runs" → run entry, "VERB").

    Returns only the CATEGORY-PRIORITY winner (NOUN before VERB, ...) when
    several entries claim the surface — "loves" reads as the noun here.
    Call sites that know the expected POS must use
    ``lookup_lexicon_entries(word, pos=...)`` instead.

    Args:
        word: Word string (case-sensitive).

    Returns:
        (entry_dict, pos_label) if found, else None.
    """
    candidates = lookup_lexicon_entries(word)
    return candidates[0] if candidates else None
