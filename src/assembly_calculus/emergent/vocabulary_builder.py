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

from .grounding import GroundingContext

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
    from src.lexicon.data import (
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


# ---- Lexicon lookup (lazy-initialized) ----

_LEXICON_INDEX: Optional[Dict[str, Tuple[dict, str]]] = None


def _build_lexicon_index() -> Dict[str, Tuple[dict, str]]:
    """Build a flat word → (entry, pos) index over all lexicon data.

    Indexes both lemmas and inflected forms.
    """
    from src.lexicon.data import (
        NOUNS, VERBS, ADJECTIVES, ADVERBS,
        PREPOSITIONS, PRONOUNS, DETERMINERS, CONJUNCTIONS,
    )

    index: Dict[str, Tuple[dict, str]] = {}

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
            lemma = entry["lemma"]
            if lemma not in index:
                index[lemma] = (entry, pos)
            # Index inflected forms too
            for form_name, form_val in entry.get("forms", {}).items():
                if isinstance(form_val, str) and form_val not in index:
                    index[form_val] = (entry, pos)

    return index


def lookup_verb_form(word: str) -> Optional[Tuple[str, str]]:
    """Look up whether a word is a verb form and identify its tense.

    Args:
        word: Word string to check.

    Returns:
        (lemma, tense_label) if the word is a verb form, else None.
        tense_label is one of: "PRESENT", "PAST", "PROGRESSIVE", "PERFECT".
    """
    result = lookup_lexicon_entry(word)
    if result is None:
        return None
    entry, pos = result
    if pos != "VERB":
        return None

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

    Args:
        word: Word string (case-sensitive).

    Returns:
        (entry_dict, pos_label) if found, else None.
    """
    global _LEXICON_INDEX
    if _LEXICON_INDEX is None:
        _LEXICON_INDEX = _build_lexicon_index()
    return _LEXICON_INDEX.get(word)
