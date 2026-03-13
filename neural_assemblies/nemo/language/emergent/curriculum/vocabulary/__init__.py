"""
Vocabulary Module
=================

Provides word definitions with grounding information.

Structure:
- types.py: Core data types (WordInfo)
- grounding.py: Domain-to-grounding mapping
- loader.py: Vocabulary loading from lexicon

Key insight: The grounding context determines the emergent category.
- VISUAL grounding → NOUN
- MOTOR grounding → VERB  
- PROPERTY grounding → ADJECTIVE
- SPATIAL grounding → PREPOSITION
- SOCIAL grounding → PRONOUN
- TEMPORAL grounding → ADVERB
- COGNITIVE grounding → COGNITIVE VERB
- No grounding → FUNCTION WORD
"""

from typing import Dict, Optional

from .types import WordInfo
from .grounding import (
    create_grounding_from_domains,
    create_pronoun_grounding,
    create_question_word_grounding,
    DOMAIN_MODALITY_MAP,
)
from .loader import load_from_lexicon, load_basic_vocabulary


# Cached vocabulary
_vocabulary_cache: Optional[Dict[str, WordInfo]] = None


def get_vocabulary(category: Optional[str] = None, 
                   use_cache: bool = True) -> Dict[str, WordInfo]:
    """
    Get vocabulary with grounding information.
    
    Args:
        category: Optional filter for word category (NOUN, VERB, etc.)
        use_cache: Whether to use cached vocabulary
    
    Returns:
        Dictionary mapping lemmas to WordInfo
    """
    global _vocabulary_cache
    
    # Load vocabulary if not cached
    if _vocabulary_cache is None or not use_cache:
        # Try lexicon first, fall back to basic
        _vocabulary_cache = load_from_lexicon()
        if not _vocabulary_cache:
            _vocabulary_cache = load_basic_vocabulary()
    
    # Filter by category if specified
    if category is None:
        return _vocabulary_cache.copy()
    
    # Filter based on grounding (which determines category)
    filtered = {}
    for lemma, info in _vocabulary_cache.items():
        if _matches_category(info, category):
            filtered[lemma] = info
    
    return filtered


def _matches_category(info: WordInfo, category: str) -> bool:
    """Check if word matches the specified category based on grounding."""
    g = info.grounding
    
    if category == 'NOUN':
        return bool(g.visual) and not bool(g.motor)
    elif category == 'VERB':
        return bool(g.motor) or bool(g.cognitive)
    elif category == 'ADJECTIVE':
        return bool(g.properties) and not bool(g.motor)
    elif category == 'ADVERB':
        return bool(g.temporal) and not bool(g.visual)
    elif category == 'PRONOUN':
        return bool(g.social) and not bool(g.visual)
    elif category == 'PREPOSITION':
        return bool(g.spatial) and not bool(g.motor)
    elif category == 'FUNCTION':
        return not any([g.visual, g.motor, g.properties, g.spatial, g.social, g.temporal, g.cognitive])
    elif category == 'QUESTION':
        return 'QUESTION' in info.domains
    else:
        return False


def get_word_info(word: str) -> Optional[WordInfo]:
    """Get information about a specific word."""
    vocab = get_vocabulary()
    return vocab.get(word.lower())


def clear_cache():
    """Clear the vocabulary cache."""
    global _vocabulary_cache
    _vocabulary_cache = None


# Export
__all__ = [
    'WordInfo',
    'get_vocabulary',
    'get_word_info',
    'clear_cache',
    'create_grounding_from_domains',
    'create_pronoun_grounding',
    'create_question_word_grounding',
    'DOMAIN_MODALITY_MAP',
]
