"""
Base Generator
==============

Base class and utilities for sentence generators.
"""

from typing import List, Optional
import random

from ..vocabulary import get_vocabulary, WordInfo


class BaseGenerator:
    """
    Base class for sentence pattern generators.
    
    Provides common functionality:
    - Vocabulary access
    - Random selection
    - Word filtering by features
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.vocab = get_vocabulary()
        self.rng = random.Random(seed)
        
        # Cached word lists
        self._cache = {}
    
    def _get_words_by_grounding(self, grounding_type: str) -> List[WordInfo]:
        """Get words with specific grounding type."""
        cache_key = f'grounding_{grounding_type}'
        if cache_key not in self._cache:
            words = []
            for info in self.vocab.values():
                grounding_list = getattr(info.grounding, grounding_type, [])
                if grounding_list:
                    words.append(info)
            self._cache[cache_key] = words
        return self._cache[cache_key]
    
    def _get_words_by_feature(self, feature: str, value: bool = True) -> List[WordInfo]:
        """Get words with specific feature."""
        cache_key = f'feature_{feature}_{value}'
        if cache_key not in self._cache:
            words = [
                info for info in self.vocab.values()
                if info.features.get(feature) == value
            ]
            self._cache[cache_key] = words
        return self._cache[cache_key]
    
    def _get_words_by_domain(self, domain: str) -> List[WordInfo]:
        """Get words in specific domain."""
        cache_key = f'domain_{domain}'
        if cache_key not in self._cache:
            words = [
                info for info in self.vocab.values()
                if domain.upper() in [d.upper() for d in info.domains]
            ]
            self._cache[cache_key] = words
        return self._cache[cache_key]
    
    @property
    def nouns(self) -> List[WordInfo]:
        """Get all nouns (visual grounding, no motor)."""
        if 'nouns' not in self._cache:
            self._cache['nouns'] = [
                w for w in self.vocab.values()
                if w.grounding.visual and not w.grounding.motor
            ]
        return self._cache['nouns']
    
    @property
    def animate_nouns(self) -> List[WordInfo]:
        """Get animate nouns."""
        if 'animate_nouns' not in self._cache:
            self._cache['animate_nouns'] = [
                n for n in self.nouns if n.is_animate
            ]
        return self._cache['animate_nouns']
    
    @property
    def inanimate_nouns(self) -> List[WordInfo]:
        """Get inanimate nouns."""
        if 'inanimate_nouns' not in self._cache:
            self._cache['inanimate_nouns'] = [
                n for n in self.nouns if not n.is_animate
            ]
        return self._cache['inanimate_nouns']
    
    @property
    def verbs(self) -> List[WordInfo]:
        """Get all verbs (motor or cognitive grounding)."""
        if 'verbs' not in self._cache:
            self._cache['verbs'] = [
                w for w in self.vocab.values()
                if w.grounding.motor or w.grounding.cognitive
            ]
        return self._cache['verbs']
    
    @property
    def intransitive_verbs(self) -> List[WordInfo]:
        """Get intransitive verbs."""
        if 'intransitive_verbs' not in self._cache:
            self._cache['intransitive_verbs'] = [
                v for v in self.verbs if v.is_intransitive
            ]
        return self._cache['intransitive_verbs']
    
    @property
    def transitive_verbs(self) -> List[WordInfo]:
        """Get transitive verbs."""
        if 'transitive_verbs' not in self._cache:
            self._cache['transitive_verbs'] = [
                v for v in self.verbs if v.is_transitive
            ]
        return self._cache['transitive_verbs']
    
    @property
    def cognitive_verbs(self) -> List[WordInfo]:
        """Get cognitive verbs."""
        if 'cognitive_verbs' not in self._cache:
            self._cache['cognitive_verbs'] = [
                v for v in self.verbs 
                if 'COGNITION' in [d.upper() for d in v.domains]
            ]
        return self._cache['cognitive_verbs']
    
    @property
    def adjectives(self) -> List[WordInfo]:
        """Get all adjectives (property grounding, no motor)."""
        if 'adjectives' not in self._cache:
            self._cache['adjectives'] = [
                w for w in self.vocab.values()
                if w.grounding.properties and not w.grounding.motor
            ]
        return self._cache['adjectives']
    
    @property
    def pronouns(self) -> List[WordInfo]:
        """Get all pronouns (social grounding, no visual)."""
        if 'pronouns' not in self._cache:
            self._cache['pronouns'] = [
                w for w in self.vocab.values()
                if w.grounding.social and not w.grounding.visual
            ]
        return self._cache['pronouns']
    
    @property
    def self_pronouns(self) -> List[WordInfo]:
        """Get self-referential pronouns (I, you, we)."""
        if 'self_pronouns' not in self._cache:
            self._cache['self_pronouns'] = [
                self.vocab.get('i'),
                self.vocab.get('you'),
            ]
            self._cache['self_pronouns'] = [p for p in self._cache['self_pronouns'] if p]
        return self._cache['self_pronouns']
    
    def random_choice(self, items: List) -> Optional[any]:
        """Safely choose random item from list."""
        if not items:
            return None
        return self.rng.choice(items)


__all__ = ['BaseGenerator']


