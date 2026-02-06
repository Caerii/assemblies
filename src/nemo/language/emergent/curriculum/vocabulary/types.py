"""
Vocabulary Types
================

Core data types for vocabulary representation.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

from ...params import GroundingContext


@dataclass
class WordInfo:
    """
    Complete information about a word for NEMO learning.
    
    Attributes:
        lemma: Base form of the word
        forms: Inflected forms (e.g., {"plural": "dogs", "3sg": "runs"})
        grounding: Sensory-motor grounding context
        domains: Semantic domains (e.g., ["ANIMAL", "PET"])
        features: Semantic features (e.g., {"animate": True})
        frequency: Log frequency in language
        aoa: Age of acquisition (years)
    """
    lemma: str
    forms: Dict[str, str] = field(default_factory=dict)
    grounding: GroundingContext = field(default_factory=GroundingContext)
    domains: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    frequency: float = 1.0
    aoa: float = 3.0
    
    def get_all_forms(self) -> List[str]:
        """Get all word forms including lemma."""
        forms = [self.lemma]
        forms.extend(self.forms.values())
        return list(set(forms))
    
    def get_form(self, form_name: str) -> str:
        """Get a specific form, defaulting to lemma."""
        return self.forms.get(form_name, self.lemma)
    
    @property
    def is_animate(self) -> bool:
        """Check if word refers to animate entity."""
        return self.features.get('animate', False)
    
    @property
    def is_human(self) -> bool:
        """Check if word refers to human."""
        return self.features.get('human', False)
    
    @property
    def is_transitive(self) -> bool:
        """Check if verb is transitive."""
        return self.features.get('transitive', False)
    
    @property
    def is_intransitive(self) -> bool:
        """Check if verb is intransitive."""
        return self.features.get('intransitive', False)


__all__ = ['WordInfo']


