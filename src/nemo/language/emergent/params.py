"""
Emergent NEMO Parameters and Data Classes
==========================================

Version: 2.0.0
Date: 2025-12-01

Parameter classes and grounding context definitions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from enum import Enum

__all__ = ['EmergentParams', 'GroundingContext', 'GroundingModality', 'GroundedSentence']


class GroundingModality(Enum):
    """Sensory modalities for grounding"""
    VISUAL = 0      # Objects, scenes → NOUN
    MOTOR = 1       # Actions, movements → VERB
    PROPERTY = 2    # Properties (color, size) → ADJECTIVE
    SPATIAL = 3     # Locations → PREPOSITION
    SOCIAL = 4      # People → PRONOUN
    TEMPORAL = 5    # Time → ADVERB
    EMOTION = 6     # Emotions → ADJ/ADV
    COGNITIVE = 7   # Mental states → COGNITIVE VERB
    NONE = 8        # No grounding → FUNCTION WORD


@dataclass
class GroundingContext:
    """
    Grounding context for a word.
    
    In real learning, this comes from the environment.
    A word's category emerges from WHICH modalities are active when it's heard.
    
    Modality → Category mapping:
    - visual → NOUN
    - motor → VERB
    - properties → ADJECTIVE
    - spatial → PREPOSITION
    - social → PRONOUN
    - temporal → ADVERB
    - emotional → ADJECTIVE/ADVERB
    - none → FUNCTION WORD
    """
    visual: List[str] = field(default_factory=list)      # Objects present → NOUN
    motor: List[str] = field(default_factory=list)       # Actions happening → VERB
    properties: List[str] = field(default_factory=list)  # Properties observed → ADJECTIVE
    spatial: List[str] = field(default_factory=list)     # Spatial relations → PREPOSITION
    social: List[str] = field(default_factory=list)      # Social context → PRONOUN
    temporal: List[str] = field(default_factory=list)    # Time concepts → ADVERB
    emotional: List[str] = field(default_factory=list)   # Emotions → ADJ/ADV
    cognitive: List[str] = field(default_factory=list)   # Mental states → COGNITIVE VERB


@dataclass
class GroundedSentence:
    """A sentence with full grounding information for training"""
    words: List[str]
    contexts: List[GroundingContext]
    roles: List[str] = None  # 'agent', 'patient', 'action', None
    mood: str = 'declarative'
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = [None] * len(self.words)


@dataclass
class EmergentParams:
    """Parameters for emergent NEMO brain and learner"""
    n: int = 10000         # Neurons per area
    k: int = None          # Winners (sqrt(n) if None)
    p: float = 0.05        # Connection probability
    beta: float = 0.1      # Hebbian plasticity
    w_max: float = 10.0    # Weight saturation
    tau: int = 3           # Firing steps per word
    
    # Stability threshold for category assignment
    stability_threshold: float = 0.3
    
    # Maximum learned connections per area
    max_learned_factor: int = 500  # max_learned = k * k * factor
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))
    
    @property
    def max_learned(self) -> int:
        return self.k * self.k * self.max_learned_factor

