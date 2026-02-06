"""
Grounding Mapping
=================

Maps semantic domains and features to grounding contexts.

This is the KEY mapping that determines emergent categories:
- VISUAL grounding → NOUN
- MOTOR grounding → VERB
- PROPERTY grounding → ADJECTIVE
- SPATIAL grounding → PREPOSITION
- SOCIAL grounding → PRONOUN
- TEMPORAL grounding → ADVERB
- COGNITIVE grounding → COGNITIVE VERB
- No grounding → FUNCTION WORD
"""

from typing import Dict, List, Any

from ...params import GroundingContext


# Domain to modality mapping
DOMAIN_MODALITY_MAP = {
    # Visual/Object domains → NOUN
    'ANIMAL': 'visual',
    'PERSON': 'visual',
    'OBJECT': 'visual',
    'FOOD': 'visual',
    'PLANT': 'visual',
    'BUILDING': 'visual',
    'BODY_PART': 'visual',
    'FURNITURE': 'visual',
    
    # Action domains → MOTOR (VERB)
    'MOTION': 'motor',
    'CONSUMPTION': 'motor',
    'CREATION': 'motor',
    'DESTRUCTION': 'motor',
    'PERCEPTION': 'motor',
    'COMMUNICATION': 'motor',
    'POSSESSION': 'motor',
    
    # Cognitive domains → COGNITIVE
    'COGNITION': 'cognitive',
    
    # Emotion → EMOTIONAL
    'EMOTION': 'emotional',
    
    # Spatial → SPATIAL (PREPOSITION)
    'SPACE': 'spatial',
    'LOCATION': 'spatial',
    
    # Temporal → TEMPORAL (ADVERB)
    'TIME': 'temporal',
    'TEMPORAL': 'temporal',
    
    # Quality → PROPERTY (ADJECTIVE)
    'QUALITY': 'properties',
    'PROPERTY': 'properties',
    
    # Social → SOCIAL (PRONOUN)
    'SOCIAL': 'social',
}


def create_grounding_from_domains(domains: List[str], 
                                   features: Dict[str, Any] = None) -> GroundingContext:
    """
    Create grounding context from semantic domains and features.
    
    Args:
        domains: List of semantic domains (e.g., ['ANIMAL', 'PET'])
        features: Dictionary of semantic features (e.g., {'animate': True})
    
    Returns:
        GroundingContext with appropriate modalities filled
    """
    features = features or {}
    ctx = GroundingContext()
    
    # Map domains to grounding modalities
    for domain in domains:
        domain_upper = domain.upper()
        modality = DOMAIN_MODALITY_MAP.get(domain_upper)
        
        if modality:
            modality_list = getattr(ctx, modality)
            modality_list.append(domain_upper)
    
    # Add feature-based grounding
    if features.get('animate'):
        ctx.visual.append('ANIMATE')
    if features.get('human'):
        ctx.social.append('HUMAN')
    if features.get('abstract'):
        ctx.cognitive.append('ABSTRACT')
    
    return ctx


def create_pronoun_grounding(lemma: str) -> GroundingContext:
    """
    Create grounding for pronouns.
    
    Pronouns have special social grounding:
    - "I" → SELF, SPEAKER
    - "you" → ADDRESSEE, LISTENER
    - "we" → SELF, GROUP
    - "he/she/they" → PERSON, GENDER
    """
    lemma_lower = lemma.lower()
    
    if lemma_lower == 'i':
        return GroundingContext(social=['SELF', 'SPEAKER'])
    elif lemma_lower == 'you':
        return GroundingContext(social=['ADDRESSEE', 'LISTENER'])
    elif lemma_lower == 'we':
        return GroundingContext(social=['SELF', 'GROUP'])
    elif lemma_lower == 'he':
        return GroundingContext(social=['PERSON', 'MALE'])
    elif lemma_lower == 'she':
        return GroundingContext(social=['PERSON', 'FEMALE'])
    elif lemma_lower == 'it':
        return GroundingContext(social=['THING'])
    elif lemma_lower == 'they':
        return GroundingContext(social=['GROUP'])
    else:
        return GroundingContext(social=[lemma_lower.upper()])


def create_question_word_grounding(lemma: str) -> GroundingContext:
    """
    Create grounding for question words.
    
    Question words query for specific types of information:
    - "what" → queries for OBJECT/VISUAL
    - "who" → queries for PERSON/SOCIAL
    - "where" → queries for LOCATION/SPATIAL
    - "when" → queries for TIME/TEMPORAL
    - "why" → queries for REASON/COGNITIVE
    - "how" → queries for MANNER/PROPERTY
    """
    lemma_lower = lemma.lower()
    
    if lemma_lower == 'what':
        return GroundingContext(visual=['QUERY', 'OBJECT'])
    elif lemma_lower == 'who':
        return GroundingContext(social=['QUERY', 'PERSON'])
    elif lemma_lower == 'where':
        return GroundingContext(spatial=['QUERY', 'LOCATION'])
    elif lemma_lower == 'when':
        return GroundingContext(temporal=['QUERY', 'TIME'])
    elif lemma_lower == 'why':
        return GroundingContext(cognitive=['QUERY', 'REASON'])
    elif lemma_lower == 'how':
        return GroundingContext(properties=['QUERY', 'MANNER'])
    else:
        return GroundingContext()


__all__ = [
    'create_grounding_from_domains',
    'create_pronoun_grounding', 
    'create_question_word_grounding',
    'DOMAIN_MODALITY_MAP',
]


