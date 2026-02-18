"""
Emergent NEMO Brain Areas
=========================

Version: 2.0.0
Date: 2025-12-01

Defines the 37 neurobiologically plausible brain areas for emergent
language learning. ALL categories emerge from grounding patterns.

Based on:
- Mitropolsky & Papadimitriou (2025)
- parser.py, learner.py from Assembly Calculus
- Neuroscience literature on language processing
"""

from enum import Enum
from typing import List

__all__ = [
    'Area', 'NUM_AREAS', 
    'MUTUAL_INHIBITION_GROUPS', 'GROUNDING_TO_CORE',
    'INPUT_AREAS', 'LEXICAL_AREAS', 'CORE_AREAS',
    'THEMATIC_AREAS', 'PHRASE_AREAS', 'SYNTACTIC_AREAS',
    'CONTROL_AREAS', 'VP_COMPONENT_AREAS'
]


class Area(Enum):
    """
    Brain areas - ALL categories EMERGE from grounding patterns.
    
    Based on neuroscience and NEMO papers:
    - Input areas: sensory modalities
    - Lexical areas: content vs function words
    - Core areas: grammatical categories (emerge from grounding)
    - Thematic roles: semantic roles (mutual inhibition)
    - Phrase structure: compositional units
    - Syntactic roles: grammatical functions
    - Sequence/Control: word order, mood, tense
    """
    
    # =========== INPUT MODALITIES (8 areas) ===========
    PHON = 0           # Phonological input (auditory cortex)
    VISUAL = 1         # Visual grounding - objects (inferotemporal)
    MOTOR = 2          # Motor grounding - actions (motor cortex)
    PROPERTY = 3       # Properties - size, color (temporal-parietal)
    SPATIAL = 4        # Spatial relations (parietal cortex)
    TEMPORAL = 5       # Time concepts (prefrontal)
    SOCIAL = 6         # Social/people (temporal pole)
    EMOTION = 7        # Emotional concepts (amygdala, insula)
    
    # =========== LEXICAL AREAS (2 areas) ===========
    LEX_CONTENT = 8    # Content words (middle temporal)
    LEX_FUNCTION = 9   # Function words (inferior frontal)
    
    # =========== CORE/CATEGORY AREAS (8 areas) ===========
    # These EMERGE from consistent grounding patterns
    NOUN_CORE = 10     # Noun category (← VISUAL grounding)
    VERB_CORE = 11     # Verb category (← MOTOR grounding)
    ADJ_CORE = 12      # Adjective category (← PROPERTY grounding)
    ADV_CORE = 13      # Adverb category (← manner/degree)
    PREP_CORE = 14     # Preposition category (← SPATIAL grounding)
    DET_CORE = 15      # Determiner category (← high freq, no grounding)
    PRON_CORE = 16     # Pronoun category (← SOCIAL grounding)
    CONJ_CORE = 17     # Conjunction category (← linking, no grounding)
    
    # =========== THEMATIC ROLE AREAS (6 areas) ===========
    # Under MUTUAL INHIBITION - only one active at a time
    ROLE_AGENT = 18    # Agent/doer of action
    ROLE_PATIENT = 19  # Patient/undergoer
    ROLE_THEME = 20    # Theme/moved entity
    ROLE_GOAL = 21     # Goal/destination
    ROLE_SOURCE = 22   # Source/origin
    ROLE_LOCATION = 23 # Location
    
    # =========== PHRASE STRUCTURE (5 areas) ===========
    NP = 24            # Noun phrase (DET + ADJ + N)
    VP = 25            # Verb phrase (V + NP/PP)
    PP = 26            # Prepositional phrase (PREP + NP)
    ADJP = 27          # Adjective phrase
    SENT = 28          # Full sentence (NP + VP)
    
    # =========== SYNTACTIC ROLES (3 areas) ===========
    # Under MUTUAL INHIBITION
    SUBJ = 29          # Subject
    OBJ = 30           # Direct object
    IOBJ = 31          # Indirect object
    
    # =========== SEQUENCE/CONTROL (5 areas) ===========
    SEQ = 32           # Sequence memory (word order)
    MOOD = 33          # Sentence mood (declarative, interrogative)
    TENSE = 34         # Tense marking
    POLARITY = 35      # Affirmative/Negative
    NUMBER = 36        # Morphological number (SG/PL)

    # =========== ERROR DETECTION (1 area) ===========
    ERROR = 37         # Parse error (wobbly = error)
    
    # =========== VP COMPONENT AREAS (3 areas) ===========
    # These are MERGE areas that preserve component information
    # Unlike VP which blends all components, these keep them separate
    # This enables emergent retrieval: VP_SUBJ can be decoded to find subjects
    VP_SUBJ = 38       # Subject component of VP (projects from NOUN_CORE/PRON_CORE)
    VP_VERB = 39       # Verb component of VP (projects from VERB_CORE)
    VP_OBJ = 40        # Object component of VP (projects from NOUN_CORE)


NUM_AREAS = 41

# Mutual inhibition groups (only one area in each group can be active)
MUTUAL_INHIBITION_GROUPS: List[List[Area]] = [
    # Thematic roles compete
    [Area.ROLE_AGENT, Area.ROLE_PATIENT, Area.ROLE_THEME, 
     Area.ROLE_GOAL, Area.ROLE_SOURCE, Area.ROLE_LOCATION],
    # Syntactic roles compete
    [Area.SUBJ, Area.OBJ, Area.IOBJ],
]

# Grounding → Core area mappings (for emergent categorization)
GROUNDING_TO_CORE = {
    'VISUAL': Area.NOUN_CORE,
    'MOTOR': Area.VERB_CORE,
    'PROPERTY': Area.ADJ_CORE,
    'SPATIAL': Area.PREP_CORE,
    'SOCIAL': Area.PRON_CORE,
    'TEMPORAL': Area.ADV_CORE,
    'NONE': Area.DET_CORE,
}

# Area groupings for convenience
INPUT_AREAS = [Area.PHON, Area.VISUAL, Area.MOTOR, Area.PROPERTY, 
               Area.SPATIAL, Area.TEMPORAL, Area.SOCIAL, Area.EMOTION]

LEXICAL_AREAS = [Area.LEX_CONTENT, Area.LEX_FUNCTION]

CORE_AREAS = [Area.NOUN_CORE, Area.VERB_CORE, Area.ADJ_CORE, Area.ADV_CORE,
              Area.PREP_CORE, Area.DET_CORE, Area.PRON_CORE, Area.CONJ_CORE]

THEMATIC_AREAS = [Area.ROLE_AGENT, Area.ROLE_PATIENT, Area.ROLE_THEME,
                  Area.ROLE_GOAL, Area.ROLE_SOURCE, Area.ROLE_LOCATION]

PHRASE_AREAS = [Area.NP, Area.VP, Area.PP, Area.ADJP, Area.SENT]

SYNTACTIC_AREAS = [Area.SUBJ, Area.OBJ, Area.IOBJ]

CONTROL_AREAS = [Area.SEQ, Area.MOOD, Area.TENSE, Area.POLARITY, Area.NUMBER, Area.ERROR]

# VP Component areas - these preserve component information for emergent retrieval
VP_COMPONENT_AREAS = [Area.VP_SUBJ, Area.VP_VERB, Area.VP_OBJ]

