"""
Emergent NEMO Language Learning
===============================

Version: 2.1.0
Date: 2025-11-30

A neurobiologically plausible language learner where ALL categories
emerge from grounding patterns. No pre-labeled categories.

Modules:
- areas: Brain area definitions (37 areas)
- params: Parameters and data classes
- brain: EmergentNemoBrain class
- learner: EmergentLanguageLearner class
- generator: Sentence generation
- training_data: Grounded training sentences
- parser/: Parsing and comprehension submodule

Example:
    from nemo.language.emergent import EmergentLanguageLearner, create_training_data
    
    learner = EmergentLanguageLearner()
    data = create_training_data()
    
    for sentence in data:
        learner.present_grounded_sentence(
            sentence.words, sentence.contexts,
            roles=sentence.roles, mood=sentence.mood
        )
    
    vocab = learner.get_vocabulary_by_category()
    print(vocab)
    
    # Parsing
    from nemo.language.emergent.parser import SentenceParser, QuestionAnswerer
    parser = SentenceParser(learner)
    result = parser.parse(['the', 'dog', 'runs'])
"""

__version__ = "2.1.0"

from .areas import (
    Area, NUM_AREAS,
    MUTUAL_INHIBITION_GROUPS, GROUNDING_TO_CORE,
    INPUT_AREAS, LEXICAL_AREAS, CORE_AREAS,
    THEMATIC_AREAS, PHRASE_AREAS, SYNTACTIC_AREAS, CONTROL_AREAS
)

from .params import (
    EmergentParams, GroundingContext, GroundingModality, GroundedSentence
)

from .brain import EmergentNemoBrain

from .learner import EmergentLanguageLearner

from .generator import SentenceGenerator

from .training_data import create_training_data, create_simple_training_data

# Parser submodule
from .parser import SentenceParser, ParseResult, QuestionAnswerer

__all__ = [
    # Areas
    'Area', 'NUM_AREAS',
    'MUTUAL_INHIBITION_GROUPS', 'GROUNDING_TO_CORE',
    'INPUT_AREAS', 'LEXICAL_AREAS', 'CORE_AREAS',
    'THEMATIC_AREAS', 'PHRASE_AREAS', 'SYNTACTIC_AREAS', 'CONTROL_AREAS',
    
    # Params
    'EmergentParams', 'GroundingContext', 'GroundingModality', 'GroundedSentence',
    
    # Classes
    'EmergentNemoBrain', 'EmergentLanguageLearner', 'SentenceGenerator',
    
    # Parser
    'SentenceParser', 'ParseResult', 'QuestionAnswerer',
    
    # Functions
    'create_training_data', 'create_simple_training_data',
]

