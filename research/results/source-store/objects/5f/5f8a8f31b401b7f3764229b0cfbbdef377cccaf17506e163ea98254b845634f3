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
    from neural_assemblies.nemo.language.emergent import EmergentLanguageLearner, create_training_data
    
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
    from neural_assemblies.nemo.language.emergent.parser import SentenceParser, QuestionAnswerer
    parser = SentenceParser(learner)
    result = parser.parse(['the', 'dog', 'runs'])
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .brain import EmergentNemoBrain
    from .learner import EmergentLanguageLearner
    from .generator import SentenceGenerator
    from .parser import SentenceParser, ParseResult, QuestionAnswerer

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

_LAZY_EXPORTS = {
    "EmergentNemoBrain": (".brain", "EmergentNemoBrain"),
    "EmergentLanguageLearner": (".learner", "EmergentLanguageLearner"),
    "SentenceGenerator": (".generator", "SentenceGenerator"),
    "create_training_data": (".training_data", "create_training_data"),
    "create_simple_training_data": (".training_data", "create_simple_training_data"),
    "SentenceParser": (".parser", "SentenceParser"),
    "ParseResult": (".parser", "ParseResult"),
    "QuestionAnswerer": (".parser", "QuestionAnswerer"),
}


def __getattr__(name: str):
    """Load GPU-backed language components only when explicitly requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value

__all__ = ['Area', 'NUM_AREAS', 'MUTUAL_INHIBITION_GROUPS', 'GROUNDING_TO_CORE',
           'INPUT_AREAS', 'LEXICAL_AREAS', 'CORE_AREAS', 'THEMATIC_AREAS',
           'PHRASE_AREAS', 'SYNTACTIC_AREAS', 'CONTROL_AREAS', 'EmergentParams',
           'GroundingContext', 'GroundingModality', 'GroundedSentence',
           'EmergentNemoBrain', 'EmergentLanguageLearner', 'SentenceGenerator',
           'SentenceParser', 'ParseResult', 'QuestionAnswerer',
           'create_training_data', 'create_simple_training_data']  # pyright: ignore[reportUnsupportedDunderAll]

