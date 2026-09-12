"""
NEMO Language Module
====================

Language-specific components built on the core Brain.

Key principle: these learners aim to acquire category structure and word order
from exposure rather than fixed sentence templates.

Components:
- nemo_learner: NEMO learner inspired by the papers
- learner: Simple statistical learner
- generator: Generates sentences from learned patterns
- curriculum: Structured learning from simple to complex

NEMO-style architecture (from Mitropolsky & Papadimitriou 2025):
- Phon → Lex1/Lex2 (differential for nouns/verbs)
- Visual → Lex1 (noun grounding)
- Motor → Lex2 (verb grounding)
- Role areas with mutual inhibition
- Sequence area for word order

Curriculum stages are modeled after child language acquisition:
1. Single words (naming) - 12-18 months
2. Two-word combinations - 18-24 months
3. Simple sentences (SVO) - 24-30 months
4. Full sentences - 30-36 months
"""

from importlib import import_module

# Keep the package itself CPU-importable.  The statistical utilities and the
# curriculum definitions do not need the legacy CuPy learner; resolve every
# public symbol only when requested, and let the requested optional backend
# report its own dependency error.
_LAZY_SYMBOLS = {
    'LanguageLearner': ('.learner', 'LanguageLearner'),
    'SentenceGenerator': ('.generator', 'SentenceGenerator'),
    'Curriculum': ('.curriculum', 'Curriculum'),
    'CurriculumLearner': ('.curriculum', 'CurriculumLearner'),
    'StructureType': ('.curriculum', 'StructureType'),
    'StructureDetector': ('.curriculum', 'StructureDetector'),
    'NemoLanguageLearner': ('.nemo_learner', 'NemoLanguageLearner'),
    'NemoBrain': ('.nemo_learner', 'NemoBrain'),
    'NemoParams': ('.nemo_learner', 'NemoParams'),
    'GroundedContext': ('.nemo_learner', 'GroundedContext'),
    'GroundingType': ('.nemo_learner', 'GroundingType'),
    'SpeechAct': ('.nemo_learner', 'SpeechAct'),
    'Area': ('.nemo_learner', 'Area'),
    'IntegratedNemoTrainer': ('.integrated_trainer', 'IntegratedNemoTrainer'),
    'TrainingStats': ('.integrated_trainer', 'TrainingStats'),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _LAZY_SYMBOLS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value

__all__ = list(_LAZY_SYMBOLS)  # pyright: ignore[reportUnsupportedDunderAll]

