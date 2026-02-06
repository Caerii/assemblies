"""
NEMO Language Module
====================

Language-specific components built on the core Brain.

Key principle: Grammar and word order are LEARNED, not hardcoded.

Components:
- nemo_learner: Neurobiologically plausible NEMO learner (from papers)
- learner: Simple statistical learner
- generator: Generates sentences from learned patterns
- curriculum: Structured learning from simple to complex

NEMO Architecture (from Mitropolsky & Papadimitriou 2025):
- Phon → Lex1/Lex2 (differential for nouns/verbs)
- Visual → Lex1 (noun grounding)
- Motor → Lex2 (verb grounding)
- Role areas with mutual inhibition
- Sequence area for word order

Curriculum Stages (child language acquisition):
1. Single words (naming) - 12-18 months
2. Two-word combinations - 18-24 months
3. Simple sentences (SVO) - 24-30 months
4. Full sentences - 30-36 months
"""

from .learner import LanguageLearner
from .generator import SentenceGenerator
from .curriculum import Curriculum, CurriculumLearner, StructureType, StructureDetector
from .nemo_learner import (
    NemoLanguageLearner, NemoBrain, NemoParams,
    GroundedContext, GroundingType, SpeechAct, Area
)
from .integrated_trainer import IntegratedNemoTrainer, TrainingStats

__all__ = [
    # NEMO (neurobiologically plausible)
    'NemoLanguageLearner',
    'NemoBrain', 
    'NemoParams',
    'GroundedContext',
    'GroundingType',
    'SpeechAct',
    'Area',
    # Integrated trainer (with lexicon + curriculum)
    'IntegratedNemoTrainer',
    'TrainingStats',
    # Simple statistical
    'LanguageLearner', 
    'SentenceGenerator',
    # Curriculum
    'Curriculum',
    'CurriculumLearner',
    'StructureType',
    'StructureDetector',
]

