"""
Emergent NEMO parser — 44-area architecture on numpy_sparse.

Ports the full emergent NEMO architecture (Mitropolsky & Papadimitriou 2025)
from cupy/CUDA to the proven numpy_sparse engine. Categories emerge from
grounding patterns, not hardcoded labels.

Usage::

    from src.assembly_calculus.emergent import EmergentParser

    parser = EmergentParser(n=10000, k=100, seed=42)
    parser.train()
    result = parser.parse(["the", "big", "dog", "chases", "a", "cat"])
"""

from .parser import (
    EmergentParser, CurriculumTrainer, StageResult, EvaluationSuite,
)
from .grounding import GroundingContext
from .training_data import GroundedSentence, generate_training_sentences
from .vocabulary_builder import build_vocabulary
from .areas import (
    ALL_AREAS, CORE_AREAS, CORE_TO_CATEGORY, CATEGORY_TO_CORE,
    GROUNDING_TO_CORE, PHRASE_AREAS, THEMATIC_AREAS,
    CONTEXT, PRODUCTION, PREDICTION, DEP_CLAUSE,
)

__all__ = [
    "EmergentParser",
    "CurriculumTrainer",
    "StageResult",
    "EvaluationSuite",
    "GroundingContext",
    "GroundedSentence",
    "build_vocabulary",
    "generate_training_sentences",
    "ALL_AREAS",
    "CORE_AREAS",
    "CORE_TO_CATEGORY",
    "CATEGORY_TO_CORE",
    "GROUNDING_TO_CORE",
    "PHRASE_AREAS",
    "THEMATIC_AREAS",
    "CONTEXT",
    "PRODUCTION",
    "PREDICTION",
    "DEP_CLAUSE",
]
