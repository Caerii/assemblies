"""
EmergentParser — 44-area emergent NEMO parser on numpy_sparse.

Thin assembler: composes 7 mixin modules into a single EmergentParser class
via multiple inheritance. Each mixin provides a cohesive feature group:

    CoreParserMixin        — __init__, setup, classification, batch parse
    DistributionalMixin    — distributional learning, raw text, word order
    MorphosyntaxMixin      — tense, mood, polarity, conjunctions
    IncrementalMixin       — word-by-word incremental + recursive parsing
    GenerationMixin        — language production from semantic roles
    UnsupervisedMixin      — role learning from raw exposure
    PredictionMixin        — structural next-token prediction

Cross-mixin calls resolve at runtime via Python's MRO since `self` is
the fully-composed class.  No mixin imports another mixin class.

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."
"""

from ._parser_core import CoreParserMixin, DistributionalStats
from ._parser_distributional import DistributionalMixin
from ._parser_morphosyntax import MorphosyntaxMixin
from ._parser_incremental import IncrementalMixin
from ._parser_generation import GenerationMixin
from ._parser_unsupervised import UnsupervisedMixin
from ._parser_prediction import PredictionMixin
from .curriculum import CurriculumTrainer, StageResult, _STAGE_CONFIG
from .evaluation import EvaluationSuite


class EmergentParser(
    CoreParserMixin,
    DistributionalMixin,
    MorphosyntaxMixin,
    IncrementalMixin,
    GenerationMixin,
    UnsupervisedMixin,
    PredictionMixin,
):
    """44-area emergent NEMO parser composed from feature mixins."""
    pass


__all__ = [
    "EmergentParser",
    "CurriculumTrainer",
    "StageResult",
    "_STAGE_CONFIG",
    "EvaluationSuite",
    "DistributionalStats",
]
