"""
Learning and adaptation mechanisms.

This module implements various learning algorithms and adaptation
mechanisms for neural assemblies, including Hebbian learning and
memory consolidation.
"""

from .hebbian_learning import HebbianLearning
from .adaptation import NeuralAdaptation
from .learning_rules import LearningRules
from .memory_consolidation import MemoryConsolidation

__all__ = ['HebbianLearning', 'NeuralAdaptation', 'LearningRules', 'MemoryConsolidation']
