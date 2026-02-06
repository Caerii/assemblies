"""
NEMO: Neural Assembly Model
===========================

Version: 2.0.0
Author: Assembly Calculus Project
Date: 2025-11-30

A biologically-inspired neural network model based on Assembly Calculus.

Architecture:
- core/: Minimal GPU components (kernel, area, brain)
- language/: Language learning and generation (no hardcoded grammar)
- archive/: Old versions kept for reference

Key Principle:
  Grammar and word order are LEARNED from data, not hardcoded.
  This makes the model scientifically valuable for testing
  whether assemblies can learn linguistic structure.

Usage:
    from src.nemo.core import Brain, BrainParams
    from src.nemo.language import LanguageLearner, SentenceGenerator
    
    # Create learner
    learner = LanguageLearner()
    
    # Train on sentences (word order learned, not specified)
    learner.hear_sentence(['dog', 'chases', 'cat'])
    learner.hear_sentence(['cat', 'sees', 'dog'])
    
    # Generate from learned patterns
    generator = SentenceGenerator(learner)
    sentence = generator.generate_sentence()

Changelog:
- 2.0.0: Modular architecture, no hardcoded grammar
- 1.x.x: Archived (hardcoded SVO/SOV)
"""

__version__ = "2.0.0"
__author__ = "Assembly Calculus Project"

# Core components
from .core import Brain, BrainParams, Area, AreaParams

# Language components  
from .language import LanguageLearner, SentenceGenerator

__all__ = [
    'Brain', 'BrainParams', 
    'Area', 'AreaParams',
    'LanguageLearner', 'SentenceGenerator',
]
