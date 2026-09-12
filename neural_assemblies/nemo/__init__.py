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
    from neural_assemblies.nemo.core import Brain, BrainParams
    from neural_assemblies.nemo.language import LanguageLearner, SentenceGenerator
    
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

# NEMO has an optional CuPy implementation. Keep the package namespace
# importable on CPU-only installations; resolve GPU-backed symbols only when
# a caller asks for them. This preserves the public API while making the
# dependency boundary explicit and avoiding an eager CuPy import that also
# interferes with the NumPy/Torch engines.
from importlib import import_module

_LAZY_SYMBOLS = {
    "Brain": (".core", "Brain"),
    "BrainParams": (".core", "BrainParams"),
    "Area": (".core", "Area"),
    "AreaParams": (".core", "AreaParams"),
    "LanguageLearner": (".language", "LanguageLearner"),
    "SentenceGenerator": (".language", "SentenceGenerator"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _LAZY_SYMBOLS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = list(_LAZY_SYMBOLS)

