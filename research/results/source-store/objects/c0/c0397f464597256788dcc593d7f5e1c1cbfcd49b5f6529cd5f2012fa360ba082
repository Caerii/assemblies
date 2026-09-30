"""
NEMO Emergent Parser Module
===========================

Version: 1.0.0
Date: 2025-11-30

A neurobiologically plausible parser that uses learned assemblies
to parse sentences and answer questions.

Key concepts from Assembly Calculus parsing:
- PRE_RULES / POST_RULES: Control fiber inhibition per lexeme
- Fiber states: Track which connections are active
- Readout via fiber activation: Use activated fibers for parse tree
- Fix/Unfix assemblies: Freeze assemblies during certain operations

Module structure:
- core.py: Core parsing logic and ParseResult
- comprehension.py: Question answering
- rules.py: Fiber/Area rules (inspired by parser.py)
"""

from .core import SentenceParser, ParseResult
from .comprehension import QuestionAnswerer

__all__ = ['SentenceParser', 'ParseResult', 'QuestionAnswerer']

