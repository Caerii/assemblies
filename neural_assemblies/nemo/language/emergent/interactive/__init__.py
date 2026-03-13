"""
Interactive NEMO - Learning from Dialogue
==========================================

Extends the EmergentLanguageLearner for interactive,
continuous learning from conversation.

Key principles:
1. Every interaction is a learning opportunity
2. Categories and patterns emerge from experience
3. Self-knowledge emerges from self-referential dialogue
4. No pre-programmed responses - everything learned
"""

from .dialogue import DialogueState, Turn
from .grounding import GroundingInference
from .response import ResponseGenerator
from .interactive_learner import InteractiveLearner

__all__ = [
    'DialogueState',
    'Turn', 
    'GroundingInference',
    'ResponseGenerator',
    'InteractiveLearner',
]

