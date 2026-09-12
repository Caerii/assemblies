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

_LAZY_EXPORTS = {
    "ResponseGenerator": (".response", "ResponseGenerator"),
    "InteractiveLearner": (".interactive_learner", "InteractiveLearner"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value

__all__ = [
    'DialogueState',
    'Turn', 
    'GroundingInference',
    'ResponseGenerator',
    'InteractiveLearner',
]

