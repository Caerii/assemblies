"""
NEMO Core Module
================

The minimal, reusable components for Assembly Calculus simulation.

Components:
- kernel: CUDA projection kernel (the only GPU code needed)
- area: Brain area with projection and learning
- brain: Base brain class with multiple areas

All language-specific logic should be in separate modules.
"""

from .kernel import projection_kernel
from .area import Area, AreaParams
from .brain import Brain, BrainParams

__all__ = [
    'projection_kernel',
    'Area', 'AreaParams', 
    'Brain', 'BrainParams'
]

