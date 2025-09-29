"""
Core neural assembly primitives.

This module contains the fundamental data structures and orchestrator
for neural assembly simulations.
"""

from .brain import Brain
from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome

__all__ = ['Brain', 'Area', 'Stimulus', 'Connectome']
