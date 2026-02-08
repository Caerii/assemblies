"""
Core neural assembly primitives.

This module contains the fundamental data structures, orchestrator,
and compute engine interface for neural assembly simulations.
"""

from .brain import Brain
from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome
from .engine import ComputeEngine, ProjectionResult, create_engine, list_engines
from .backend import set_backend, get_xp, get_backend_name, to_cpu, to_xp

__all__ = [
    'Brain', 'Area', 'Stimulus', 'Connectome',
    'ComputeEngine', 'ProjectionResult', 'create_engine', 'list_engines',
    'set_backend', 'get_xp', 'get_backend_name', 'to_cpu', 'to_xp',
]
