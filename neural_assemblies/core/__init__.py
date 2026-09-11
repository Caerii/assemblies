"""
Core neural assembly primitives.

This module contains the fundamental data structures, orchestrator,
and compute engine interface for neural assembly simulations.
"""

from .brain import Brain
from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome
from .engine import (
    ComputeEngine, EngineUnavailableError, ProjectionResult, create_engine,
    list_engines,
)
from .projection_fidelity import ProjectionFidelity
from .semantics import (
    ArithmeticMode,
    CandidateDomain,
    ConnectomeMode,
    ModelSemantics,
    NormalizationMode,
    PlasticityRule,
    SampledRecurrencePolicy,
    StimulusDriveLaw,
    TieBreakRule,
)
from .backend import set_backend, get_xp, get_backend_name, to_cpu, to_xp

__all__ = [
    'Brain', 'Area', 'Stimulus', 'Connectome',
    'ComputeEngine', 'EngineUnavailableError', 'ProjectionResult',
    'create_engine', 'list_engines',
    'ProjectionFidelity', 'SampledRecurrencePolicy', 'ModelSemantics',
    'ConnectomeMode', 'CandidateDomain', 'StimulusDriveLaw',
    'TieBreakRule', 'ArithmeticMode', 'NormalizationMode', 'PlasticityRule',
    'set_backend', 'get_xp', 'get_backend_name', 'to_cpu', 'to_xp',
]
