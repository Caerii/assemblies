"""
Core neural assembly primitives.

This module contains the fundamental data structures, orchestrator,
and compute engine interface for neural assembly simulations.
"""

from .brain import Brain
from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome
from .activity import PopulationCounts, PreKwtaObservation
from .engine import (
    ComputeEngine, EngineUnavailableError, ProjectionResult, create_engine,
    list_engines,
)
from .projection_fidelity import ProjectionFidelity
from .semantics import (
    ArithmeticMode,
    AlignerSemantics,
    AlignmentInferenceSchedule,
    AlignmentStore,
    AlignmentTrainingSchedule,
    CandidateDomain,
    ConnectomeMode,
    ExecutionKind,
    ExecutionSemantics,
    InferenceSchedule,
    ModelSemantics,
    NormalizationMode,
    OrganKind,
    OrganSemantics,
    PlasticityRule,
    SampledRecurrencePolicy,
    StimulusDriveLaw,
    TieBreakRule,
    TrainingSchedule,
    StateCode,
    describe_assembly_memory,
    describe_brain_model,
    describe_hashed_arc_fsm,
    describe_hashed_aligner,
    describe_hashed_transducer,
)
from .backend import set_backend, get_xp, get_backend_name, to_cpu, to_xp

__all__ = [
    'Brain', 'Area', 'Stimulus', 'Connectome', 'PopulationCounts',
    'PreKwtaObservation',
    'ComputeEngine', 'EngineUnavailableError', 'ProjectionResult',
    'create_engine', 'list_engines',
    'ProjectionFidelity', 'SampledRecurrencePolicy', 'ModelSemantics',
    'AlignerSemantics', 'AlignmentStore', 'AlignmentTrainingSchedule',
    'AlignmentInferenceSchedule',
    'ExecutionSemantics', 'ExecutionKind',
    'OrganSemantics', 'OrganKind', 'StateCode', 'TrainingSchedule',
    'InferenceSchedule',
    'ConnectomeMode', 'CandidateDomain', 'StimulusDriveLaw',
    'TieBreakRule', 'ArithmeticMode', 'NormalizationMode', 'PlasticityRule',
    'describe_brain_model', 'describe_assembly_memory',
    'describe_hashed_arc_fsm', 'describe_hashed_transducer',
    'describe_hashed_aligner',
    'set_backend', 'get_xp', 'get_backend_name', 'to_cpu', 'to_xp',
]
