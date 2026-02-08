"""
Neural Assembly Simulation Framework

A modular, mathematically rigorous framework for simulating neural assemblies
based on the Assembly Calculus and NEMO model.

Based on:
- Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (2020)
- Mitropolsky et al. "Architecture of a Biologically Plausible Language Organ" (2023)
"""

# Core modules
from .core import (
    Brain, Area, Stimulus, Connectome,
    ComputeEngine, ProjectionResult, create_engine, list_engines,
)

# Mathematical primitives
from .compute import (
    StatisticalEngine, NeuralComputationEngine, WinnerSelector, PlasticityEngine,
)

# Constants
from .constants import DEFAULT_P, DEFAULT_BETA

# Utilities
from .utils import normalize_features, select_top_k_indices, heapq_select_top_k, binomial_ppf

# Assembly Calculus operations
from .assembly_calculus import (
    Assembly, overlap, chance_overlap,
    project, reciprocal_project, associate, merge, pattern_complete, separate,
    FiberCircuit,
)

# GPU availability (based on cupy detection)
try:
    import cupy  # noqa: F401
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__version__ = "0.3.0"
__author__ = "Neural Assembly Research Team"

__all__ = [
    # Core classes
    'Brain', 'Area', 'Stimulus', 'Connectome',

    # Compute engine API
    'ComputeEngine', 'ProjectionResult', 'create_engine', 'list_engines',

    # Mathematical engines
    'StatisticalEngine', 'NeuralComputationEngine', 'WinnerSelector', 'PlasticityEngine',

    # Constants
    'DEFAULT_P', 'DEFAULT_BETA',

    # Utilities
    'normalize_features', 'select_top_k_indices', 'heapq_select_top_k', 'binomial_ppf',

    # Assembly Calculus operations
    'Assembly', 'overlap', 'chance_overlap',
    'project', 'reciprocal_project', 'associate', 'merge', 'pattern_complete', 'separate',
    'FiberCircuit',

    # GPU availability flag
    'GPU_AVAILABLE',
]
