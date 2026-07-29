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
    StatisticalEngine, NeuralComputationEngine,
    TopKPolicy, ThresholdPolicy, RelativeThresholdPolicy, EPercentPolicy, WinnerPolicy,
    WinnerSelector, PlasticityEngine,
)

# Constants
from .constants import DEFAULT_P, DEFAULT_BETA

# Utilities
from .utils import normalize_features, select_top_k_indices, heapq_select_top_k, binomial_ppf

# Assembly Calculus operations
from .assembly_calculus import (
    Assembly, AssemblyTrace, PatternCompletionDiagnostic, ResponseDiagnostic,
    ResponseTrace, TraceStep, overlap, chance_overlap,
    project, reciprocal_project, associate, merge, pattern_complete, separate,
    project_trace, reciprocal_project_trace, associate_trace, merge_trace,
    pattern_complete_trace, ordered_recall_trace,
    snapshot_area, source_response_traces,
    FiberCircuit,
)

# GPU availability. DETECTED WITHOUT IMPORTING CuPy, and that is load-bearing
# rather than a micro-optimisation.
#
# This line used to be `import cupy`, purely to set the boolean below. Importing
# CuPy loads its copy of the CUDA runtime and cuBLAS, and on Windows whichever
# of CuPy / PyTorch loads first wins DLL resolution for the rest of the process.
# Because this runs at PACKAGE import time, CuPy always won -- so any later
# `import torch` bound against CuPy's libraries and crashed on larger cuBLAS
# calls with a bare `Windows fatal exception: access violation`, no traceback
# and no Python-level error.
#
# Measured, cupy 14.1.1 (CUDA 13.02) against torch 2.12.1+cu130:
#
#     import torch, cupy   -> BatchedSeqTrainer trains fine
#     import cupy, torch   -> access violation in `act @ self.W`
#     import cupy alone, running NO CuPy operations -> same crash
#
# It took down two test files (test_batched_trainer, test_batched_next_token)
# whose own code is correct -- they pass standalone and crash under pytest,
# because conftest imports this package first. `find_spec` answers the same
# question with no CUDA initialisation at all.
from importlib.util import find_spec as _find_spec

GPU_AVAILABLE = _find_spec("cupy") is not None

# Version kept in sync with pyproject.toml for the installed package
__version__ = "0.0.1a1"  # kept in sync with pyproject.toml
__author__ = "Superintelligent Group"

__all__ = [
    # Core classes
    'Brain', 'Area', 'Stimulus', 'Connectome',

    # Compute engine API
    'ComputeEngine', 'ProjectionResult', 'create_engine', 'list_engines',

    # Mathematical engines
    'StatisticalEngine', 'NeuralComputationEngine',
    'TopKPolicy', 'ThresholdPolicy', 'RelativeThresholdPolicy', 'EPercentPolicy', 'WinnerPolicy',
    'WinnerSelector', 'PlasticityEngine',

    # Constants
    'DEFAULT_P', 'DEFAULT_BETA',

    # Utilities
    'normalize_features', 'select_top_k_indices', 'heapq_select_top_k', 'binomial_ppf',

    # Assembly Calculus operations
    'Assembly', 'AssemblyTrace', 'PatternCompletionDiagnostic',
    'ResponseDiagnostic', 'ResponseTrace', 'TraceStep',
    'overlap', 'chance_overlap',
    'project', 'reciprocal_project', 'associate', 'merge', 'pattern_complete', 'separate',
    'project_trace', 'reciprocal_project_trace', 'associate_trace',
    'merge_trace', 'pattern_complete_trace', 'ordered_recall_trace',
    'snapshot_area', 'source_response_traces',
    'FiberCircuit',

    # GPU availability flag
    'GPU_AVAILABLE',
]
