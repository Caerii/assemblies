"""
Neural Assembly Simulation Framework

A modular, mathematically rigorous framework for simulating neural assemblies
based on the Assembly Calculus and NEMO model.

This framework implements:
- Assembly Calculus operations (projection, association, merge)
- NEMO model for biologically plausible neural computation
- Modular architecture for systematic composition and upgrades
- Comprehensive mathematical primitives for neural computation

Based on:
- Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (2020)
- Mitropolsky et al. "Architecture of a Biologically Plausible Language Organ" (2023)
"""

# Core modules
from .core import Brain, Area, Stimulus, Connectome

# Mathematical primitives
from .math_primitives import StatisticalEngine, NeuralComputationEngine, WinnerSelector, PlasticityEngine

# Constants
from .constants import DEFAULT_P, DEFAULT_BETA

# Utilities
from .utils import normalize_features, select_top_k_indices, heapq_select_top_k, binomial_ppf

# GPU acceleration (optional)
try:
    from .gpu import CupyBrain, TorchBrain, GPUUtils, MemoryManager, PerformanceProfiler
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Neural Assembly Research Team"

__all__ = [
    # Core classes
    'Brain', 'Area', 'Stimulus', 'Connectome',
    
    # Mathematical engines
    'StatisticalEngine', 'NeuralComputationEngine', 'WinnerSelector', 'PlasticityEngine',
    
    # Constants
    'DEFAULT_P', 'DEFAULT_BETA',
    
    # Utilities
    'normalize_features', 'select_top_k_indices', 'heapq_select_top_k', 'binomial_ppf',
    
    # GPU acceleration (if available)
    'GPU_AVAILABLE'
]

# Add GPU classes to __all__ if available
if GPU_AVAILABLE:
    __all__.extend(['CupyBrain', 'TorchBrain', 'GPUUtils', 'MemoryManager', 'PerformanceProfiler'])