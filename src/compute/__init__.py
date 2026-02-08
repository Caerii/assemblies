"""
Mathematical primitives for neural assembly computations.

This module contains the mathematical foundations for neural assembly
simulations, including statistical distributions, neural computation
algorithms, and mathematical approximations.
"""

from .statistics import StatisticalEngine
from .neural_computation import NeuralComputationEngine
from .winner_selection import WinnerSelector
from .plasticity import PlasticityEngine
from .sparse_simulation import SparseSimulationEngine
from .explicit_projection import ExplicitProjectionEngine
from .image_activation import ImageActivationEngine

__all__ = ['StatisticalEngine', 'NeuralComputationEngine', 'WinnerSelector', 'PlasticityEngine',
           'SparseSimulationEngine', 'ExplicitProjectionEngine', 'ImageActivationEngine']
