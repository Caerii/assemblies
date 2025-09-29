"""
Connectome operations and management.

This module handles synaptic connections, weight updates, connectivity
patterns, and dynamic connectome expansion.
"""

from .connectome_manager import ConnectomeManager
from .weight_updates import WeightUpdateEngine
from .connectivity_patterns import ConnectivityPatterns
from .sparse_connectomes import SparseConnectomeManager
from .connectome_expansion import ConnectomeExpansionEngine

__all__ = ['ConnectomeManager', 'WeightUpdateEngine', 'ConnectivityPatterns', 
           'SparseConnectomeManager', 'ConnectomeExpansionEngine']
