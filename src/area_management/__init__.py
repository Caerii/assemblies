"""
Area lifecycle and management.

This module handles the creation, management, and lifecycle of brain areas,
including explicit areas, sparse areas, and area state management.
"""

from .area_factory import AreaFactory
from .explicit_areas import ExplicitAreaManager
from .sparse_areas import SparseAreaManager
from .area_state import AreaStateManager

__all__ = ['AreaFactory', 'ExplicitAreaManager', 'SparseAreaManager', 'AreaStateManager']
