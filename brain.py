"""Backward-compatible shim. Import from src.core instead."""

from src.core.brain import Brain
from src.core.area import Area

__all__ = ['Brain', 'Area']
