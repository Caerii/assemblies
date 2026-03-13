"""Backward-compatible shim. Import from neural_assemblies.core instead."""

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.area import Area

__all__ = ['Brain', 'Area']
