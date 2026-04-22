"""Backward-compatible repo-root shim.

This file exists for historical checkout workflows that still do ``import brain``
from the repository root. The primary supported library API is the installable
package:

    from neural_assemblies.core import Brain, Area
"""

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.area import Area

__all__ = ['Brain', 'Area']
