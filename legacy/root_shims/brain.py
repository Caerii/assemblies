"""Historical ``import brain`` shim.

Daniel Mitropolsky's original library was a single ``brain.py`` at the
repository root, and the archived scripts and simulations import it by that
name. This re-export keeps those imports working with ``legacy/root_shims``
on ``PYTHONPATH``. The supported API is the package:

    from neural_assemblies.core import Brain, Area
"""

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.area import Area

__all__ = ['Brain', 'Area']
