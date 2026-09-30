"""Historical ``import brain`` shim.

Daniel Mitropolsky's original library lived at the repository root:
``brain.py`` (the engine), ``brain_util.py``, ``learner.py``, ``parser.py``,
``recursive_parser.py``, ``simulations.py`` and the MATLAB prototypes, with
every script reaching the engine by ``import brain``. Those modules are
archived under ``legacy/root_modules/``; this re-export keeps ``import
brain`` working with ``legacy/root_shims`` on ``PYTHONPATH``. The supported
API is the package:

    from neural_assemblies.core import Brain, Area
"""

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.area import Area

__all__ = ['Brain', 'Area']
