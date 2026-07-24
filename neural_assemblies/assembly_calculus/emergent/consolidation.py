"""Backward-compatible shim — prefer ``training.consolidation``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.consolidation")
from .training.consolidation import *  # noqa: F401,F403
