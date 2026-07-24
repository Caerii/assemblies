"""Backward-compatible shim — prefer ``training.compiled``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.compiled")
from .training.compiled import *  # noqa: F401,F403
