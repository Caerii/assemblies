"""Backward-compatible shim — prefer ``training.compiler``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.compiler")
from .training.compiler import *  # noqa: F401,F403
