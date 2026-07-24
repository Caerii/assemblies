"""Backward-compatible shim — prefer ``core.areas``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.core.areas")
from .core.areas import *  # noqa: F401,F403
