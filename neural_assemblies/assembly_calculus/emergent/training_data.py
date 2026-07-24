"""Backward-compatible shim — prefer ``curriculum.data``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.curriculum.data")
from .curriculum.data import *  # noqa: F401,F403
