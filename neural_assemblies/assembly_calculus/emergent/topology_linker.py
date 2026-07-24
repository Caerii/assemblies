"""Backward-compatible shim — prefer ``training.linker``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.linker")
from .training.linker import *  # noqa: F401,F403
