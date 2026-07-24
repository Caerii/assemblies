"""Backward-compatible shim — prefer ``training.schedule``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.schedule")
from .training.schedule import *  # noqa: F401,F403
