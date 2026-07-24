"""Backward-compatible shim — prefer ``training.batch``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.batch")
from .training.batch import *  # noqa: F401,F403
