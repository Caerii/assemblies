"""Backward-compatible shim — prefer ``training.perf``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.training.perf")
from .training.perf import *  # noqa: F401,F403
