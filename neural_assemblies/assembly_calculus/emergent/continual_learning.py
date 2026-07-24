"""Backward-compatible shim — prefer ``acquisition.continual``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.acquisition.continual")
from .acquisition.continual import *  # noqa: F401,F403
