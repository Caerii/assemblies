"""Backward-compatible shim — prefer ``core.grounding``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.core.grounding")
from .core.grounding import *  # noqa: F401,F403
