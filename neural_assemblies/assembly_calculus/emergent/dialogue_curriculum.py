"""Backward-compatible shim — prefer ``curriculum.dialogue``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.curriculum.dialogue")
from .curriculum.dialogue import *  # noqa: F401,F403
