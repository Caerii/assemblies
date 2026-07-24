"""Backward-compatible shim — prefer ``curriculum.conversation``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.curriculum.conversation")
from .curriculum.conversation import *  # noqa: F401,F403
