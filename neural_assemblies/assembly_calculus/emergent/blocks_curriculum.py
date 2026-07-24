"""Backward-compatible shim — prefer ``curriculum.blocks``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.curriculum.blocks")
from .curriculum.blocks import *  # noqa: F401,F403
