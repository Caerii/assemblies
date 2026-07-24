"""Backward-compatible shim — prefer ``acquisition.pos_inference``."""
from .._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference")
from .pos_inference import *  # noqa: F401,F403
