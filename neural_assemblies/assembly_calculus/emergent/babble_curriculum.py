"""Backward-compatible shim — prefer ``acquisition.babble``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.acquisition.babble")
from .acquisition.babble import *  # noqa: F401,F403
