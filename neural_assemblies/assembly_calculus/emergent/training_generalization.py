"""Backward-compatible shim — prefer ``evaluation.generalization``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.evaluation.generalization")
from .evaluation.generalization import *  # noqa: F401,F403
