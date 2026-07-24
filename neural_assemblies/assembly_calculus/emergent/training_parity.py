"""Backward-compatible shim — prefer ``evaluation.parity``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.evaluation.parity")
from .evaluation.parity import *  # noqa: F401,F403
