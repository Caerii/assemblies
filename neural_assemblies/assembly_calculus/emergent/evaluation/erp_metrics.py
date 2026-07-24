"""Backward-compatible shim — prefer ``evaluation.erp.gates``."""
from .._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.evaluation.erp.gates")
from .erp.gates import *  # noqa: F401,F403
