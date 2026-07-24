"""Backward-compatible shim — prefer ``core.corpus_index``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.core.corpus_index")
from .core.corpus_index import *  # noqa: F401,F403
