"""Backward-compatible shim — prefer ``parser_mixins.incremental``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.incremental")
from .parser_mixins.incremental import *  # noqa: F401,F403
