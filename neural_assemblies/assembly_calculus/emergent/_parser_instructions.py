"""Backward-compatible shim — prefer ``parser_mixins.instructions``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.instructions")
from .parser_mixins.instructions import *  # noqa: F401,F403
