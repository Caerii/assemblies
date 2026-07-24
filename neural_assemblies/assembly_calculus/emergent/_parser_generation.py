"""Backward-compatible shim — prefer ``parser_mixins.generation``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.generation")
from .parser_mixins.generation import *  # noqa: F401,F403
