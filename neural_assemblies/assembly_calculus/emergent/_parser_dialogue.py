"""Backward-compatible shim — prefer ``parser_mixins.dialogue``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.dialogue")
from .parser_mixins.dialogue import *  # noqa: F401,F403
