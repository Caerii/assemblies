"""Backward-compatible shim — prefer ``parser_mixins.structured``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.structured")
from .parser_mixins.structured import *  # noqa: F401,F403
