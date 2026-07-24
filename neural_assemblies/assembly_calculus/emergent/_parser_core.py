"""Backward-compatible shim — prefer ``parser_mixins.core``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.core")
from .parser_mixins.core import *  # noqa: F401,F403
