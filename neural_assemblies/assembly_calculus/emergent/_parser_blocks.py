"""Backward-compatible shim — prefer ``parser_mixins.blocks``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.blocks")
from .parser_mixins.blocks import *  # noqa: F401,F403
