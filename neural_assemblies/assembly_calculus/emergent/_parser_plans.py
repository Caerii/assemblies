"""Backward-compatible shim — prefer ``parser_mixins.plans``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.plans")
from .parser_mixins.plans import *  # noqa: F401,F403
