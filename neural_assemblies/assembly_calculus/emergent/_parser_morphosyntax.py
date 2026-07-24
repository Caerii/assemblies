"""Backward-compatible shim — prefer ``parser_mixins.morphosyntax``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.morphosyntax")
from .parser_mixins.morphosyntax import *  # noqa: F401,F403
