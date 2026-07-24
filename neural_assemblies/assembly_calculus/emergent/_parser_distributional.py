"""Backward-compatible shim — prefer ``parser_mixins.distributional``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.distributional")
from .parser_mixins.distributional import *  # noqa: F401,F403
