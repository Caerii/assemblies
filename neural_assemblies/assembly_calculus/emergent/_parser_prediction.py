"""Backward-compatible shim — prefer ``parser_mixins.prediction``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.parser_mixins.prediction")
from .parser_mixins.prediction import *  # noqa: F401,F403
