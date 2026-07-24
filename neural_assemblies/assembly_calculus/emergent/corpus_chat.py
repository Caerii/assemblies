"""Backward-compatible shim — prefer ``session.novel_chat``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.session.novel_chat")
from .session.novel_chat import *  # noqa: F401,F403
