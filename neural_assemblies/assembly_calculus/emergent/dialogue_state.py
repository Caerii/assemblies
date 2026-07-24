"""Backward-compatible shim — prefer ``session.dialogue_state``."""
from ._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.session.dialogue_state")
from .session.dialogue_state import *  # noqa: F401,F403
