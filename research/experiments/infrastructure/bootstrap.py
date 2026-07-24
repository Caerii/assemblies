"""Backward-compatible shim — prefer ``materialize_structural_weights``."""
from .materialize_structural_weights import (  # noqa: F401
    bootstrap_structural_connectivity,
    materialize_structural_connectivity,
)
