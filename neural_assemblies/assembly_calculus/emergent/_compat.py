"""Deprecation helpers for backward-compatible import shims."""

from __future__ import annotations

import warnings


def deprecate_shim(module_name: str, target: str) -> None:
    """Emit a DeprecationWarning for a legacy import path."""
    warnings.warn(
        f"{module_name} is deprecated; import from {target} instead",
        DeprecationWarning,
        stacklevel=3,
    )
