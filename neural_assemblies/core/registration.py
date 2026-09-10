"""Shared population registration contract, independent of numerical storage."""
from numbers import Integral


def validate_area_registration(name, n, k, *, existing=()) -> tuple[int, int]:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-area-registration

    Canonicalize counts without allocating or consuming random state. The shared
    neuron-ID boundary uses uint32, so the largest population has 2**32 IDs.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("area name must be a nonempty string")
    if name in existing:
        raise ValueError(f"area {name!r} is already registered")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (n, k)):
        raise ValueError("area n and k must be nonboolean integers")
    if not 0 < k <= n <= 2**32:
        raise ValueError("area dimensions require 0 < k <= n <= 2**32")
    return int(n), int(k)
