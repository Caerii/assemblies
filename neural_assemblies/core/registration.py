"""Shared population registration contract, independent of numerical storage."""
from numbers import Integral


def validate_area_registration(name, n, k, *, existing=(), reserved=()) -> tuple[int, int]:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-area-registration

    Canonicalize counts without allocating or consuming random state. The shared
    neuron-ID boundary uses uint32, so the largest population has 2**32 IDs.
    """
    _validate_name(name, "area", existing, reserved)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (n, k)):
        raise ValueError("area n and k must be nonboolean integers")
    if not 0 < k <= n <= 2**32:
        raise ValueError("area dimensions require 0 < k <= n <= 2**32")
    return int(n), int(k)


def validate_slot_configuration(n, slot_count, winner_policy=None) -> int:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-slot-options"""
    if (isinstance(slot_count, bool) or not isinstance(slot_count, Integral)
            or not 0 <= slot_count <= n):
        raise ValueError("slot_count must be an integer between zero and the population size")
    if slot_count > 1:
        if n % slot_count:
            raise ValueError("slots must partition the whole population evenly")
        if winner_policy is not None:
            raise NotImplementedError("custom winner policies combined with multiple slots are unsupported")
    return int(slot_count)


def _validate_name(name, kind, existing, reserved):
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} name must be a nonempty string")
    if name in existing or name in reserved:
        raise ValueError(f"{kind} {name!r} is already registered or conflicts with another node")


def validate_stimulus_registration(name, size, *, existing=(), reserved=()) -> int:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-stimulus-registration"""
    _validate_name(name, "stimulus", existing, reserved)
    if isinstance(size, bool) or not isinstance(size, Integral) or not 0 <= size <= 2**32:
        raise ValueError("stimulus size must be a nonboolean integer in [0, 2**32]")
    return int(size)
