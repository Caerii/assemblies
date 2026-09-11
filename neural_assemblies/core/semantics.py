"""Validated runtime policies that guard scientifically distinct semantics.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-sampled-recurrence
"""

from enum import Enum


class SampledRecurrencePolicy(str, Enum):
    """Admission policy for recurrence over an unmaterialized connectome."""

    WARN = "warn"
    ACKNOWLEDGED = "acknowledged"
    FORBID = "forbid"

    @classmethod
    def normalize(cls, value: object) -> "SampledRecurrencePolicy":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                "sampled_recurrence_policy must be 'warn', 'acknowledged', or 'forbid'"
            )
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                "sampled_recurrence_policy must be 'warn', 'acknowledged', or 'forbid'"
            ) from exc
