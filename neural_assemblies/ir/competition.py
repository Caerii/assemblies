"""Specification: neural_assemblies/ir/VERIFICATION.md#contract-competition-wire"""
from dataclasses import asdict

from ..compute.winner_policies import (
    EPercentPolicy, RelativeThresholdPolicy, ThresholdPolicy, TopKPolicy,
)
from .protocol import validate_schema_document

_POLICY_TYPES = {"top-k": TopKPolicy, "threshold": ThresholdPolicy,
                 "relative-threshold": RelativeThresholdPolicy, "e-percent": EPercentPolicy}
_COUNTS = ("k", "min_winners", "max_winners")


def policy_from_document(document):
    """Read a complete policy document; unknown fields and omitted defaults fail."""
    errors = validate_schema_document(document, "competition.schema.json")
    if errors:
        raise ValueError(f"invalid competition document: {errors}")
    for name in _COUNTS:
        if document.get(name) is not None and type(document[name]) is not int:
            raise ValueError(f"{name} must be a JSON integer, not a floating-point encoding")
    return _POLICY_TYPES[document["kind"]](
        **{key: value for key, value in document.items() if key not in ("profile", "kind")})


def policy_to_document(policy):
    """Export explicit policy settings after validating the complete document."""
    kinds = {cls: kind for kind, cls in _POLICY_TYPES.items()}
    if type(policy) not in kinds:
        raise ValueError("unsupported competition policy type")
    document = {"profile": "competition-v1", "kind": kinds[type(policy)], **asdict(policy)}
    policy_from_document(document)
    return document
