"""Repository configuration identity; see research/README.md#environment-identity.

This detects configuration differences. Hashes cannot reconstruct configuration
and do not certify the environment already captured by imported native modules.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable

ENVIRONMENT_POLICY = "repository-environment-v1"
ENVIRONMENT_PREFIXES = ("ASSEMBLIES_", "NEURAL_ASSEMBLIES_", "EMERGENT_")


def environment_signature(*, exclude: Iterable[str] = ()) -> tuple[tuple[str, str], ...]:
    """Snapshot exact values without storing them; absent and empty differ.

    Consumers may exclude settings only when those settings cannot affect the
    object they identify. Experiment records use the complete repository scope.
    """
    excluded = frozenset(exclude)
    return tuple(sorted(
        (name, hashlib.sha256(value.encode("utf-8")).hexdigest())
        for name, value in os.environ.copy().items()
        if name.startswith(ENVIRONMENT_PREFIXES) and name not in excluded
    ))


def environment_record() -> dict:
    return {"policy": ENVIRONMENT_POLICY,
            "variables_sha256": dict(environment_signature())}
