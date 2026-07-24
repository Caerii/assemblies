"""JSON schema definitions for structured agent output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class JsonSchema:
    """Lightweight schema for ``StructuredRecord`` validation."""

    name: str
    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    allowed_types: Dict[str, type] = field(default_factory=dict)

    @property
    def all_fields(self) -> FrozenSet[str]:
        return frozenset(self.required) | frozenset(self.optional)


TOOL_CALL_SCHEMA = JsonSchema(
    name="tool_call",
    required=("tool", "arguments"),
    allowed_types={"tool": str, "arguments": dict},
)

MOVE_BLOCK_SCHEMA = JsonSchema(
    name="move_block",
    required=("tool", "arguments"),
    optional=(),
    allowed_types={"tool": str, "arguments": dict},
)

INSTRUCTION_SCHEMA = JsonSchema(
    name="instruction",
    required=("action",),
    optional=("agent", "patient", "destination", "mood"),
    allowed_types={
        "action": str,
        "agent": str,
        "patient": str,
        "destination": str,
        "mood": str,
    },
)

SCHEMA_REGISTRY: Dict[str, JsonSchema] = {
    s.name: s for s in (TOOL_CALL_SCHEMA, MOVE_BLOCK_SCHEMA, INSTRUCTION_SCHEMA)
}
