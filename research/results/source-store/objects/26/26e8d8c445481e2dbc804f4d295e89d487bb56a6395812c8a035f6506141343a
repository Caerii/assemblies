"""Structured records: validation, conversion, and language roundtrip."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .json_schemas import JsonSchema, SCHEMA_REGISTRY
from .structured_io import InstructionFrame, ToolCall, validate_tool_schema


@dataclass
class StructuredRecord:
    """Schema-tagged key-value record (JSON-serializable)."""

    schema: str
    data: Dict[str, Any]

    def to_dict(self) -> dict:
        return dict(self.data)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.data, **kwargs)

    @classmethod
    def from_json(cls, text: str, schema: str = "tool_call") -> "StructuredRecord":
        data = json.loads(text)
        return cls(schema=schema, data=data)

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> "StructuredRecord":
        return cls(schema="tool_call", data=call.to_dict())

    @classmethod
    def from_instruction_frame(cls, frame: InstructionFrame) -> "StructuredRecord":
        payload: Dict[str, Any] = {
            "action": frame.action,
            "mood": frame.mood,
        }
        if frame.agent:
            payload["agent"] = frame.agent
        if frame.patient:
            payload["patient"] = frame.patient
        if frame.destination:
            payload["destination"] = frame.destination
        return cls(schema="instruction", data=payload)


def validate_record(
    record: StructuredRecord,
    schema: Optional[JsonSchema] = None,
) -> Tuple[bool, List[str]]:
    """Validate *record* against a schema; return (ok, error messages)."""
    schema = schema or SCHEMA_REGISTRY.get(record.schema)
    if schema is None:
        return False, [f"unknown schema: {record.schema!r}"]

    errors: List[str] = []
    for key in schema.required:
        if key not in record.data or record.data[key] in (None, ""):
            errors.append(f"missing required field: {key}")

    for key, val in record.data.items():
        expected = schema.allowed_types.get(key)
        if expected is not None and val is not None and not isinstance(val, expected):
            errors.append(
                f"field {key!r}: expected {expected.__name__}, got {type(val).__name__}"
            )

    if record.schema == "tool_call" and record.data.get("tool") == "move_block":
        args = record.data.get("arguments", {})
        call = ToolCall(name="move_block", arguments=dict(args))
        if not validate_tool_schema(call, ("block", "destination")):
            errors.append("move_block missing block or destination in arguments")

    return len(errors) == 0, errors


def record_to_tool_call(record: StructuredRecord) -> Optional[ToolCall]:
    if "tool" not in record.data:
        return None
    return ToolCall.from_dict(record.data)


def _surface_block_token(token: str) -> str:
    t = str(token).lower()
    if t.startswith("blk_"):
        return t[4:]
    return t


def structured_to_words(record: StructuredRecord) -> List[str]:
    """Generate surface command words from a structured record."""
    if record.schema == "tool_call" or "tool" in record.data:
        call = record_to_tool_call(record)
        if call is None:
            return []
        args = call.arguments
        if call.name == "move_block":
            block = _surface_block_token(str(args.get("block", "a")))
            dest = _surface_block_token(str(args.get("destination", "table")))
            return ["move", block, "to", dest]
        if call.name == "chase":
            target = _surface_block_token(str(args.get("target", "")))
            return ["chases", "the", target] if target else ["chases"]
        if call.name == "observe":
            obj = _surface_block_token(str(args.get("object", "")))
            return ["sees", "the", obj] if obj else ["sees"]

    if record.schema == "instruction":
        d = record.data
        action = d.get("action")
        patient = d.get("patient")
        if action and patient:
            pat = _surface_block_token(str(patient))
            if pat == "table":
                return [str(action), pat]
            return [str(action), "the", pat]
        if action:
            return [str(action)]
    return []


def compare_tool_calls(a: Optional[ToolCall], b: Optional[ToolCall]) -> bool:
    if a is None or b is None:
        return a is b
    if a.name != b.name:
        return False
    keys = set(a.arguments) | set(b.arguments)
    for k in keys:
        if str(a.arguments.get(k)) != str(b.arguments.get(k)):
            return False
    return True
