"""StructuredMixin — language ↔ JSON structured output.

The application-facing end of the pipeline: it turns a parse into a tool call
or JSON record, and back.  See ``..structured_io`` for what is lost in that
conversion -- in short, everything graded.  A record says which role held
which word; it cannot say how confidently, so parse-quality diagnostics must
be taken before this layer.

Known nit: ``words_to_tool_call`` annotates its return as ``Optional["ToolCall"]``
but ``ToolCall`` is not imported here.  Harmless at runtime under
``from __future__ import annotations`` (the annotation is never evaluated),
but ``typing.get_type_hints`` on this method would raise.  Left alone since
fixing it means touching imports for no behavioural gain.
"""

from __future__ import annotations

from typing import List, Optional

from ..structured_json import (
    StructuredRecord,
    compare_tool_calls,
    structured_to_words,
    validate_record,
)
from ..tools import ToolRegistry


class StructuredMixin:
    """Emit and consume JSON tool/instruction records from language."""

    def words_to_tool_call(self, words: List[str]) -> Optional["ToolCall"]:
        frame = self.parse_instruction(words)
        return ToolRegistry().dispatch(frame)

    def words_to_structured(self, words: List[str]) -> Optional[StructuredRecord]:
        call = self.words_to_tool_call(words)
        if call is not None:
            return StructuredRecord.from_tool_call(call)
        frame = self.parse_instruction(words)
        if frame.action:
            return StructuredRecord.from_instruction_frame(frame)
        return None

    def words_to_json(self, words: List[str]) -> Optional[str]:
        record = self.words_to_structured(words)
        if record is None:
            return None
        ok, _ = validate_record(record)
        if not ok and record.schema == "tool_call":
            return None
        return record.to_json()

    def json_to_words(self, text: str) -> List[str]:
        record = StructuredRecord.from_json(text, schema="tool_call")
        return structured_to_words(record)

    def structured_roundtrip(self, words: List[str]) -> dict:
        """Language → JSON → words → tool call; report match metrics."""
        call_a = self.words_to_tool_call(words)
        record = self.words_to_structured(words)
        if record is None:
            return {
                "success": False,
                "json": None,
                "roundtrip_words": [],
                "tool_match": False,
            }

        json_text = record.to_json()
        ok, errors = validate_record(record)
        roundtrip_words = self.json_to_words(json_text)
        call_b = self.words_to_tool_call(roundtrip_words)

        return {
            "success": ok and compare_tool_calls(call_a, call_b),
            "json": json_text,
            "schema_valid": ok,
            "schema_errors": errors,
            "roundtrip_words": roundtrip_words,
            "tool_match": compare_tool_calls(call_a, call_b),
            "call_a": call_a.to_dict() if call_a else None,
            "call_b": call_b.to_dict() if call_b else None,
        }

    def parse_json_command(self, text: str) -> StructuredRecord:
        record = StructuredRecord.from_json(text, schema="tool_call")
        ok, errors = validate_record(record)
        if not ok:
            raise ValueError("; ".join(errors))
        return record
