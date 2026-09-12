"""JSON-serializable frames for instruction following and tool calling.

THE BOUNDARY THIS FILE MARKS.  Everything upstream of here is assemblies:
words are winner sets, roles are areas, and there are no symbols anywhere in
the representation.  Everything downstream -- tool dispatch, the blocks
executor, JSON output -- needs discrete named values.  An
:class:`InstructionFrame` is where the conversion happens, and it is worth
being explicit that the conversion is LOSSY BY DESIGN.

The frame's fields mirror the parser's thematic role areas (agent, action,
patient, destination) plus the morphosyntactic feature areas (mood,
polarity).  Filling a field means: read out the assembly in the corresponding
area against a lexicon and take the winning label.  What is discarded in that
step is everything graded -- how strongly the role was bound, how close the
runner-up was, whether the parse was unstable.  A frame that reads
``agent="dog"`` looks equally confident whether the readout won by 0.8 or by
0.01.

So a frame is a commitment, not a representation.  Diagnostics about parse
quality (ERP measures, binding strength, instability) must be taken from the
assemblies before this point; they cannot be recovered from the frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InstructionFrame:
    """Semantic frame extracted from user language."""

    speaker: str = "user"
    intent: Optional[str] = None
    mood: str = "DECLARATIVE"
    agent: Optional[str] = None
    action: Optional[str] = None
    patient: Optional[str] = None
    destination: Optional[str] = None
    polarity: str = "AFFIRMATIVE"
    raw_words: list[str] = field(default_factory=list)
    categories: Dict[str, str] = field(default_factory=dict)
    roles: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "InstructionFrame":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolCall:
    """Structured tool invocation (JSON-tool-calling compatible shape)."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tool": self.name, "arguments": self.arguments}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        if "tool" in data:
            return cls(name=data["tool"], arguments=dict(data.get("arguments", {})))
        return cls(name=data["name"], arguments=dict(data.get("arguments", {})))


def validate_tool_schema(call: ToolCall, required_args: tuple[str, ...]) -> bool:
    """Return True when all required argument keys are present and non-empty."""
    for key in required_args:
        val = call.arguments.get(key)
        if val is None or val == "":
            return False
    return True
