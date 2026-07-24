"""Tool registry and dispatch from parsed instruction frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .structured_io import InstructionFrame, ToolCall, validate_tool_schema


@dataclass
class ToolSpec:
    """Maps a verb lemma to a tool name and argument slot bindings."""

    name: str
    verb_lemmas: tuple[str, ...]
    arg_slots: Dict[str, str]  # tool_arg -> frame field ("action", "patient", ...)
    required_args: tuple[str, ...] = ()
    description: str = ""


# Default closed-vocabulary tools aligned with toy training verbs.
DEFAULT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="chase",
        verb_lemmas=("chase", "chases", "chasing"),
        arg_slots={"agent": "agent", "target": "patient"},
        required_args=("target",),
        description="Pursue a target entity",
    ),
    ToolSpec(
        name="observe",
        verb_lemmas=("see", "sees", "find", "finds", "watch", "watches", "reads"),
        arg_slots={"agent": "agent", "object": "patient"},
        required_args=("object",),
        description="Perceive or locate an object",
    ),
    ToolSpec(
        name="move_block",
        verb_lemmas=("move", "put", "place", "stack"),
        arg_slots={"block": "patient", "destination": "destination"},
        required_args=("block", "destination"),
        description="Move a block onto another block or the table",
    ),
)


class ToolRegistry:
    """Verb→tool routing with optional custom handlers."""

    def __init__(self, specs: Optional[List[ToolSpec]] = None):
        self.specs = list(specs or DEFAULT_TOOL_SPECS)
        self._verb_index: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, Callable[[ToolCall], object]] = {}
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self._verb_index.clear()
        for spec in self.specs:
            for lemma in spec.verb_lemmas:
                self._verb_index[lemma.lower()] = spec

    def register_handler(self, tool_name: str, handler: Callable[[ToolCall], object]) -> None:
        self._handlers[tool_name] = handler

    def lookup_spec(self, verb: Optional[str]) -> Optional[ToolSpec]:
        if not verb:
            return None
        return self._verb_index.get(verb.lower())

    def dispatch(self, frame: InstructionFrame) -> Optional[ToolCall]:
        """Map an instruction frame to a tool call, or None if no match."""
        from .blocks_bridge import is_blocks_command

        if is_blocks_command(frame.raw_words):
            blocks_call = dispatch_blocks_tool(frame)
            if blocks_call is not None:
                return blocks_call

        verb = frame.action or frame.intent
        spec = self.lookup_spec(verb)
        if spec is None:
            return None

        arguments: Dict[str, object] = {}
        for arg_name, frame_field in spec.arg_slots.items():
            val = getattr(frame, frame_field, None)
            if val is not None:
                arguments[arg_name] = val

        # Imperatives: implicit addressee when agent missing
        if "agent" in spec.arg_slots and "agent" not in arguments:
            if frame.agent:
                arguments["agent"] = frame.agent
            elif frame.mood == "IMPERATIVE":
                arguments["agent"] = "you"

        call = ToolCall(name=spec.name, arguments=arguments)
        if spec.required_args and not validate_tool_schema(call, spec.required_args):
            return None
        return call

    def execute(self, call: ToolCall) -> object:
        """Run a registered handler or return the call unchanged."""
        handler = self._handlers.get(call.name)
        if handler is None:
            return call
        return handler(call)


def dispatch_blocks_tool(frame: InstructionFrame) -> Optional[ToolCall]:
    """Dispatch move commands using blocks frame slots."""
    from .blocks_bridge import frame_to_blocks_action

    action = frame_to_blocks_action(frame)
    if action is None:
        return None
    return ToolCall(
        name="move_block",
        arguments={
            "block": frame.patient,
            "destination": frame.destination or "table",
            "src": action.src,
            "dst": action.dst,
        },
    )
