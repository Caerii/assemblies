"""Language → blocks-world: parse move commands and execute via STRIPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from neural_assemblies.programs.planning import (
    BlockState,
    BlocksAction,
    BlocksWorldPlanner,
    apply_strips_plan,
    bfs_plan,
    make_toy_problem,
)

from .structured_io import InstructionFrame, ToolCall


_BLOCK_ALIASES = {
    "a": "A", "b": "B", "c": "C", "d": "D",
    "blk_a": "A", "blk_b": "B", "blk_c": "C", "blk_d": "D",
    "block_a": "A", "block_b": "B", "block_c": "C", "block_d": "D",
}

_MOVE_VERBS = frozenset({"move", "put", "place", "stack"})
_PREP_TO = frozenset({"to", "on", "onto"})


def normalize_block_token(token: str) -> Optional[str]:
    """Map surface form → block id (A–D) or ``table``."""
    t = token.lower().strip()
    if t == "table":
        return "table"
    if t in _BLOCK_ALIASES:
        return _BLOCK_ALIASES[t]
    if len(t) == 1 and t.upper() in "ABCD":
        return t.upper()
    return None


def normalize_blocks_command(words: List[str]) -> List[str]:
    """Expand single-letter blocks to ``blk_*`` tokens when in move context."""
    if not words:
        return words
    if words[0].lower() not in _MOVE_VERBS:
        return words
    out = [words[0]]
    for w in words[1:]:
        blk = normalize_block_token(w)
        if blk and blk != "table":
            out.append(f"blk_{blk.lower()}")
        else:
            out.append(w)
    return out


def is_blocks_command(words: List[str]) -> bool:
    if not words:
        return False
    return words[0].lower() in _MOVE_VERBS


def parse_blocks_command(words: List[str]) -> Optional[InstructionFrame]:
    """Rule-based parse for ``move A to B`` / ``put A on B`` patterns."""
    lw = [w.lower() for w in words]
    if not lw or lw[0] not in _MOVE_VERBS:
        return None

    src = dst = None
    prep_idx = None
    for i, w in enumerate(lw):
        if w in _PREP_TO:
            prep_idx = i
            break

    if prep_idx is not None and prep_idx >= 1 and prep_idx + 1 < len(lw):
        src = normalize_block_token(lw[prep_idx - 1])
        dst = normalize_block_token(lw[prep_idx + 1])
    elif len(lw) >= 3:
        src = normalize_block_token(lw[1])
        dst = normalize_block_token(lw[2])

    if src is None:
        return None

    dst_id = "table" if dst is None else dst
    patient_surface = f"blk_{src.lower()}" if src != "table" else "table"
    dest_surface = "table" if dst_id == "table" else f"blk_{dst_id.lower()}"

    return InstructionFrame(
        intent=lw[0],
        mood="IMPERATIVE",
        agent="you",
        action=lw[0],
        patient=patient_surface,
        destination=dest_surface,
        polarity="AFFIRMATIVE",
        raw_words=list(words),
        categories={},
        roles={
            patient_surface: "PATIENT",
            dest_surface: "GOAL",
            "_block_src": src,
            "_block_dst": dst_id,
        },
    )


def frame_to_blocks_action(frame: InstructionFrame) -> Optional[BlocksAction]:
    """Map instruction frame → discrete ``BlocksAction``."""
    src = frame.roles.get("_block_src")
    dst = frame.roles.get("_block_dst")
    if src is None and frame.patient:
        src = normalize_block_token(frame.patient.replace("blk_", ""))
    if dst is None and frame.destination:
        dst = normalize_block_token(frame.destination.replace("blk_", ""))
    if src is None or dst is None:
        verb = (frame.action or frame.intent or "").lower()
        if verb not in _MOVE_VERBS:
            return None
        return None
    dst_arg = "table" if dst == "table" else dst
    return BlocksAction(src=src, dst=dst_arg)


def tool_call_to_blocks_action(call: ToolCall) -> Optional[BlocksAction]:
    if call.name != "move_block":
        return None
    block = call.arguments.get("block")
    dest = call.arguments.get("destination")
    if block is None or dest is None:
        return None
    src = normalize_block_token(str(block))
    dst = normalize_block_token(str(dest))
    if src is None or dst is None:
        return None
    return BlocksAction(src=src, dst="table" if dst == "table" else dst)


@dataclass
class BlocksLanguageExecutor:
    """Maintain blocks state; apply language-derived move commands."""

    blocks: Tuple[str, ...] = ("A", "B", "C")
    state: BlockState = field(default_factory=lambda: BlockState(
        on={"A": None, "B": None, "C": None},
        clear={"A": True, "B": True, "C": True},
    ))
    planner: BlocksWorldPlanner = field(init=False)

    def __post_init__(self) -> None:
        self.planner = BlocksWorldPlanner(blocks=list(self.blocks))

    @classmethod
    def from_toy_problem(cls) -> "BlocksLanguageExecutor":
        start, _ = make_toy_problem()
        ex = cls(blocks=("A", "B"))
        ex.state = start.copy()
        return ex

    def reset(self, state: BlockState) -> None:
        self.state = state.copy()

    def apply_action(self, action: BlocksAction) -> BlockState:
        self.state = apply_strips_plan(self.state, [action])
        return self.state.copy()

    def apply_frame(self, frame: InstructionFrame) -> Tuple[Optional[BlocksAction], BlockState]:
        action = frame_to_blocks_action(frame)
        if action is None:
            return None, self.state.copy()
        try:
            new_state = self.apply_action(action)
        except ValueError:
            return action, self.state.copy()
        return action, new_state

    def plan_to_goal(self, goal: BlockState) -> Optional[List[BlocksAction]]:
        return bfs_plan(self.state, goal)

    def describe(self) -> str:
        parts = []
        for blk, below in sorted(self.state.on.items()):
            if below is None:
                parts.append(f"{blk} on table")
            else:
                parts.append(f"{blk} on {below}")
        return "; ".join(parts) if parts else "empty"
