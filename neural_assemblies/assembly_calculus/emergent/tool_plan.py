"""Multi-tool plans: sequential commands and goal-directed BFS."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from neural_assemblies.programs.planning import BlockState, BlocksAction

from .blocks_bridge import BlocksLanguageExecutor, normalize_block_token
from .structured_io import ToolCall


@dataclass
class ToolPlan:
    """Ordered tool invocations (explicit list or BFS-derived)."""

    steps: List[ToolCall] = field(default_factory=list)
    source: str = "explicit"  # "explicit" | "bfs"
    goal_description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "goal": self.goal_description,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "ToolPlan":
        data = json.loads(text)
        steps = [ToolCall.from_dict(s) for s in data.get("steps", [])]
        return cls(
            steps=steps,
            source=data.get("source", "explicit"),
            goal_description=data.get("goal"),
        )

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass
class ToolPlanResult:
    """Outcome of executing a ``ToolPlan``."""

    plan: ToolPlan
    executed: int
    results: List[str]
    final_state_description: str
    success: bool

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "success": self.success,
            "results": self.results,
            "final_state": self.final_state_description,
            "plan": self.plan.to_dict(),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


_GOAL_TRIGGERS = frozenset({"stack", "make", "build", "arrange"})
_COMMAND_SPLITTERS = re.compile(r"\s+then\s+|\s*;\s*|\s+and then\s+", re.IGNORECASE)


def tokenize_text(text: str) -> List[str]:
    text = text.strip().lower()
    text = re.sub(r"([?.!,])", r" \1", text)
    return [w for w in text.split() if w]


def split_command_clauses(text: str) -> List[str]:
    """Split multi-step instructions on ``then`` / ``;``."""
    text = text.strip().lower()
    if not text:
        return []
    parts = _COMMAND_SPLITTERS.split(text)
    return [p.strip() for p in parts if p.strip()]


def blocks_action_to_tool_call(action: BlocksAction) -> ToolCall:
    block = f"blk_{action.src.lower()}"
    dest = "table" if action.dst == "table" else f"blk_{action.dst.lower()}"
    return ToolCall(
        name="move_block",
        arguments={
            "block": block,
            "destination": dest,
            "src": action.src,
            "dst": action.dst,
        },
    )


def parse_simple_goal(text: str, blocks: List[str], current: BlockState) -> BlockState:
    """Parse ``a on b`` / ``a on table`` phrases into a goal ``BlockState``."""
    goal = current.copy()
    words = re.sub(r"([?.!,])", r" \1", text.strip().lower()).split()

    i = 0
    while i < len(words):
        if words[i] == "on" and i >= 1:
            top = normalize_block_token(words[i - 1])
            bottom = (
                normalize_block_token(words[i + 1])
                if i + 1 < len(words)
                else None
            )
            if top and top in goal.on:
                if bottom == "table" or bottom is None:
                    below = goal.on.get(top)
                    if below is not None:
                        goal.clear[below] = True
                    goal.on[top] = None
                    goal.clear[top] = True
                elif bottom in goal.on:
                    if goal.on.get(top) != bottom:
                        old_below = goal.on.get(top)
                        if old_below is not None:
                            goal.clear[old_below] = True
                        goal.on[top] = bottom
                        goal.clear[top] = True
                        goal.clear[bottom] = False
        i += 1

    for b in blocks:
        if b not in goal.on:
            goal.on[b] = None
            goal.clear[b] = True

    return goal


def is_goal_directed(text: str) -> bool:
    lw = text.strip().lower().split()
    return bool(lw) and lw[0] in _GOAL_TRIGGERS


def build_explicit_plan(parser, text: str) -> ToolPlan:
    """Parse semicolon/then-separated move commands into a tool plan."""
    steps: List[ToolCall] = []
    for clause in split_command_clauses(text):
        words = tokenize_text(clause)
        call = parser.words_to_tool_call(words)
        if call is not None:
            steps.append(call)
    return ToolPlan(steps=steps, source="explicit")


def build_bfs_plan(executor: BlocksLanguageExecutor, goal_text: str) -> Optional[ToolPlan]:
    """BFS from executor state to a language-specified goal."""
    goal = parse_simple_goal(
        goal_text,
        list(executor.blocks),
        executor.state,
    )
    discrete_plan = executor.plan_to_goal(goal)
    if discrete_plan is None:
        return None
    steps = [blocks_action_to_tool_call(a) for a in discrete_plan]
    return ToolPlan(steps=steps, source="bfs", goal_description=goal_text.strip())


def execute_tool_plan(
    executor: BlocksLanguageExecutor,
    plan: ToolPlan,
    registry=None,
) -> ToolPlanResult:
    """Run each step sequentially on the blocks executor."""
    results: List[str] = []
    executed = 0

    for call in plan.steps:
        if call.name != "move_block":
            results.append(f"skip {call.name}")
            continue
        from .blocks_bridge import tool_call_to_blocks_action

        action = tool_call_to_blocks_action(call)
        if action is None:
            results.append("invalid move_block")
            break
        try:
            executor.apply_action(action)
            results.append(f"ok {action.src} on {action.dst}")
            executed += 1
        except ValueError:
            results.append(f"illegal {action.src} to {action.dst}")
            break

    success = executed == len(plan.steps) and len(plan.steps) > 0
    return ToolPlanResult(
        plan=plan,
        executed=executed,
        results=results,
        final_state_description=executor.describe(),
        success=success,
    )
