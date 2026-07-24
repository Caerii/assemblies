"""PlansMixin — multi-tool plans and goal-directed BFS."""

from __future__ import annotations

from typing import List, Optional

from ..blocks_bridge import BlocksLanguageExecutor
from ..tool_plan import (
    ToolPlan,
    ToolPlanResult,
    build_bfs_plan,
    build_explicit_plan,
    execute_tool_plan,
    is_goal_directed,
    split_command_clauses,
    tokenize_text,
)


class PlansMixin:
    """Multi-step tool plans: explicit commands or blocks-world BFS."""

    def text_to_tool_plan(
        self,
        text: str,
        executor: Optional[BlocksLanguageExecutor] = None,
    ) -> Optional[ToolPlan]:
        """Build a tool plan from natural language."""
        text = text.strip().lower()
        if not text:
            return None

        if is_goal_directed(text):
            if executor is None:
                executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
            return build_bfs_plan(executor, text)

        if split_command_clauses(text):
            plan = build_explicit_plan(self, text)
            if plan.length > 0:
                return plan

        call = self.words_to_tool_call(tokenize_text(text))
        if call is not None:
            from ..tool_plan import ToolPlan as TP

            return TP(steps=[call], source="explicit")
        return None

    def plan_to_json(self, plan: ToolPlan) -> str:
        return plan.to_json()

    def json_to_plan(self, text: str) -> ToolPlan:
        return ToolPlan.from_json(text)

    def execute_plan(
        self,
        plan: ToolPlan,
        executor: Optional[BlocksLanguageExecutor] = None,
    ) -> ToolPlanResult:
        if executor is None:
            executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        return execute_tool_plan(executor, plan)
