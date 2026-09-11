"""
AAAI 2022 blocks-world planning (d'Amore et al.).

STRIPS search over discrete block states, with neural operator grounding via
``FiberCircuit``, per-block assemblies, and an FSM control scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from neural_assemblies.assembly_calculus import FiberCircuit, FSMNetwork, project
from neural_assemblies.assembly_calculus.ops import _snap


@dataclass(frozen=True)
class BlockState:
    """Blocks-world snapshot (toy STRIPS-style)."""

    on: Dict[str, Optional[str]]  # block -> block below or None for table
    clear: Dict[str, bool]

    def copy(self) -> "BlockState":
        return BlockState(dict(self.on), dict(self.clear))


@dataclass
class BlocksAction:
    """Move block ``src`` onto ``dst`` (block name or ``table``)."""

    src: str
    dst: str


def blocks_world_successors(state: BlockState) -> List[Tuple[BlocksAction, BlockState]]:
    """Legal STRIPS moves for a tiny blocks world."""
    moves = []
    for src, below in state.on.items():
        if not state.clear.get(src, False):
            continue
        for dst in list(state.on.keys()) + ["table"]:
            if dst == src:
                continue
            if dst != "table" and not state.clear.get(dst, False):
                continue
            nxt = state.copy()
            below = state.on[src]
            if below is not None:
                nxt.clear[below] = True
            if dst == "table":
                nxt.on[src] = None
            else:
                nxt.on[src] = dst
                nxt.clear[dst] = False
            nxt.clear[src] = True
            moves.append((BlocksAction(src, dst), nxt))
    return moves


def bfs_plan(
    start: BlockState,
    goal: BlockState,
    max_depth: int = 12,
) -> Optional[List[BlocksAction]]:
    """Breadth-first plan over discrete block states."""
    if start.on == goal.on:
        return []
    frontier = [(start, [])]
    seen = {frozenset(start.on.items())}
    while frontier:
        state, path = frontier.pop(0)
        if len(path) >= max_depth:
            continue
        for action, nxt in blocks_world_successors(state):
            key = frozenset(nxt.on.items())
            if key in seen:
                continue
            if nxt.on == goal.on:
                return path + [action]
            seen.add(key)
            frontier.append((nxt, path + [action]))
    return None


def apply_strips_plan(
    state: BlockState,
    plan: List[BlocksAction],
) -> BlockState:
    """Apply a discrete STRIPS plan and return the resulting state."""
    current = state.copy()
    for action in plan:
        matched = False
        for candidate, nxt in blocks_world_successors(current):
            if candidate.src == action.src and candidate.dst == action.dst:
                current = nxt
                matched = True
                break
        if not matched:
            raise ValueError(f"Illegal action {action} in state {current.on}")
    return current


class BlocksWorldAC:
    """Neural blocks-world: FSM control + fiber-gated pick/put operators."""

    def __init__(
        self,
        brain,
        blocks: List[str],
        n: int = 5000,
        k: int = 50,
        beta: float = 0.08,
        rounds: int = 8,
        prefix: str = "_bw",
    ):
        self.brain = brain
        self.blocks = blocks
        self.rounds = rounds
        self.prefix = prefix

        self.holding_area = f"{prefix}_hold"
        self.table_area = f"{prefix}_table"
        brain.add_area(self.holding_area, n, k, beta)
        brain.add_area(self.table_area, n, k, beta)

        states = ["search", "apply", "done"]
        symbols = (
            [f"pick_{b}" for b in blocks]
            + [f"put_{b}" for b in blocks]
            + ["put_table"]
        )
        transitions = []
        for b in blocks:
            transitions.append(("search", f"pick_{b}", "apply"))
            transitions.append(("apply", f"put_{b}", "search"))
        transitions.append(("apply", "put_table", "search"))

        self.fsm = FSMNetwork(
            brain, states, symbols, transitions, "search",
            n=n, k=k, beta=beta, rounds=rounds, prefix=f"{prefix}_fsm",
        )
        self.circuit = FiberCircuit(brain)
        self.circuit.add(self.holding_area, self.holding_area)
        self.circuit.add(self.holding_area, self.table_area)

        self._block_stim: Dict[str, str] = {}
        self._block_area: Dict[str, str] = {}
        for b in blocks:
            stim = f"{prefix}_block_{b}"
            area = f"{prefix}_blk_{b}"
            brain.add_stimulus(stim, k)
            brain.add_area(area, n, k, beta)
            self._block_stim[b] = stim
            self._block_area[b] = area
            self.circuit.add_stim(stim, self.holding_area)
            self.circuit.add(area, self.holding_area)
            project(brain, stim, area, rounds=rounds)
            brain.reset_area_connections(area)

        if blocks:
            project(brain, self._block_stim[blocks[0]], self.table_area, rounds=rounds)

        self._discrete_state: Optional[BlockState] = None

    def initialize_discrete_state(self, state: BlockState) -> None:
        """Set the tracked STRIPS state for neural plan execution."""
        self._discrete_state = state.copy()

    @property
    def discrete_state(self) -> Optional[BlockState]:
        return self._discrete_state

    def encode_action(self, action: BlocksAction) -> List[str]:
        put_sym = "put_table" if action.dst == "table" else f"put_{action.dst}"
        return [f"pick_{action.src}", put_sym]

    def execute_action(self, action: BlocksAction) -> None:
        """Run one STRIPS move through the neural operator scaffold."""
        if self._discrete_state is not None:
            self._discrete_state = apply_strips_plan(
                self._discrete_state, [action],
            )

        self.fsm.step(f"pick_{action.src}")
        src_area = self._block_area[action.src]
        self.brain.project({}, {src_area: [self.holding_area]})
        for _ in range(self.rounds - 1):
            self.brain.project(
                {},
                {
                    src_area: [self.holding_area],
                    self.holding_area: [self.holding_area],
                },
            )

        put_sym = "put_table" if action.dst == "table" else f"put_{action.dst}"
        self.fsm.step(put_sym)
        dst_area = (
            self.table_area
            if action.dst == "table"
            else self._block_area[action.dst]
        )
        self.brain.project({}, {self.holding_area: [dst_area]})
        for _ in range(self.rounds - 1):
            self.brain.project(
                {},
                {
                    self.holding_area: [dst_area],
                    dst_area: [dst_area],
                },
            )

    def run_plan(self, plan: List[BlocksAction]) -> List[str]:
        """Execute a discrete plan through FSM + neural operators."""
        self.fsm.reset()
        trajectory = [self.fsm.current_state]
        for action in plan:
            self.execute_action(action)
            trajectory.append(self.fsm.current_state)
        return trajectory

    def run_plan_from(
        self,
        start: BlockState,
        plan: List[BlocksAction],
    ) -> Tuple[List[str], BlockState]:
        """Execute *plan* from *start* and return FSM trajectory + final state."""
        self.initialize_discrete_state(start)
        trajectory = self.run_plan(plan)
        if self._discrete_state is None:
            raise RuntimeError("Discrete state not initialized")
        return trajectory, self._discrete_state.copy()

    def holding_snapshot(self):
        return _snap(self.brain, self.holding_area)


class BlocksWorldPlanner:
    """STRIPS search + neural execution."""

    def __init__(self, blocks: List[str], **ac_kwargs):
        self.blocks = blocks
        self.ac_kwargs = ac_kwargs
        self._model: Optional[BlocksWorldAC] = None

    def _ensure_model(self, brain) -> BlocksWorldAC:
        if self._model is None or self._model.brain is not brain:
            self._model = BlocksWorldAC(brain, self.blocks, **self.ac_kwargs)
        return self._model

    def plan(self, start: BlockState, goal: BlockState) -> Optional[List[BlocksAction]]:
        return bfs_plan(start, goal)

    def solve_and_run(
        self,
        brain,
        start: BlockState,
        goal: BlockState,
    ) -> Tuple[Optional[List[BlocksAction]], List[str], Optional[BlockState]]:
        """Search for a plan and execute it on ``brain``."""
        plan = self.plan(start, goal)
        if plan is None:
            return None, [], None
        model = self._ensure_model(brain)
        trajectory, final_state = model.run_plan_from(start, plan)
        return plan, trajectory, final_state

    def verify_plan(
        self,
        start: BlockState,
        goal: BlockState,
        plan: Optional[List[BlocksAction]],
    ) -> bool:
        """Return True when a discrete plan reaches ``goal`` from ``start``."""
        if plan is None:
            return start.on == goal.on
        try:
            end = apply_strips_plan(start, plan)
        except ValueError:
            return False
        return end.on == goal.on


def make_toy_problem() -> Tuple[BlockState, BlockState]:
    """Classic A-on-B-on-table → B-on-A-on-table."""
    start = BlockState(
        on={"A": "B", "B": None},
        clear={"A": True, "B": False},
    )
    goal = BlockState(
        on={"B": "A", "A": None},
        clear={"B": True, "A": False},
    )
    return start, goal


def make_three_block_problem() -> Tuple[BlockState, BlockState]:
    """A-on-B-on-C-on-table → C-on-B-on-A-on-table."""
    start = BlockState(
        on={"A": "B", "B": "C", "C": None},
        clear={"A": True, "B": False, "C": False},
    )
    goal = BlockState(
        on={"C": "B", "B": "A", "A": None},
        clear={"C": True, "B": False, "A": False},
    )
    return start, goal


def make_four_block_problem() -> Tuple[BlockState, BlockState]:
    """A-on-B-on-C-on-D → D-on-C-on-B-on-A (tower inversion)."""
    start = BlockState(
        on={"A": "B", "B": "C", "C": "D", "D": None},
        clear={"A": True, "B": False, "C": False, "D": False},
    )
    goal = BlockState(
        on={"D": "C", "C": "B", "B": "A", "A": None},
        clear={"D": True, "B": False, "C": False, "A": False},
    )
    return start, goal
