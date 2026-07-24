"""
Minimal Turing-machine style demo via ``FiberCircuit`` + ``FSMNetwork``.

Implements a unary increment machine: on input ``1`` symbols, transitions
q0 → q1 and extends a tape cell; blank ``_`` halts in q_halt.
"""

from __future__ import annotations

from typing import List

from neural_assemblies.assembly_calculus import FiberCircuit, FSMNetwork, project
from neural_assemblies.assembly_calculus.ops import _snap


class MinimalTMDemo:
    """Unary increment TM with neural state and tape areas."""

    def __init__(
        self,
        brain,
        n: int = 5000,
        k: int = 50,
        beta: float = 0.08,
        rounds: int = 8,
        prefix: str = "_tm",
    ):
        self.brain = brain
        self.rounds = rounds
        self.prefix = prefix

        self.tape_area = f"{prefix}_tape"
        self.head_area = f"{prefix}_head"
        brain.add_area(self.tape_area, n, k, beta)
        brain.add_area(self.head_area, n, k, beta)

        self._sym_one = f"{prefix}_one"
        self._sym_blank = f"{prefix}_blank"
        brain.add_stimulus(self._sym_one, k)
        brain.add_stimulus(self._sym_blank, k)

        project(brain, self._sym_one, self.tape_area, rounds=rounds)
        brain._engine.reset_area_connections(self.tape_area)
        project(brain, self._sym_blank, self.tape_area, rounds=rounds)
        brain._engine.reset_area_connections(self.tape_area)
        project(brain, self._sym_one, self.head_area, rounds=rounds)

        states = ["q0", "q1", "q_halt"]
        symbols = ["1", "_"]
        transitions = [
            ("q0", "1", "q1"),
            ("q1", "1", "q1"),
            ("q1", "_", "q_halt"),
        ]
        self.fsm = FSMNetwork(
            brain, states, symbols, transitions, "q0",
            n=n, k=k, beta=beta, rounds=rounds, prefix=f"{prefix}_fsm",
        )

        self.circuit = FiberCircuit(brain)
        self.circuit.add_stim(self._sym_one, self.tape_area)
        self.circuit.add_stim(self._sym_blank, self.tape_area)
        self.circuit.add(self.head_area, self.tape_area)
        self.circuit.add(self.tape_area, self.tape_area)

    def reset(self) -> None:
        self.fsm.reset()
        project(self.brain, self._sym_blank, self.tape_area, rounds=self.rounds)

    def step(self, symbol: str) -> str:
        """Read one tape symbol and advance TM state."""
        stim = self._sym_one if symbol == "1" else self._sym_blank
        self.circuit.disinhibit(stim, self.tape_area)
        other = self._sym_blank if symbol == "1" else self._sym_one
        self.circuit.inhibit(other, self.tape_area)
        self.circuit.step()
        return self.fsm.step(symbol)

    def run(self, tape: str) -> List[str]:
        """Run on a unary tape string (``1`` and ``_`` only)."""
        self.reset()
        trajectory = []
        for ch in tape:
            trajectory.append(self.step(ch))
        return trajectory

    def tape_snapshot(self):
        return _snap(self.brain, self.tape_area)
