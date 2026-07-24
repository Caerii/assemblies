"""FSMNetwork numpy port (mdabagia/nemo ``brain.py``)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .areas import FFArea, RefractedArea


class FSMNetwork:
    """Symbol + state areas coupled through a refracted arc area."""

    def __init__(
        self,
        n_symbol_neurons: int,
        n_state_neurons: int,
        n_arc_neurons: int,
        cap_size: int,
        density: float,
        plasticity: float,
        rng: np.random.Generator,
        *,
        norm_init: bool = False,
    ):
        self.cap_size = cap_size
        self.state_area = FFArea(
            n_arc_neurons, n_state_neurons, cap_size, density, plasticity, rng,
            norm_init=norm_init,
        )
        self.arc_area = RefractedArea(
            [n_symbol_neurons, n_state_neurons],
            n_arc_neurons,
            cap_size,
            density,
            plasticity,
            rng,
            norm_init=norm_init,
        )

    def inhibit(self) -> None:
        self.state_area.inhibit()
        self.arc_area.inhibit()

    def forward(self, symbol, *, update: bool = True) -> None:
        """One FSM step: arc(symbol, current state) → next state."""
        self.arc_area.forward([symbol, self.state_area.read()], update=update)
        self.state_area.forward(self.arc_area.read(), update=update)

    def read(self) -> np.ndarray:
        return self.state_area.read()

    def train(self, symbol, state, new_state) -> None:
        self.inhibit()
        self.arc_area.forward([symbol, state])
        self.state_area.set_input(self.arc_area.read())
        self.state_area.fire(new_state)


def build_mod3_symbols_states(
    cap_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference notebook symbol/state assembly arrays."""
    n_symbols = 11
    n_states = 5
    symbols = np.arange(n_symbols * cap_size, dtype=int).reshape(n_symbols, cap_size)
    states = np.arange(n_states * cap_size, dtype=int).reshape(n_states, cap_size)
    return symbols, states


def mod3_transition_list() -> List[List[int]]:
    transitions: List[List[int]] = []
    for mod in range(3):
        for digit in range(10):
            transitions.append([mod, digit, (mod + digit) % 3])
    transitions.extend([[0, 10, 3], [1, 10, 4], [2, 10, 4]])
    return transitions


def run_mod3_fsm_numpy(
    *,
    seed: int = 42,
    n_symbol_neurons: int = 1000,
    n_state_neurons: int = 500,
    n_arc_neurons: int = 5000,
    cap_size: int = 70,
    density: float = 0.2,
    plasticity: float = 0.1,
    presentations: int = 15,
    positive_sequence: Sequence[int] = (3, 0, 4, 7, 1, 10),
    negative_sequence: Sequence[int] = (6, 7, 3, 10),
) -> dict:
    """Train and test mod-3 FSM using reference numpy dynamics."""
    rng = np.random.default_rng(seed)
    fsm = FSMNetwork(
        n_symbol_neurons, n_state_neurons, n_arc_neurons,
        cap_size, density, plasticity, rng,
    )
    symbols, states = build_mod3_symbols_states(cap_size, rng)
    transition_list = mod3_transition_list()

    for _ in range(presentations):
        for transition in transition_list:
            fsm.train(
                symbols[transition[1]],
                states[transition[0]],
                states[transition[2]],
            )

    def _run(sequence: Sequence[int]) -> str:
        labels = {0: "0", 1: "1", 2: "2", 3: "accept", 4: "reject"}
        fsm.inhibit()
        fsm.state_area.fire(states[0], update=False)
        for sym_idx in sequence:
            fsm.forward(symbols[sym_idx], update=False)
        read = fsm.read()
        best_idx = max(
            range(len(states)),
            key=lambda i: len(np.intersect1d(read, states[i])) / max(len(read), 1),
        )
        return labels[best_idx]

    pos_final = _run(positive_sequence)
    neg_final = _run(negative_sequence)
    return {
        "positive_accepted": pos_final == "accept",
        "negative_rejected": neg_final == "reject",
        "positive_final_state": pos_final,
        "negative_final_state": neg_final,
    }
