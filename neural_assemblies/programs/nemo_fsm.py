"""
NEMO arc-area FSM and Markov PFA — mdabagia/nemo port.

Reference: dabagia.org/nemo/sequences/, dabagia.org/nemo/coinflipping/
           .reference/mdabagia-nemo/brain.py (FSMNetwork, PFANetwork)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap, project
from neural_assemblies.assembly_calculus.pfa import RandomChoiceArea
from neural_assemblies.assembly_calculus.transitions import TransitionLike, TransitionMap


class NemoArcFSM:
    """Symbol + state + refracted arc FSM (reference FSMNetwork)."""

    def __init__(
        self,
        brain,
        states: List[str],
        symbols: List[str],
        transitions: List[TransitionLike],
        *,
        n: int = 5000,
        k: int = 80,
        beta: float = 0.1,
        rounds: int = 10,
        refracted_strength: float = 0.1,
        prefix: str = "_nemo_fsm",
    ):
        self.brain = brain
        self.states = list(states)
        self.symbols = list(symbols)
        self.transition_map = TransitionMap(transitions)
        self.rounds = rounds
        self.prefix = prefix

        self.state_area = f"{prefix}_state"
        self.symbol_area = f"{prefix}_symbol"
        self.arc_area = f"{prefix}_arc"

        brain.add_area(self.state_area, n, k, beta)
        brain.add_area(self.symbol_area, n, k, beta)
        brain.add_area(
            self.arc_area, n, k, beta,
            refracted=True, refracted_strength=refracted_strength,
        )

        self._sym_stim: Dict[str, str] = {}
        self._st_stim: Dict[str, str] = {}
        for sym in symbols:
            s = f"{prefix}_sym_{sym}"
            brain.add_stimulus(s, k)
            self._sym_stim[sym] = s
            project(brain, s, self.symbol_area, rounds=rounds)

        for st in states:
            s = f"{prefix}_st_{st}"
            brain.add_stimulus(s, k)
            self._st_stim[st] = s
            project(brain, s, self.state_area, rounds=rounds)

        self._table: Dict[Tuple[str, str], str] = (
            self.transition_map.deterministic_table()
        )

    def reset(self) -> None:
        self.brain.clear_refracted_bias(self.arc_area)

    def _project_arc(self) -> None:
        self.brain.project(
            {},
            {self.symbol_area: [self.arc_area], self.state_area: [self.arc_area]},
        )

    def train_transition(self, symbol: str, from_state: str, to_state: str) -> None:
        """Fire symbol+state into refracted arc, bind next state."""
        self.reset()
        project(self.brain, self._sym_stim[symbol], self.symbol_area, rounds=1)
        project(self.brain, self._st_stim[from_state], self.state_area, rounds=1)
        self._project_arc()
        self.brain.project({}, {self.arc_area: [self.state_area]})
        project(self.brain, self._st_stim[to_state], self.state_area, rounds=self.rounds)

    def step_symbol(self, symbol: str, from_state: str) -> str:
        """One FSM step: (from_state, symbol) → next state assembly."""
        to_state = self._table[(from_state, symbol)]
        project(self.brain, self._sym_stim[symbol], self.symbol_area, rounds=1)
        project(self.brain, self._st_stim[from_state], self.state_area, rounds=1)
        self._project_arc()
        self.brain.project({}, {self.arc_area: [self.state_area]})
        project(self.brain, self._st_stim[to_state], self.state_area, rounds=self.rounds)
        return to_state

    def train_from_list(
        self, transition_list: List[Tuple[str, str, str]], presentations: int = 1,
    ) -> None:
        for _ in range(presentations):
            for sym, fr, to in transition_list:
                self.train_transition(sym, fr, to)


class NemoMarkovPFA:
    """Markov PFA with refracted arc + RandomChoiceArea (reference PFANetwork).

    Alternates inhibition between current and next state areas during sampling,
    matching the dabagia.org coinflipping Markov architecture description.
    """

    def __init__(
        self,
        brain,
        states: List[str],
        transitions: List[Tuple[str, str, str, float]],
        initial_state: str,
        *,
        symbol: str = "flip",
        n: int = 5000,
        k: int = 80,
        beta: float = 0.1,
        refracted_strength: float = 0.1,
        flip_mode: str = "compete",
        prefix: str = "_nemo_pfa",
    ):
        self.brain = brain
        self.states = list(states)
        self.symbol = symbol
        self.k = k
        self._current = initial_state
        self._initial_state = initial_state
        self._active_is_current = True
        self.flip_mode = flip_mode

        self.current_area = f"{prefix}_current"
        self.next_area = f"{prefix}_next"
        self.symbol_area = f"{prefix}_symbol"
        self.arc_area = f"{prefix}_arc"

        brain.add_area(self.current_area, n, k, beta)
        brain.add_area(self.next_area, n, k, beta)
        brain.add_area(self.symbol_area, n, k, beta)
        brain.add_area(
            self.arc_area, n, k, beta,
            refracted=True, refracted_strength=refracted_strength,
        )

        self._st_stim: Dict[str, str] = {}
        self._state_winners: Dict[str, np.ndarray] = {}
        for st in states:
            s = f"{prefix}_st_{st}"
            brain.add_stimulus(s, k)
            self._st_stim[st] = s
            asm = project(brain, s, self.current_area, rounds=8)
            self._state_winners[st] = asm.winners.copy()
            project(brain, s, self.next_area, rounds=8)

        sym_stim = f"{prefix}_sym_{symbol}"
        brain.add_stimulus(sym_stim, k)
        self._sym_stim = sym_stim
        project(brain, sym_stim, self.symbol_area, rounds=8)

        self._coin = RandomChoiceArea(
            brain, area_name="rand", n=n, k=k, beta=beta, prefix=f"{prefix}_coin",
        )

        self._trans: Dict[str, List[Tuple[str, float]]] = {}
        for fr, sym, to, prob in transitions:
            if sym != symbol:
                continue
            self._trans.setdefault(fr, []).append((to, prob))

        self._train_from_transitions()

    def _train_from_transitions(self) -> None:
        for fr, targets in self._trans.items():
            if len(targets) == 1:
                self.train_transition(fr, 0, targets[0][0])
            elif len(targets) >= 2:
                self.train_transition(fr, 0, targets[0][0])
                self.train_transition(fr, 1, targets[1][0])

    @property
    def current_state(self) -> str:
        return self._current

    def _active_area(self) -> str:
        return self.current_area if self._active_is_current else self.next_area

    def _inactive_area(self) -> str:
        return self.next_area if self._active_is_current else self.current_area

    def _inhibit_area(self, area: str) -> None:
        b = self.brain
        b.areas[area].unfix_assembly()
        b._engine.set_winners(area, np.array([], dtype=np.uint32))

    def _cue_state(self, area: str, state: str) -> None:
        winners = self._state_winners[state].astype(np.uint32)
        b = self.brain
        b.areas[area].unfix_assembly()
        b.areas[area]._winners = winners
        b._engine.set_winners(area, winners)
        b.areas[area].fix_assembly()

    def reset(self) -> None:
        self.brain.clear_refracted_bias(self.arc_area)
        self._current = self._initial_state
        self._active_is_current = True

    def train_transition(
        self, from_state: str, coin_bit: int, to_state: str, *, rounds: int = 8,
    ) -> None:
        """Train one probabilistic branch (coin_bit 0 or 1 selects attractor)."""
        self.brain.clear_refracted_bias(self.arc_area)
        coin_asm = self._coin.asm0 if coin_bit == 0 else self._coin.asm1
        winners = coin_asm.winners.copy()

        self._cue_state(self.current_area, from_state)
        b = self.brain
        b.areas[self.arc_area]._winners = winners.astype(np.uint32)
        b._engine.set_winners(self.arc_area, winners.astype(np.uint32))
        b.project({}, {self.current_area: [self.arc_area]})
        b.project({}, {self.arc_area: [self.symbol_area]})
        project(b, self._sym_stim, self.symbol_area, rounds=rounds)
        b.project({}, {self.arc_area: [self.next_area]})
        project(b, self._st_stim[to_state], self.next_area, rounds=rounds)
        b.areas[self.current_area].unfix_assembly()

    def sample_step(self, seed: int | None = None) -> str:
        """Sample next state via refracted arc + coin; alternate active area."""
        b = self.brain
        active = self._active_area()
        inactive = self._inactive_area()

        self._inhibit_area(inactive)
        self._cue_state(active, self._current)
        b.clear_refracted_bias(self.arc_area)

        coin_choice = self._coin.flip(
            bias=0.5, rounds=10, seed=seed, mode=self.flip_mode,
        )
        coin_asm = self._coin.asm0 if coin_choice == 0 else self._coin.asm1
        winners = coin_asm.winners.copy()

        b.areas[self.arc_area]._winners = winners.astype(np.uint32)
        b._engine.set_winners(self.arc_area, winners.astype(np.uint32))
        b.project({}, {active: [self.arc_area]})
        b.project({}, {self.arc_area: [self.symbol_area]})
        b.project({}, {self.arc_area: [inactive]})

        targets = self._trans.get(self._current, [(self._current, 1.0)])
        if len(targets) == 1:
            nxt = targets[0][0]
        elif len(targets) == 2:
            nxt = targets[0][0] if coin_choice == 0 else targets[1][0]
        else:
            nxt = targets[coin_choice % len(targets)][0]

        project(b, self._st_stim[nxt], inactive, rounds=8)
        b.areas[active].unfix_assembly()
        self._current = nxt
        self._active_is_current = not self._active_is_current
        return nxt


class AlternatingMarkovNetwork:
    """Markov sampler with alternating current/next state inhibition."""

    def __init__(
        self,
        brain,
        states: List[str],
        transitions: List[Tuple[str, str, str, float]],
        initial_state: str,
        *,
        symbol: str = "flip",
        n: int = 5000,
        k: int = 80,
        beta: float = 0.1,
        flip_mode: str = "compete",
        prefix: str = "_markov",
    ):
        self._inner = NemoMarkovPFA(
            brain, states, transitions, initial_state,
            symbol=symbol, n=n, k=k, beta=beta,
            flip_mode=flip_mode, prefix=prefix,
        )

    def sample_step(self, seed: int | None = None) -> str:
        return self._inner.sample_step(seed=seed)

    @property
    def current_state(self) -> str:
        return self._inner.current_state
