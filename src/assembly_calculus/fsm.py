"""
FSMNetwork: finite state machine simulation via neural assemblies.

Composes two Brain areas (symbol, state) with a transition table to
implement a deterministic finite automaton.  States and symbols are
encoded as neural assemblies; the transition logic uses the explicit
transition table for reliability while maintaining neural state
representations for composability with other assembly-based components.

Architecture:
    - **Symbol area**: One trained assembly per input symbol.
    - **State area**: One trained assembly per FSM state.
    - **Transition table**: Explicit (from_state, symbol) → to_state mapping.

Neural computation flow:
    Training: Project each state and symbol stimulus to form stable
      assemblies.  Build lexicons for readout.

    Step (runtime):
      1. Activate input symbol via stimulus projection.
      2. Look up transition: (current_state, symbol) → target_state.
      3. Project target state from stimulus for clean neural representation.

The neural value is in the assembly encoding: states and symbols are
represented as robust, noise-tolerant assemblies that compose with
other neural components (readout, merge, pattern completion, etc.).

Reference:
    Dabagia, M., Papadimitriou, C. H., & Vempala, S. S. (2023).
    "Computation with Sequences of Assemblies in a Model of the Brain."
    arXiv:2306.03812.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .assembly import Assembly, overlap
from .readout import fuzzy_readout, build_lexicon, Lexicon
from .ops import project, _snap, _fix, _unfix


class FSMNetwork:
    """Deterministic finite state machine over neural assemblies.

    States and symbols are encoded as neural assemblies with full
    assembly calculus support (pattern completion, readout, merge).
    Transitions use an explicit table — the neural computation provides
    robust encoding, not transition selection.

    Args:
        brain: Brain instance (will be mutated — areas and stimuli added).
        states: List of state names (strings).
        symbols: List of input symbol names (strings).
        transitions: List of (from_state, symbol, to_state) tuples.
        initial_state: Name of the starting state.
        n: Neurons per area (default 10000).
        k: Assembly size (default 100).
        beta: Plasticity rate (default 0.05).
        rounds: Projection rounds for training (default 10).
        prefix: Namespace prefix for area/stimulus names (default "_fsm").
    """

    def __init__(
        self,
        brain,
        states: List[str],
        symbols: List[str],
        transitions: List[tuple],
        initial_state: str,
        n: int = 10000,
        k: int = 100,
        beta: float = 0.05,
        rounds: int = 10,
        prefix: str = "_fsm",
    ):
        self.brain = brain
        self.states = list(states)
        self.symbols = list(symbols)
        self.transitions = list(transitions)
        self.initial_state = initial_state
        self.n = n
        self.k = k
        self.beta = beta
        self.rounds = rounds
        self.prefix = prefix

        # Area names
        self.symbol_area = f"{prefix}_symbol"
        self.state_area = f"{prefix}_state"

        # Transition table: (from_state, symbol) → to_state
        self._transition_table: Dict[Tuple[str, str], str] = {}
        for from_st, sym, to_st in transitions:
            self._transition_table[(from_st, sym)] = to_st

        # Build infrastructure, train, and reset
        self._setup_areas()
        self._train()
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_areas(self):
        """Create brain areas and stimuli."""
        b = self.brain

        b.add_area(self.symbol_area, self.n, self.k, self.beta)
        b.add_area(self.state_area, self.n, self.k, self.beta)

        self._sym_stim = {}
        for sym in self.symbols:
            stim_name = f"{self.prefix}_sym_{sym}"
            b.add_stimulus(stim_name, self.k)
            self._sym_stim[sym] = stim_name

        self._st_stim = {}
        for st in self.states:
            stim_name = f"{self.prefix}_st_{st}"
            b.add_stimulus(stim_name, self.k)
            self._st_stim[st] = stim_name

    def _train(self):
        """Train symbol and state assemblies."""
        b = self.brain
        r = self.rounds

        # Train stable symbol assemblies
        self.symbol_lexicon: Lexicon = {}
        for sym in self.symbols:
            stim = self._sym_stim[sym]
            asm = project(b, stim, self.symbol_area, rounds=r)
            self.symbol_lexicon[sym] = asm
            b._engine.reset_area_connections(self.symbol_area)

        # Train stable state assemblies
        self.state_lexicon: Lexicon = {}
        for st in self.states:
            stim = self._st_stim[st]
            asm = project(b, stim, self.state_area, rounds=r)
            self.state_lexicon[st] = asm
            b._engine.reset_area_connections(self.state_area)

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to the initial state."""
        b = self.brain
        project(b, self._st_stim[self.initial_state],
                self.state_area, rounds=self.rounds)
        self._current_state = self.initial_state

    @property
    def current_state(self) -> str:
        """Return the name of the current FSM state."""
        return self._current_state

    def step(self, symbol: str) -> str:
        """Process one input symbol and return the new state name.

        Activates the symbol in the symbol area, looks up the transition,
        and projects the target state into the state area.

        Args:
            symbol: Input symbol name.

        Returns:
            Name of the new state after the transition.

        Raises:
            KeyError: If no transition exists for (current_state, symbol).
        """
        b = self.brain

        # 1. Activate symbol via stimulus (maintains neural representation)
        project(b, self._sym_stim[symbol], self.symbol_area,
                rounds=self.rounds)

        # 2. Look up transition
        new_state = self._transition_table[(self._current_state, symbol)]
        self._current_state = new_state

        # 3. Project target state from stimulus for clean representation
        project(b, self._st_stim[new_state], self.state_area,
                rounds=self.rounds)

        return new_state

    def run(self, input_symbols: List[str]) -> List[str]:
        """Process a sequence of symbols and return the state trajectory.

        Args:
            input_symbols: Sequence of input symbol names.

        Returns:
            List of state names after each symbol (length = len(input_symbols)).
        """
        trajectory = []
        for sym in input_symbols:
            new_state = self.step(sym)
            trajectory.append(new_state)
        return trajectory
