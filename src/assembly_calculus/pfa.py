"""
RandomChoiceArea and PFANetwork: probabilistic computation via assemblies.

RandomChoiceArea implements a neural coin-flip: two trained attractor
assemblies compete after mixed initialization, producing a stochastic
binary output.

PFANetwork extends FSMNetwork with probabilistic transitions.  When
multiple transitions exist for the same (state, symbol), uses
RandomChoiceArea to select which target state fires.

Reference:
    Dabagia, M., Papadimitriou, C. H., & Vempala, S. S. (2023).
    "Computation with Sequences of Assemblies in a Model of the Brain."
    arXiv:2306.03812.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from .assembly import Assembly, overlap
from .ops import project, _snap, _fix, _unfix
from .fsm import FSMNetwork


class RandomChoiceArea:
    """Neural coin-flip: two attractor assemblies compete stochastically.

    Creates a brain area with two trained assemblies (attractors).
    ``flip()`` seeds the area with a mixed activation, self-projects,
    and reads which attractor won.

    Args:
        brain: Brain instance.
        area_name: Name for the coin area (default "_coin").
        n: Neurons in the area (default 10000).
        k: Assembly size (default 100).
        beta: Plasticity rate (default 0.05).
        rounds_train: Training rounds per attractor (default 15).
        prefix: Namespace prefix (default "_coin").
    """

    def __init__(
        self,
        brain,
        area_name: str = "_coin",
        n: int = 10000,
        k: int = 100,
        beta: float = 0.05,
        rounds_train: int = 15,
        prefix: str = "_coin",
    ):
        self.brain = brain
        self.area_name = f"{prefix}_{area_name}"
        self.n = n
        self.k = k

        # Create area and two stimuli
        brain.add_area(self.area_name, n, k, beta)

        self._stim0 = f"{prefix}_s0"
        self._stim1 = f"{prefix}_s1"
        brain.add_stimulus(self._stim0, k)
        brain.add_stimulus(self._stim1, k)

        # Train two distinct attractors
        self.asm0 = project(brain, self._stim0, self.area_name,
                            rounds=rounds_train)
        brain._engine.reset_area_connections(self.area_name)
        self.asm1 = project(brain, self._stim1, self.area_name,
                            rounds=rounds_train)
        brain._engine.reset_area_connections(self.area_name)

        # Re-train both to strengthen connections in shared connectome
        for _ in range(3):
            project(brain, self._stim0, self.area_name, rounds=rounds_train)
            project(brain, self._stim1, self.area_name, rounds=rounds_train)

        # Refresh snapshots after shared training
        self.asm0 = project(brain, self._stim0, self.area_name,
                            rounds=rounds_train)
        self.asm1 = project(brain, self._stim1, self.area_name,
                            rounds=rounds_train)

    def flip(self, bias: float = 0.5, rounds: int = 10,
             seed: int = None) -> int:
        """Flip the neural coin.

        Seeds the area with a mixed activation of both attractors
        (proportional to *bias*), then self-projects for *rounds*
        steps.  The attractor that captures the assembly determines
        the output.

        Args:
            bias: Probability of outcome 0 (0.0 to 1.0).
            rounds: Self-projection rounds for attractor competition.
            seed: Optional random seed for the mix.

        Returns:
            0 or 1.
        """
        b = self.brain
        rng = np.random.default_rng(seed)

        # Create mixed activation
        w0 = self.asm0.winners.copy()
        w1 = self.asm1.winners.copy()

        # Sample from each attractor proportional to bias
        n0 = int(self.k * bias)
        n1 = self.k - n0

        if n0 > len(w0):
            n0 = len(w0)
        if n1 > len(w1):
            n1 = len(w1)

        chosen0 = rng.choice(w0, size=n0, replace=False)
        chosen1 = rng.choice(w1, size=n1, replace=False)

        mixed = np.unique(np.concatenate([chosen0, chosen1]))
        # If we have more than k, subsample
        if len(mixed) > self.k:
            mixed = rng.choice(mixed, size=self.k, replace=False)

        # Inject mixed activation
        b.areas[self.area_name]._winners = mixed.astype(np.uint32)
        b._engine.set_winners(self.area_name, mixed.astype(np.uint32))

        # Self-project to let attractors compete
        for _ in range(rounds):
            b.project({}, {self.area_name: [self.area_name]})

        # Read out which attractor won
        result = _snap(b, self.area_name)
        ov0 = overlap(result, self.asm0)
        ov1 = overlap(result, self.asm1)

        return 0 if ov0 >= ov1 else 1


class PFANetwork:
    """Probabilistic finite automaton over neural assemblies.

    Extends FSMNetwork with probabilistic transitions.  When multiple
    transitions exist for the same (state, symbol), uses
    RandomChoiceArea to select which target state fires.

    Args:
        brain: Brain instance.
        states: List of state names.
        symbols: List of input symbol names.
        transitions: List of (from_state, symbol, to_state, probability)
            tuples.  For each (from_state, symbol), probabilities should
            sum to 1.0.  Deterministic transitions use probability=1.0.
        initial_state: Starting state name.
        n, k, beta, rounds: Passed to internal FSMNetwork.
        prefix: Namespace prefix (default "_pfa").
    """

    def __init__(
        self,
        brain,
        states: List[str],
        symbols: List[str],
        transitions: List[Tuple[str, str, str, float]],
        initial_state: str,
        n: int = 10000,
        k: int = 100,
        beta: float = 0.05,
        rounds: int = 10,
        prefix: str = "_pfa",
    ):
        self.brain = brain
        self.initial_state = initial_state
        self.prefix = prefix

        # Group transitions by (from_state, symbol)
        self._trans_map: Dict[Tuple[str, str], List[Tuple[str, float]]] = \
            defaultdict(list)
        for from_st, sym, to_st, prob in transitions:
            self._trans_map[(from_st, sym)].append((to_st, prob))

        # Split into deterministic and probabilistic
        det_transitions = []
        self._prob_keys: List[Tuple[str, str]] = []

        for (from_st, sym), targets in self._trans_map.items():
            if len(targets) == 1:
                det_transitions.append((from_st, sym, targets[0][0]))
            else:
                self._prob_keys.append((from_st, sym))
                # Probabilistic transitions are handled by the coin flip,
                # NOT by the FSM.  Only add deterministic transitions.

        # Build the underlying FSM with deterministic transitions only
        self._fsm = FSMNetwork(
            brain, states, symbols, det_transitions, initial_state,
            n=n, k=k, beta=beta, rounds=rounds, prefix=f"{prefix}_fsm",
        )

        # Build coin flip area for probabilistic selections
        self._coin = RandomChoiceArea(
            brain, area_name="flip", n=n, k=k, beta=beta,
            prefix=f"{prefix}_coin",
        )

        self._current_state = initial_state

    def reset(self):
        """Reset to initial state."""
        self._fsm.reset()
        self._current_state = self.initial_state

    @property
    def current_state(self) -> str:
        return self._current_state

    def step(self, symbol: str, seed: int = None) -> str:
        """Process one symbol and return the new state.

        For deterministic transitions, delegates to the FSM.
        For probabilistic transitions, uses the coin flip to select.

        Args:
            symbol: Input symbol name.
            seed: Optional random seed for probabilistic choice.

        Returns:
            New state name.
        """
        key = (self._current_state, symbol)
        targets = self._trans_map.get(key, [])

        if len(targets) <= 1:
            # Deterministic: delegate to FSM
            # Ensure FSM state matches our state
            self._fsm._current_state = self._current_state
            project(self.brain,
                    self._fsm._st_stim[self._current_state],
                    self._fsm.state_area,
                    rounds=self._fsm.rounds)
            new_state = self._fsm.step(symbol)
        elif len(targets) == 2:
            # Binary probabilistic: use coin flip
            to_st_0, prob_0 = targets[0]
            to_st_1, prob_1 = targets[1]
            result = self._coin.flip(bias=prob_0, rounds=10, seed=seed)
            new_state = to_st_0 if result == 0 else to_st_1
        else:
            # Multi-way: cascade of binary choices
            rng = np.random.default_rng(seed)
            remaining = list(targets)
            new_state = remaining[-1][0]  # default fallback
            cum_prob = 0.0
            for to_st, prob in remaining[:-1]:
                # Coin bias: prob / (1 - cum_prob)
                remaining_prob = 1.0 - cum_prob
                if remaining_prob <= 0:
                    break
                coin_bias = min(prob / remaining_prob, 1.0)
                result = self._coin.flip(
                    bias=coin_bias, rounds=10,
                    seed=int(rng.integers(0, 2**31)),
                )
                if result == 0:
                    new_state = to_st
                    break
                cum_prob += prob
            else:
                new_state = remaining[-1][0]

        self._current_state = new_state
        return new_state

    def run(self, input_symbols: List[str],
            seed: int = None) -> List[str]:
        """Process a sequence of symbols.

        Args:
            input_symbols: Input symbol sequence.
            seed: Optional random seed.

        Returns:
            State trajectory.
        """
        rng = np.random.default_rng(seed)
        trajectory = []
        for sym in input_symbols:
            s = int(rng.integers(0, 2**31))
            new_state = self.step(sym, seed=s)
            trajectory.append(new_state)
        return trajectory
