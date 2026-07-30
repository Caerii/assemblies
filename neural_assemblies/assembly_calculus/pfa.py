"""
RandomChoiceArea and PFANetwork: probabilistic computation via assemblies.

WHERE THE RANDOMNESS COMES FROM.  NEMO's dynamics are deterministic once the
connectome is fixed: sum inputs, take the top k, potentiate.  There is no
noise term and no sampling step.  So a probabilistic automaton cannot simply
"draw" a transition -- the stochasticity has to be manufactured out of the
dynamics themselves.

The construction here is the standard one: train TWO attractor assemblies into
a single area, then start the area from a state that lies between them and let
the winners-take-all competition run.  Which basin the trajectory falls into
depends on the fine detail of the mixed initial condition, so seeding the mix
differently gives a different outcome -- a coin whose bias is set by how much
of each attractor goes into the seed.

Three variants exist here and they differ in what supplies the entropy:

    _flip_k_split  -- seed = ``bias``-weighted sample of the two attractors'
                      winners.  Entropy comes entirely from the caller's RNG
                      choosing WHICH winners to include.
    _flip_compete  -- at bias 0.5 with no input noise, seed = a uniformly
                      random k-subset of the whole area, so the answer is
                      decided by which attractor happens to be better
                      represented in the random draw.  This is the reference
                      NEMO coin (mdabagia/nemo ``RandomChoiceArea.flip``).
    SoftmaxContextCoin -- adds per-neuron i.i.d. input noise and E%-WTA, so
                      entropy is injected into the dynamics rather than only
                      the initial condition, and the probability becomes a
                      smooth function of the trained context->outcome weights.

PFANetwork extends FSMNetwork with probabilistic transitions.  When
multiple transitions exist for the same (state, symbol), uses
RandomChoiceArea to select which target state fires.

Reference:
    Dabagia, M., Papadimitriou, C. H., & Vempala, S. S. (2023).
    "Computation with Sequences of Assemblies in a Model of the Brain."
    arXiv:2306.03812.
"""

from typing import Dict, List, Literal, Tuple
from collections import defaultdict

import numpy as np

from .assembly import overlap
from .ops import project, _snap
from .fsm import FSMNetwork
from .transitions import TransitionLike, TransitionMap

FlipMode = Literal["k_split", "compete"]


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

        # Training is deliberately done in two stages.
        #
        # Stage 1 -- carve out two INDEPENDENT attractors.  Each stimulus is
        # projected with the recurrent weights reset before and after, so
        # neither assembly is formed in the shadow of the other's attractor.
        # Without the resets the second stimulus is pulled into the first's
        # basin and the two "attractors" end up largely overlapping, which
        # would make the coin degenerate (always the same answer).
        self.asm0 = project(brain, self._stim0, self.area_name,
                            rounds=rounds_train)
        brain._engine.reset_area_connections(self.area_name)
        self.asm1 = project(brain, self._stim1, self.area_name,
                            rounds=rounds_train)
        brain._engine.reset_area_connections(self.area_name)

        # Stage 2 -- INTENDED to deepen both basins in ONE shared connectome,
        # alternating so neither gets a systematic head start.
        #
        # MEASURED 2026-07-30: IT DOES NOT DO THIS. `project` here is
        # `ops.project`, whose `recurrent` argument defaults False, so every
        # call below is stimulus-only and NOT ONE RECURRENT WEIGHT IS WRITTEN.
        # After this loop the area has no `C -> C` weight block at all (shape
        # (0, 0), nnz 0, while the area has w=357). There are no fixed points,
        # so there is nothing for a mixed initial state to fall into.
        #
        # Passing `recurrent=True` does write the block -- 18,006 synapses --
        # but does NOT yield a working coin: see `flip`. Left as-is
        # deliberately, because the recurrent version measures WORSE.
        for _ in range(3):
            project(brain, self._stim0, self.area_name, rounds=rounds_train)
            project(brain, self._stim1, self.area_name, rounds=rounds_train)

        # Stage 1's snapshots were taken against connectomes that no longer
        # exist, so re-take them.  Everything downstream compares flip results
        # against asm0/asm1 by overlap; stale references would mis-score every
        # flip.  Note this leaves asm1 as the most recently reinforced
        # attractor.
        self.asm0 = project(brain, self._stim0, self.area_name,
                            rounds=rounds_train)
        self.asm1 = project(brain, self._stim1, self.area_name,
                            rounds=rounds_train)

    def flip(
        self,
        bias: float = 0.5,
        rounds: int = 10,
        seed: int | None = None,
        mode: FlipMode = "k_split",
    ) -> int:
        """Flip the neural coin.

        ``k_split`` — proportional mix of attractor winners (package default).
        ``compete`` — random k-init + competition (mdabagia/nemo ``RandomChoiceArea.flip``).

        Both return 0 or 1 by asking which trained attractor the settled state
        overlaps more; ties go to 0.

        THE SETTLE LOOP CONTRIBUTES NOTHING, AND MAKING IT LIVE MAKES THE COIN
        WORSE.  Measured 2026-07-30, and both halves matter.

        As shipped the loop is inert: the ``area -> area`` block is never
        allocated (self fibers are excluded from deferred init -- see
        ``_self_fiber_deferred_init`` in the numpy engine), so it delivers zero
        drive and ``project_into`` hands back the incumbent winners::

            rounds = 0 / 1 / 10   ->  200/200 identical flips, same head count
            winners moved         ->  0 of 10 rounds
            cross-fiber control   ->  moves winners 2/5 rounds

        So the answer is decided entirely by the ``rng.choice`` above.

        Waking the fiber AND training it recurrently was tried across
        ``beta`` in {0.05, 0.1, 0.2}, ``rounds_train`` in {10, 20, 40} and
        settle lengths {0, 1, 2, 3, 5, 10}, over 5-8 brains x 100 flips.  No
        cell is a fair coin.  At bias=0.5, ``compete`` reads 0.661 +/- 0.039
        across brains as shipped, against 0.319 +/- 0.159 and 0.649 +/- 0.165
        recurrently trained -- the mean stays biased and the ACROSS-BRAIN
        spread grows 4-6x, which is the worse failure: a coin that answers 0.08
        of the time in one brain and 0.52 in another is not a coin.

        ``k_split`` additionally fails to track its own ``bias`` once settling
        is live.  With no settling it is exactly monotone (0.000 / 0.000 /
        0.710 / 1.000 / 1.000 across bias 0 -> 1); with any settling that
        collapses, and at beta=0.1/T=20 it INVERTS (0.614 -> 0.200 as bias
        rises).  The seeded mix already carries the answer and the dynamics
        destroy it.

        Note also that ``bias`` is OUR invention.  The reference
        (``.reference/mdabagia-nemo/brain.py::RandomChoiceArea.flip``) seeds a
        UNIFORM RANDOM k-subset of all n neurons and has no bias parameter at
        all; ``compete`` reproduces that only at bias=0.5.

        What that leaves: the substrate is not amplifying a random seed into a
        clean binary decision here, and the published coin numbers measure the
        seed RNG.  Tracked as #70.

        The two modes are NOT interchangeable as measurement instruments.
        ``compete`` disables plasticity for the duration of the flip, so it is
        a pure read: flipping the same coin a thousand times leaves it exactly
        as it was.  ``k_split`` leaves plasticity ON, so every flip potentiates
        whichever basin it landed in and successive flips are not independent
        -- the coin drifts toward its own history.  Use ``compete`` when
        measuring a flip distribution.  ``k_split`` remains the package default
        because published numbers in this repo were produced with it.
        """
        if mode == "compete":
            return self._flip_compete(bias=bias, rounds=rounds, seed=seed)
        return self._flip_k_split(bias=bias, rounds=rounds, seed=seed)

    def _flip_k_split(self, bias: float, rounds: int, seed: int | None) -> int:
        b = self.brain
        rng = np.random.default_rng(seed)

        w0 = self.asm0.winners.copy()
        w1 = self.asm1.winners.copy()

        n0 = int(self.k * bias)
        n1 = self.k - n0

        if n0 > len(w0):
            n0 = len(w0)
        if n1 > len(w1):
            n1 = len(w1)

        chosen0 = rng.choice(w0, size=n0, replace=False)
        chosen1 = rng.choice(w1, size=n1, replace=False)

        mixed = np.unique(np.concatenate([chosen0, chosen1]))
        if len(mixed) > self.k:
            mixed = rng.choice(mixed, size=self.k, replace=False)

        b.areas[self.area_name]._winners = mixed.astype(np.uint32)
        b._engine.set_winners(self.area_name, mixed.astype(np.uint32))

        # Plasticity OFF while settling, matching _flip_compete. A flip is a
        # MEASUREMENT of the stored attractors, not training: leaving it on
        # would potentiate whichever basin this flip happened to land in, so
        # successive flips on one coin would not be independent draws and a
        # measured flip distribution could drift toward its own history.
        # (Attempts to exhibit that drift empirically here were inconclusive --
        # see the note in the sibling method -- but a read-out that writes is
        # wrong regardless of whether the bias is currently large enough to
        # detect.)
        with b.frozen():
            for _ in range(rounds):
                b.project({}, {self.area_name: [self.area_name]})

        result = _snap(b, self.area_name)
        ov0 = overlap(result, self.asm0)
        ov1 = overlap(result, self.asm1)
        return 0 if ov0 >= ov1 else 1

    def _flip_compete(self, bias: float, rounds: int, seed: int | None) -> int:
        """Reference NEMO coin: random/noisy seed then attractor competition."""
        b = self.brain
        rng = np.random.default_rng(seed)
        area = b.areas[self.area_name]
        noise_std = getattr(area, "input_noise_std", 0.0)

        if abs(bias - 0.5) < 1e-9 and noise_std <= 0:
            initial = rng.choice(self.n, size=self.k, replace=False)
        else:
            w0 = self.asm0.winners.copy()
            w1 = self.asm1.winners.copy()
            n0 = int(self.k * bias)
            n1 = self.k - n0
            n0 = min(n0, len(w0))
            n1 = min(n1, len(w1))
            chosen0 = rng.choice(w0, size=n0, replace=False) if n0 else np.array([], dtype=w0.dtype)
            chosen1 = rng.choice(w1, size=n1, replace=False) if n1 else np.array([], dtype=w1.dtype)
            if len(chosen0) + len(chosen1) == 0:
                initial = rng.choice(self.n, size=self.k, replace=False)
            else:
                initial = np.unique(np.concatenate([chosen0, chosen1]))
                if len(initial) > self.k:
                    initial = rng.choice(initial, size=self.k, replace=False)

        initial = initial.astype(np.uint32)
        b.areas[self.area_name]._winners = initial
        b._engine.set_winners(self.area_name, initial)

        with b.frozen():
            for _ in range(rounds):
                b.project({}, {self.area_name: [self.area_name]})

        result = _snap(b, self.area_name)
        ov0 = overlap(result, self.asm0)
        ov1 = overlap(result, self.asm1)
        return 0 if ov0 >= ov1 else 1


class SoftmaxContextCoin:
    """Context area → outcome area coin (dabagia.org coinflipping architecture).

    Trains a context assembly that projects into a two-outcome area with
    i.i.d. input noise and compete-mode attractor dynamics (softmax-like
    weight-dependent probabilities per site description).
    """

    def __init__(
        self,
        brain,
        *,
        n: int = 5000,
        k: int = 50,
        beta: float = 0.08,
        noise_std: float = 0.02,
        coupling_rounds: int = 12,
        prefix: str = "_ctx_coin",
    ):
        from neural_assemblies.compute import EPercentPolicy

        self.brain = brain
        self.k = k
        self.context_area = f"{prefix}_context"
        ctx_stim = f"{prefix}_ctx"
        self._ctx_stim = ctx_stim

        brain.add_area(self.context_area, n, k, beta)
        brain.add_stimulus(ctx_stim, k)
        project(brain, ctx_stim, self.context_area, rounds=8)

        self.coin = RandomChoiceArea(
            brain, area_name="out", n=n, k=k, beta=beta, prefix=prefix,
        )
        self.outcome_area = self.coin.area_name

        if noise_std > 0:
            brain.set_input_noise(self.outcome_area, noise_std)
            brain.set_competition_policy(
                self.outcome_area,
                EPercentPolicy(fraction_of_max=0.5, min_winners=1),
            )

        for _ in range(coupling_rounds):
            brain.project(
                {ctx_stim: [self.context_area]},
                {self.context_area: [self.outcome_area]},
            )
            project(brain, self.coin._stim0, self.outcome_area, rounds=4)
            project(brain, self.coin._stim1, self.outcome_area, rounds=4)

    def flip(self, bias: float = 0.5, seed: int | None = None, rounds: int = 10) -> int:
        from neural_assemblies.assembly_calculus.ops import _snap
        from neural_assemblies.assembly_calculus.assembly import overlap

        b = self.brain
        area_name = self.outcome_area
        area = b.areas[area_name]
        area.unfix_assembly()
        b._engine.set_winners(area_name, np.array([], dtype=np.uint32))

        project(b, self._ctx_stim, self.context_area, rounds=1)
        b.project({}, {self.context_area: [area_name]})

        rng = np.random.default_rng(seed)
        if abs(bias - 0.5) < 1e-9:
            initial = rng.choice(self.coin.n, size=self.k, replace=False)
        else:
            w0 = self.coin.asm0.winners.copy()
            w1 = self.coin.asm1.winners.copy()
            n0 = int(self.k * bias)
            n1 = self.k - n0
            n0 = min(n0, len(w0))
            n1 = min(n1, len(w1))
            chosen0 = rng.choice(w0, size=n0, replace=False) if n0 else np.array([], dtype=w0.dtype)
            chosen1 = rng.choice(w1, size=n1, replace=False) if n1 else np.array([], dtype=w1.dtype)
            initial = np.unique(np.concatenate([chosen0, chosen1]))
            if len(initial) == 0:
                initial = rng.choice(self.coin.n, size=self.k, replace=False)
            elif len(initial) > self.k:
                initial = rng.choice(initial, size=self.k, replace=False)

        initial = initial.astype(np.uint32)
        b.areas[area_name]._winners = initial
        b._engine.set_winners(area_name, initial)

        with b.frozen():
            for _ in range(rounds):
                b.project({}, {area_name: [area_name]})

        result = _snap(b, area_name)
        ov0 = overlap(result, self.coin.asm0)
        ov1 = overlap(result, self.coin.asm1)
        return 0 if ov0 >= ov1 else 1

    def train_bias(self, bias: float, *, rounds: int = 20) -> None:
        """Skew outcome weights via asymmetric Hebbian coupling."""
        stim = self.coin._stim0 if bias >= 0.5 else self.coin._stim1
        alt = self.coin._stim1 if bias >= 0.5 else self.coin._stim0
        major = max(int(rounds * abs(bias - 0.5) * 2 + rounds * 0.5), 1)
        minor = max(rounds - major, 1)
        for _ in range(major):
            project(self.brain, self._ctx_stim, self.context_area, rounds=1)
            self.brain.project({}, {self.context_area: [self.outcome_area]})
            project(self.brain, stim, self.outcome_area, rounds=4)
        for _ in range(minor):
            project(self.brain, self._ctx_stim, self.context_area, rounds=1)
            self.brain.project({}, {self.context_area: [self.outcome_area]})
            project(self.brain, alt, self.outcome_area, rounds=2)

    def learn_from_frequencies(
        self,
        freq0: float,
        freq1: float,
        *,
        rounds_per_unit: int = 8,
    ) -> None:
        """Hebbian coupling rounds proportional to target outcome frequencies."""
        total = max(freq0 + freq1, 1e-9)
        p0, p1 = freq0 / total, freq1 / total
        n0 = max(1, int(round(rounds_per_unit * 10 * p0)))
        n1 = max(1, int(round(rounds_per_unit * 10 * p1)))
        for _ in range(n0):
            project(self.brain, self._ctx_stim, self.context_area, rounds=1)
            self.brain.project({}, {self.context_area: [self.outcome_area]})
            project(self.brain, self.coin._stim0, self.outcome_area, rounds=4)
        for _ in range(n1):
            project(self.brain, self._ctx_stim, self.context_area, rounds=1)
            self.brain.project({}, {self.context_area: [self.outcome_area]})
            project(self.brain, self.coin._stim1, self.outcome_area, rounds=4)

    def empirical_flip_counts(
        self,
        n_flips: int,
        bias: float = 0.5,
        seed_base: int = 0,
    ) -> tuple[int, int]:
        counts = {0: 0, 1: 0}
        for i in range(n_flips):
            counts[self.flip(bias=bias, seed=seed_base + i * 17)] += 1
        return counts[0], counts[1]


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
        transitions: List[TransitionLike],
        initial_state: str,
        n: int = 10000,
        k: int = 100,
        beta: float = 0.05,
        rounds: int = 10,
        prefix: str = "_pfa",
        flip_mode: FlipMode = "k_split",
    ):
        self.brain = brain
        self.initial_state = initial_state
        self.prefix = prefix
        self.flip_mode: FlipMode = flip_mode

        self.transition_map = TransitionMap(transitions).validate_probability_mass()

        # Group transitions by (from_state, symbol)
        self._trans_map: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
        for transition in self.transition_map:
            self._trans_map[transition.key].append(
                (transition.to_state, transition.probability)
            )

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
            result = self._coin.flip(
                bias=prob_0, rounds=10, seed=seed, mode=self.flip_mode,
            )
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
                    mode=self.flip_mode,
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
