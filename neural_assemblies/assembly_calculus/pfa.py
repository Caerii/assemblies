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
differently can give a different outcome. The seed mixture controls the initial
condition; it is not a calibrated probability for the returned label.

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
import math
from numbers import Integral, Real

from ..core.registration import validate_round_count
from ..core.index_spaces import validated_indices

import numpy as np

from .assembly import Assembly, overlap
from .ops import activate_assembly, project, _snap
from .fsm import FSMNetwork
from .transitions import TransitionLike, TransitionMap

FlipMode = Literal["k_split", "compete"]
Construction = Literal["legacy", "attractor"]


def _seed_winners(brain, area_name: str, neuron_ids,
                  remap: bool = True) -> np.ndarray:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-coin-seed

    Install stable IDs through the shared activation boundary. Missing IDs and
    malformed inputs fail before activity changes; no membership is dropped.
    The old untranslatable legacy path is retained only as an explicit error.
    """
    if not remap:
        raise ValueError("Legacy coin seeding confuses neuron IDs with compact indices; "
                         "use construction='attractor'. Historical goldens are not valid coins.")
    activate_assembly(brain, Assembly(area_name, neuron_ids))
    return brain.areas[area_name].winners.copy()



class RandomChoiceArea:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-coin-operation

    Neural coin-flip: two attractor assemblies compete stochastically.

    Creates a brain area with two trained assemblies (attractors). ``flip()``
    seeds the area, lets recurrence settle, and reads which attractor won.

    CHOOSE THE CONSTRUCTION DELIBERATELY -- they are not two tunings of one
    thing, and only one of them builds a coin.

    ``construction="attractor"`` is the working one. Measured at ``n=2000,
    k=200, beta=3.0`` over 4 brains x 60 flips, the settled state overlaps the
    winning attractor **0.985** against a chance floor of ``k/n = 0.100``.

    ``construction="legacy"`` (the default, for now) scores **0.159** on that
    same measurement -- barely off the floor. Its recurrent fiber is never
    allocated, so the settle loop delivers zero drive and the returned 0/1
    came from the seed RNG. Its construction remains available for inspection,
    but flip() now rejects it before changing activity. The old default therefore
    requires callers to select construction="attractor" explicitly. Historical
    ``coin2024_*`` goldens describe the invalid instrument; see ``_build_legacy``.

    THE TWO ARE INDISTINGUISHABLE ON FAIRNESS. Both read ~0.5 heads with
    comparable across-brain spread. Anything that validates this class must
    assert on overlap-with-the-winner against ``k/n``, never on the head rate;
    ``tests/test_coin_construction.py`` pins that distinction, and
    ``research/notes/coin/neural_coin_fairness.md`` has the full analysis.

    Args:
        brain: Brain instance.
        area_name: Name for the coin area (default "_coin").
        n: Neurons in the area (default 10000).
        k: Assembly size (default 100).
        beta: Plasticity rate (default 0.05). The attractor construction wants
            this HIGH (~3.0); an assembly must survive its own recurrence.
        rounds_train: Training rounds per attractor (default 15).
        prefix: Namespace prefix (default "_coin").
        construction: "legacy" (default, inspectable but flips rejected) or
            "attractor" (validated). See above.
        fires: attractor construction only -- how many times each assembly is
            force-fired into the shared connectome (default 2). Symmetric by
            construction; raising it deepens both basins equally.
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
        construction: Construction = "legacy",
        fires: int = 2,
    ):
        if construction not in ("legacy", "attractor"):
            raise ValueError("construction must be 'legacy' or 'attractor'")
        rounds_train = validate_round_count(rounds_train)
        if isinstance(fires, bool) or not isinstance(fires, Integral) or fires < 0:
            raise ValueError("fires must be a nonnegative integer")
        self.brain = brain
        self.area_name = f"{prefix}_{area_name}"
        self.n = n
        self.k = k
        self.construction = construction

        # Create area and two stimuli
        brain.add_area(self.area_name, n, k, beta)

        self._stim0 = f"{prefix}_s0"
        self._stim1 = f"{prefix}_s1"
        brain.add_stimulus(self._stim0, k)
        brain.add_stimulus(self._stim1, k)

        # Stage 1 (BOTH constructions) -- carve out two INDEPENDENT attractors.
        # Each stimulus is projected with the recurrent weights reset before and
        # after, so neither assembly is formed in the shadow of the other's
        # basin. Without the resets the second stimulus is pulled into the
        # first's basin and the two "attractors" largely overlap, which makes
        # the coin degenerate (always the same answer).
        self.asm0 = project(brain, self._stim0, self.area_name,
                            rounds=rounds_train)
        brain._engine_for(brain.areas[self.area_name]).reset_area_connections(self.area_name)
        self.asm1 = project(brain, self._stim1, self.area_name,
                            rounds=rounds_train)
        brain._engine_for(brain.areas[self.area_name]).reset_area_connections(self.area_name)

        if construction == "attractor":
            self._build_attractor(fires)
        else:
            self._build_legacy(rounds_train)

    def _build_legacy(self, rounds_train: int) -> None:
        """The shipped construction. IT DOES NOT BUILD A COIN -- see `flip`.

        Retained for inspection of the historical ``coin2024_*`` instrument.
        Its flips are rejected; historical numerical reproduction is not a
        supported path through the current strict identity boundaries.
        Two independent defects, both measured 2026-07-30 at ``n=2000, k=50``:

        1. The loop below is meant to deepen both basins in ONE shared
           connectome, alternating so neither gets a head start. It does not:
           ``ops.project``'s ``recurrent`` argument defaults False, so every
           call is stimulus-only and NOT ONE RECURRENT WEIGHT IS WRITTEN. The
           area ends with no ``C -> C`` block at all -- shape (0, 0), nnz 0,
           while the area has ``w=357``. There are no fixed points, so a mixed
           initial state has nothing to fall into.
        2. The re-snapshot is ASYMMETRIC. asm1 is reinforced last, leaving it
           systematically the deeper basin on every seed measured
           (within-asm1 / within-asm0 = 5.07/4.59, 5.10/4.30, 4.82/4.44).

        Passing ``recurrent=True`` here writes the block (18,006 synapses) but
        does not produce a working coin either, because the seed still spans
        ``w`` rather than ``n``. Use ``construction="attractor"``.
        """
        brain = self.brain
        for _ in range(3):
            project(brain, self._stim0, self.area_name, rounds=rounds_train)
            project(brain, self._stim1, self.area_name, rounds=rounds_train)

        # Stage 1's snapshots were taken against connectomes that no longer
        # exist, so re-take them. Everything downstream compares flip results
        # against asm0/asm1 by overlap; stale references would mis-score every
        # flip.
        self.asm0 = project(brain, self._stim0, self.area_name,
                            rounds=rounds_train)
        self.asm1 = project(brain, self._stim1, self.area_name,
                            rounds=rounds_train)

    def _build_attractor(self, fires: int) -> None:
        """The construction validated in ``research/notes/coin/neural_coin_fairness.md``.

        Three things the legacy path gets wrong, each of which was individually
        load-bearing:

        **Materialise the area first.** The reference allocates a dense
        ``n x n`` recurrent matrix up front and seeds a uniform random
        ``k``-subset of ALL ``n``. Under lazy materialisation the fiber spans
        only the ``w`` neurons that have won something, so at ``n=2000`` with
        ``w=357`` a ``k=50`` seed carried 9.2 +/- 2.6 real neurons -- 82% of it
        addressed no synapse.

        **Fire each assembly the same number of times, then stop.** Symmetry is
        the whole claim; a last-reinforced attractor is a bent coin by
        construction.

        **FORCE the activations.** The reference's ``fire(assm)`` pins the
        winners so plasticity writes ``assembly x assembly``. A plain
        ``project`` recomputes winners from a still-untrained block, gets
        noise, and potentiates ``assembly x noise`` instead. ``fix_assembly``
        is our equivalent.

        Measured over 40 brains x 400 flips at ``n=2000``: overlap with the
        winning attractor 0.985 against a chance floor of ``k/n = 0.1``, and
        the across-brain spread falls as ``k^-0.77`` while basin asymmetry
        falls as ``k^-1.01`` (log-log r = -0.997) out to ``n=16,000``.
        """
        brain = self.brain
        area = brain.areas[self.area_name]
        engine = brain._engine_for(area)

        materialize = getattr(engine, "materialize_area", None)
        if materialize is None:
            raise NotImplementedError(
                f"construction='attractor' needs an engine that can "
                f"materialise an area up front; {type(engine).__name__} "
                f"cannot. Use engine='numpy_sparse'.")
        materialize(self.area_name)

        # Resolve complete snapshots through the same boundary used for flips.
        c0 = _seed_winners(brain, self.area_name, self.asm0.winners)
        c1 = _seed_winners(brain, self.area_name, self.asm1.winners)
        self._compact0, self._compact1 = c0, c1

        for _ in range(fires):
            for compact in (c0, c1):
                area.winners = compact.copy()
                engine.set_winners(self.area_name, compact)
                was_fixed = area.fixed_assembly
                try:
                    area.fix_assembly()
                    brain.project({}, {self.area_name: [self.area_name]})
                finally:
                    area.fixed_assembly = was_fixed

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

        Legacy construction is rejected before seeding. The following numbers
        describe the historical invalid instrument, not executable compatibility.

        Under the historical ``construction="legacy"`` the loop was inert: the ``area -> area``
        block is never allocated (self fibers are excluded from deferred init
        -- see ``_self_fiber_deferred_init`` in the numpy engine), so it
        delivers zero drive and ``project_into`` hands back the incumbent
        winners::

            rounds = 0 / 1 / 10   ->  200/200 identical flips, same head count
            winners moved         ->  0 of 10 rounds
            cross-fiber control   ->  moves winners 2/5 rounds

        The answer is decided entirely by the ``rng.choice`` above, and the
        settled state's overlap with the winning attractor reads 0.159 against
        a 0.100 chance floor.

        Under ``construction="attractor"`` the same loop is the whole
        mechanism: 0.985 on that measurement. The ladder in
        ``research/notes/coin/neural_coin_fairness.md`` -- same construction, run
        standalone -- takes it to 1.000 by ``n=8000``, and shows decisiveness
        climbing monotonically over ``rounds`` (0.571 at 1, 0.952 at 10, 0.985
        at 20) then flat at 40: a fixed point, not a slow drift.

        NOTE THAT ``bias`` IS OUR INVENTION and does not survive live settling.
        The reference (``.reference/mdabagia-nemo/brain.py``) seeds a UNIFORM
        RANDOM k-subset of all n and has no bias parameter; ``compete``
        reproduces that only at bias=0.5. With the fiber dead, ``k_split``
        tracked bias exactly (0.000 / 0.000 / 0.710 / 1.000 / 1.000 across
        bias 0 -> 1) for the trivial reason that it was reading back its own
        seed mix. Once the dynamics run they overwrite that mix with whichever
        attractor won, so a biased seed does not yield a proportionally biased
        answer -- at beta=0.1/T=20 the relationship INVERTS (0.614 -> 0.200 as
        bias rises). If you want a biased coin, bias the BASINS (asymmetric
        ``fires``), not the seed.

        Both modes disable plasticity while settling. A flip changes activity
        but does not train the stored attractors. Prefer ``compete`` when measuring a distribution --
        it is the reference protocol. ``k_split`` remains the package default
        because published numbers in this repo were produced with it.
        """
        if mode not in ("k_split", "compete"):
            raise ValueError("mode must be 'k_split' or 'compete'")
        if (isinstance(bias, bool) or not isinstance(bias, Real)
                or not math.isfinite(bias) or not 0 <= bias <= 1):
            raise ValueError("bias must be a finite real number in [0, 1]")
        if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if self.construction == "legacy":
            raise ValueError("Legacy coin construction does not form recurrent attractors; "
                             "use construction='attractor' and validate its training regime.")
        if mode == "compete":
            return self._flip_compete(bias=bias, rounds=rounds, seed=seed)
        return self._flip_k_split(bias=bias, rounds=rounds, seed=seed)

    def _mixed_seed(self, rng, bias: float) -> np.ndarray:
        n0 = int(self.k * bias)
        chosen = [rng.choice(asm.winners, size=min(count, len(asm.winners)), replace=False)
                  for asm, count in ((self.asm0, n0), (self.asm1, self.k - n0))]
        return np.unique(np.concatenate(chosen))

    def _seed_uniform(self, rng) -> None:
        area = self.brain.areas[self.area_name]
        engine = self.brain._engine_for(area)
        count = engine.materialized_count(self.area_name)
        if count is not None and count != area.n:
            raise ValueError("uniform coin seeding requires the complete materialized population")
        compact = validated_indices(rng.choice(area.n, size=self.k, replace=False),
                                    upper=area.n, label="uniform coin seed", unique=True)
        area.winners = compact
        engine.set_winners(self.area_name, compact)

    def _flip_k_split(self, bias: float, rounds: int, seed: int | None) -> int:
        rng = np.random.default_rng(seed)
        _seed_winners(self.brain, self.area_name, self._mixed_seed(rng, bias))
        return self._settle_and_read(rounds)

    def _flip_compete(self, bias: float, rounds: int, seed: int | None) -> int:
        """Uniform compact seed at neutral bias; mixed stable IDs otherwise."""
        rng = np.random.default_rng(seed)
        area = self.brain.areas[self.area_name]
        if abs(bias - 0.5) < 1e-9 and getattr(area, "input_noise_std", 0.0) <= 0:
            self._seed_uniform(rng)
        else:
            mixed = self._mixed_seed(rng, bias)
            if len(mixed):
                _seed_winners(self.brain, self.area_name, mixed)
            else:
                self._seed_uniform(rng)
        return self._settle_and_read(rounds)

    def _settle_and_read(self, rounds: int) -> int:
        """Run the recurrent settle with plasticity OFF and score the result.

        Plasticity must stay off: a flip is a MEASUREMENT of the stored
        attractors, not training. Leaving it on potentiates whichever basin
        this flip landed in, so successive flips on one coin are not
        independent draws and a measured distribution drifts toward its own
        history.
        """
        b = self.brain
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
