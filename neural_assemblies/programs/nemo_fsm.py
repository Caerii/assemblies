"""
NEMO arc-area FSM and Markov PFA — mdabagia/nemo port.

Reference: dabagia.org/nemo/sequences/, dabagia.org/nemo/coinflipping/
           .reference/mdabagia-nemo/brain.py (FSMNetwork, PFANetwork)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import (
    _snap, activate_assembly, project,
)
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.assembly_calculus.pfa import RandomChoiceArea
from neural_assemblies.programs.arc_core import add_arc_state_core
from neural_assemblies.assembly_calculus.transitions import TransitionLike, TransitionMap


class NemoArcFSM:
    """Symbol + state + refracted arc FSM (reference ``FSMNetwork``).

    THE MACHINE DECIDES. ``step`` reads the next state OUT of the state
    assembly by nearest overlap; it does not consult the transition table.
    The previous version returned ``self._table[(from_state, symbol)]`` -- a
    dictionary lookup -- after running and discarding the dynamics, so every
    FSM claim in this repo passed with an untrained brain, with zero
    presentations, and with beta=0 (measured; see
    `research/experiments/seq_arc_refraction_reference.py`). The table is
    still held, but only training may read it.

    THE ARC IS A CONJUNCTION, AND REFRACTION IS WHAT KEEPS IT ONE. The arc
    receives the symbol and the current state; without an opposing force it
    collapses onto whichever conjunct is exposed more -- to ``A_sigma`` when
    the symbol fires more (task #92) or to ``A_q`` when the state does (the
    reference's mod-3 table, where ablating refraction takes across-symbol
    overlap to 0.989 and the task to 0/3). Two consequences for this class:

    * the accumulated bias MUST persist across transitions. The previous
      version called ``clear_refracted_bias`` at the top of every
      ``train_transition``, which is a no-op mechanism dressed as an active
      one -- the reference clears activations between transitions and never
      clears the bias.
    * the arc's two conjuncts should be exposed comparably often; see
      [[role-gain-crowds-not-margins]].

    RESULTS. Implements [[SEQ-FSM]] on the substrate of
    [[SEQ-TIME-IN-WEIGHTS]]. Its preconditions are [[SEQ-REGIME]] in EVERY
    area -- check with `diagnostics.regime_audit` before trusting a negative --
    and comparable exposure across conjuncts ([[ARC-CONJUNCT-EXPOSURE]]).
    Behaviour over long sequences is [[SEQ-EXACT-RECOVERY]]. The state
    alphabet is assigned, not induced: see [[SEQ-STATE-CODE-EMERGENT]].

    ONE TEACHER-FORCED WRITE. The reference performs
    ``state.set_input(arc.read()); state.fire(new_state)``: a single
    potentiation of arc -> state onto the TARGET assembly. The previous
    version projected arc -> state freely first, potentiating onto whatever
    the arc already preferred, and only then drove the target from its own
    stimulus -- so the transition was never taught along the pathway that has
    to carry it at test time.
    """

    def __init__(
        self,
        brain,
        states: List[str],
        symbols: List[str],
        transitions: List[TransitionLike],
        *,
        n: int = 5000,
        k: int = 80,
        n_state: int | None = None,
        beta: float = 0.1,
        organ_p: float | None = None,
        refracted_strength: float = 0.1,
        prefix: str = "_nemo_fsm",
    ):
        self.brain = brain
        self.states = list(states)
        self.symbols = list(symbols)
        self.transition_map = TransitionMap(transitions)
        self.prefix = prefix

        n_state = max(n_state or n, len(self.states) * k)
        # the refracted arc-and-state core, shared with SequenceTransducer;
        # LOCAL REGIME: `organ_p` is this organ's OWN density
        # ([[SEQ-REGIME]], [[SEQ-ORGAN-EMBEDS]]), structural, set before traffic
        self.state_area, self.arc_area = add_arc_state_core(
            brain, prefix, n_arc=n, n_state=n_state, k=k, beta=beta,
            refracted_strength=refracted_strength, organ_p=organ_p)
        self.organ_p = organ_p

        # Symbols are stimuli. The reference's symbol area is a single
        # symbol->arc matrix whose per-symbol assemblies are DISJOINT row
        # blocks; independent stimulus connectomes of size k are the same
        # object, without a second area to keep in sync.
        self._sym_stim: Dict[str, str] = {}
        for sym in symbols:
            s = f"{prefix}_sym_{sym}"
            brain.add_stimulus(s, k)
            self._sym_stim[sym] = s

        # State assemblies are DISJOINT BLOCKS, as in the reference, which
        # assigns `arange(n_states * cap).reshape(n_states, cap)`.
        #
        # They were formed by projecting one stimulus per state instead, and
        # that does not give disjointness: measured at n_state=5000, k=70,
        # five emergent assemblies overlapped 0.224 pairwise against a chance
        # of k/n = 0.014 -- 16x chance. The cause is not sparsity, so raising n
        # does not fix it; it is [[sampler-merges-at-low-load]], the sampler
        # flattening distinct inputs into overlapping winners while the area is
        # nearly empty. A nearest-overlap readout over five states cannot be
        # trusted on a code like that, and the confound is avoidable, so it is
        # avoided rather than measured around.
        #
        # `materialize_area` first: a neuron ID has no compact slot until it
        # exists, and `activate_assembly` rightly refuses IDs it cannot map.
        if organ_p is not None:
            for sym_stim in self._sym_stim.values():
                brain.add_connectivity(sym_stim, self.arc_area, organ_p)

        brain.materialize_area(self.state_area)
        self._state_asm: Dict[str, Assembly] = {
            st: Assembly(self.state_area,
                         NeuronIds(np.arange(i * k, (i + 1) * k, dtype=np.uint32)))
            for i, st in enumerate(self.states)
        }

        self._table: Dict[Tuple[str, str], str] = (
            self.transition_map.deterministic_table()
        )

    # -- state cueing -------------------------------------------------------

    def state_assembly(self, state: str) -> Assembly:
        """The stored assembly for *state*, in neuron IDs."""
        return self._state_asm[state]

    def _cue_state(self, state: str, *, fix: bool = True) -> None:
        activate_assembly(self.brain, self._state_asm[state])
        if fix:
            self.brain.areas[self.state_area].fix_assembly()

    def _unfix_state(self) -> None:
        self.brain.areas[self.state_area].unfix_assembly()

    def clear_bias(self) -> None:
        """Discard the arc's accumulated refraction.

        Deliberately NOT called during training: the accumulation across
        transitions is the mechanism that keeps the arc conjunctive.
        """
        self.brain.clear_refracted_bias(self.arc_area)

    # -- training -----------------------------------------------------------

    def train_transition(self, symbol: str, from_state: str, to_state: str) -> None:
        """One presentation of ``(from_state, symbol) -> to_state``."""
        b = self.brain
        b.inhibit_areas([self.arc_area, self.state_area])
        self._cue_state(from_state)
        b.project({self._sym_stim[symbol]: [self.arc_area]},
                  {self.state_area: [self.arc_area]})
        self._unfix_state()
        self._cue_state(to_state)
        b.project({}, {self.arc_area: [self.state_area]})
        self._unfix_state()

    def train_from_list(
        self, transition_list: List[Tuple[str, str, str]], presentations: int = 1,
    ) -> None:
        for _ in range(presentations):
            for sym, fr, to in transition_list:
                self.train_transition(sym, fr, to)

    # -- running ------------------------------------------------------------

    def read_state(self) -> str:
        """Label the state area's CURRENT assembly by nearest stored state."""
        current = _snap(self.brain, self.state_area)
        return max(self.states,
                   key=lambda st: overlap(current, self._state_asm[st]))

    def step(self, symbol: str) -> str:
        """Advance one symbol from whatever the state area currently holds."""
        b = self.brain
        b.project({self._sym_stim[symbol]: [self.arc_area]},
                  {self.state_area: [self.arc_area]})
        b.project({}, {self.arc_area: [self.state_area]})
        return self.read_state()

    def run(self, symbols: Sequence[str], start_state: str) -> List[str]:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-nemo-arc-observation

        Run a symbol string from *start_state*; return the state trajectory.

        Runs inside ``brain.probe()``, the sanctioned read context. Turning
        plasticity off is NOT sufficient: projection still RECRUITS, which
        moves the compact index space and the sampler's draws, so two identical
        runs of a trained machine returned different trajectories
        (``['q1','q1','q0']`` then ``['q0','q0','q0']``) even with the weights
        and the refraction bias provably unchanged. See
        [[probe-isolation-required]] -- recruitment, not plasticity, is the
        channel by which a readout changes what it is reading.

        The arc must already have at least k materialized neurons. Training
        can establish that population; for an untrained control, explicitly call
        brain.materialize_area(fsm.arc_area) before run(). A cold area raises
        rather than recruiting during observation. Materialization is not learning
        and does not establish correct transitions.

        ``probe()`` also implies ``frozen()``, so the arc charges no refraction
        bias here, matching the reference's ``update=False``: one step of a
        sequence cannot alter the next.
        """
        b = self.brain
        with b.probe():
            b.inhibit_areas([self.arc_area, self.state_area])
            self._cue_state(start_state)
            self._unfix_state()
            return [self.step(sym) for sym in symbols]


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
