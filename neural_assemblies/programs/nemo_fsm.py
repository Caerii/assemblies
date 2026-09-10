"""
Refracted-arc FSM and migration errors for the retired Markov instrument.

Reference: dabagia.org/nemo/sequences/, dabagia.org/nemo/coinflipping/
           .reference/mdabagia-nemo/brain.py (FSMNetwork, PFANetwork)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import (
    _snap, activate_assembly,
)
from neural_assemblies.core.index_spaces import NeuronIds
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
        states = states if isinstance(states, (str, bytes)) else tuple(states)
        symbols = symbols if isinstance(symbols, (str, bytes)) else tuple(symbols)
        self.transition_map = TransitionMap(transitions).validate_domain(
            states, symbols, states[0] if states else None)
        self._table = self.transition_map.deterministic_table()
        self.states = list(states)
        self.symbols = list(symbols)
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
    """Retired invalid instrument; retained as an explicit migration error.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-arc-markov
    The historical implementation copied IDs across areas, ignored weights and
    returned table-selected successors. It is not a supported Markov sampler.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'The historical NemoMarkovPFA/AlternatingMarkovNetwork is invalid. '
            'Use ArcMarkovNetwork with explicit ArcMarkovProtocol and SeedMixtureChoice; '
            'it is a decoded-state feedback experiment, not the alternating-area protocol. '
            'See neural_assemblies/ir/VERIFICATION.md#contract-arc-markov')


class AlternatingMarkovNetwork(NemoMarkovPFA):
    """Retired with NemoMarkovPFA; no implicit replacement of historical semantics."""
