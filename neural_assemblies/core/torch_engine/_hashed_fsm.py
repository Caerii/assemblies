"""HashedArcFSM: the assigned-state transition organ at width.

`programs/nemo_fsm.NemoArcFSM` on the hashed substrate, B brains a launch,
identical in its clock and its areas (DESIGN_sequence_port.md, GATE-3):

    sym[s]  -> ARC                symbol            (StackedStimuli)
    STATE   -> ARC                current state     (DenseOrganFiber)
    ARC     -> STATE              next state        (DenseOrganFiber)

ARC is refracted -- the conjunction of symbol and state stays a conjunction
only under refraction ([[ARC-CONJUNCT-EXPOSURE]]). STATE's alphabet is
ASSIGNED, not induced: state i is the disjoint block of compact neurons
``[i k, (i + 1) k)``, as the reference assigns it, and the readout labels
the state area's current winners by the block they overlap most.

THE TEACHER-FORCED WRITE. A transition ``(from, sym) -> to`` is one
presentation: the area is inhibited, ``from`` is cued in STATE, the symbol
and STATE drive ARC (plastic: STATE -> ARC and the symbol potentiate onto
the arc winners, the arc's bias is charged), then ``to`` is cued in STATE
and ARC -> STATE is potentiated onto it -- the engine's projection into a
FIXED target, which pins the winners and still learns
(`_fixed_target_plasticity_enabled`). No drive is computed for that write.

RUNNING is the reference's ``update=False``: frozen -- nothing written, no
bias charged -- so one step cannot alter the next through refraction.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ._torch_ops import torch_ops

from ._arc_core import HashedArcCore
from ._hashed_transducer import StackedStimuli


class HashedArcFSM:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-transition-machine"""

    def __init__(self, brain_seeds, states: Sequence[str], symbols: Sequence[str],
                 transitions: Sequence[Tuple[str, str, str]], *, n_arc: int,
                 k: int, p: float, n_state: int | None = None, beta: float = 0.1,
                 refracted_strength: float = 0.1, w_max: float = 20.0,
                 norm_init: bool = False, max_potentiations: int = 4096,
                 prefix: str = "_nemo_fsm", tie_jitter: float = 0.0,
                 zero_or_size: bool = False, device: str = "cuda",
                 organ_semantics=None):
        from ..semantics import OrganSemantics, describe_hashed_arc_fsm

        actual_semantics = describe_hashed_arc_fsm(
            w_max=w_max, norm_init=norm_init,
            refracted_strength=refracted_strength, tie_jitter=tie_jitter,
            zero_or_size=zero_or_size,
        )
        if organ_semantics is not None:
            required = OrganSemantics.normalize(organ_semantics)
            mismatch = required.mismatch(actual_semantics)
            if mismatch:
                raise ValueError(f"organ_semantics mismatch: {mismatch}")
        self.organ_semantics = actual_semantics
        self.seeds = [int(s) for s in brain_seeds]
        self.B = len(self.seeds)
        self.states, self.symbols = list(states), list(symbols)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.symbol_index = {s: i for i, s in enumerate(self.symbols)}
        self.table: Dict[Tuple[str, str], str] = {(fr, sym): to for fr, sym, to in transitions}
        self.k, self.n_arc = int(k), int(n_arc)
        self.n_state = max(int(n_state or n_arc), len(self.states) * self.k)
        self.p, self.beta = float(p), float(beta)
        self.device = device
        self.state_area = f"{prefix}_state"
        self.arc_area = f"{prefix}_arc"
        S = self.seeds
        # the refracted arc-and-state core is shared with HashedTransducer
        self.core = HashedArcCore(S, prefix=prefix, n_arc=self.n_arc, n_state=self.n_state,
                                  k=k, p=p, beta=beta, refracted_strength=refracted_strength,
                                  w_max=w_max, norm_init=norm_init,
                                  max_potentiations=max_potentiations,
                                  tie_jitter=tie_jitter, device=device)
        self.arc, self.state = self.core.arc, self.core.state
        self.state_arc, self.arc_state = self.core.state_arc, self.core.arc_state
        # symbols are stimuli of size k into ARC (the reference's disjoint
        # row blocks of one symbol matrix); the engine's stimulus into a
        # SAMPLED area is a Binomial count, so Binomial is the default here
        self.sym = StackedStimuli(S, [f"{prefix}_sym_{s}" for s in self.symbols], k,
                                  self.n_arc, p, beta=beta, w_max=w_max,
                                  norm_init=norm_init, max_rounds=max_potentiations,
                                  device=device, zero_or_size=zero_or_size)
        # the assigned code: [n_states, k] compact indices, one block per state
        self.blocks = torch_ops.arange(len(self.states) * self.k, device=device,
                                   dtype=torch_ops.int64).view(len(self.states), self.k)

    # -- helpers ---------------------------------------------------------------
    def _idx(self, x, index):
        if isinstance(x, str):
            return torch_ops.full((self.B,), index[x], dtype=torch_ops.int64, device=self.device)
        return torch_ops.as_tensor(x, dtype=torch_ops.int64, device=self.device)

    def cue_state(self, state) -> None:
        """Set STATE's winners to the assigned block of `state` (a name, or
        [B] indices) -- `activate_assembly` + `fix_assembly`."""
        self.state.winners = self.blocks[self._idx(state, self.state_index)]

    def read_state(self) -> Any:
        """[B] index of the block STATE's current winners overlap most."""
        w = self.state.winners
        hit = (w.unsqueeze(1) // self.k).eq(
            torch_ops.arange(len(self.states), device=self.device).view(1, -1, 1))
        # winners outside every block (if any) count for no state
        inside = (w < len(self.states) * self.k).unsqueeze(1)
        return (hit & inside).sum(2).argmax(1)

    # -- the write --------------------------------------------------------------
    def train_transition(self, symbol, from_state, to_state) -> None:
        """One presentation of ``(from, sym) -> to`` on every brain."""
        self.arc.inhibit()
        self.cue_state(from_state)
        self.sym.set_words(self._idx(symbol, self.symbol_index))
        self.core.conjoin([self.state_arc, self.sym])
        # teacher-forced: ARC -> STATE onto the pinned target block
        self.core.teach(self.blocks[self._idx(to_state, self.state_index)])

    def train(self, presentations: int = 1) -> None:
        for _ in range(presentations):
            for (fr, sym), to in self.table.items():
                self.train_transition(sym, fr, to)

    # -- running ------------------------------------------------------------------
    def step(self, symbol, freeze: bool = True) -> Any:
        """Advance one symbol ([B] indices or a name) from whatever STATE
        holds; returns [B] state indices read out of the assembly."""
        self.sym.set_words(self._idx(symbol, self.symbol_index))
        self.core.conjoin([self.state_arc, self.sym], freeze=freeze)
        self.core.advance(freeze=freeze)
        return self.read_state()

    def run(self, symbols: Any, start_state) -> Any:
        """`symbols` [B, L] symbol indices (-1 = idle); returns [B, L] state
        indices, -1 where idle. Frozen throughout, from an inhibited arc."""
        self.arc.inhibit()
        self.cue_state(start_state)
        out = torch_ops.full_like(symbols, -1)
        for t in range(symbols.shape[1]):
            got = self.step(symbols[:, t])
            live = symbols[:, t] >= 0
            out[:, t] = torch_ops.where(live, got, out[:, t])
        return out

    def check(self) -> None:
        self.core.check()
