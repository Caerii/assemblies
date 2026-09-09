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

from typing import Dict, Sequence, Tuple

import torch

from ._hashed import DenseOrganFiber, HashedArea
from ._hashed_aligner import pair_seeds
from ._hashed_transducer import StackedStimuli


class HashedArcFSM:
    def __init__(self, brain_seeds, states: Sequence[str], symbols: Sequence[str],
                 transitions: Sequence[Tuple[str, str, str]], *, n_arc: int,
                 k: int, p: float, n_state: int | None = None, beta: float = 0.1,
                 refracted_strength: float = 0.1, w_max: float = 20.0,
                 norm_init: bool = False, max_potentiations: int = 4096,
                 prefix: str = "_nemo_fsm", tie_jitter: float = 0.0,
                 zero_or_size: bool = False, device: str = "cuda"):
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
        self.arc = HashedArea(self.n_arc, k, pair_seeds(S, self.arc_area, self.arc_area),
                              device=device, refracted_strength=refracted_strength,
                              tie_jitter=tie_jitter)
        self.state = HashedArea(self.n_state, k, pair_seeds(S, self.state_area, self.state_area),
                                device=device, tie_jitter=tie_jitter)
        # symbols are stimuli of size k into ARC (the reference's disjoint
        # row blocks of one symbol matrix); the engine's stimulus into a
        # SAMPLED area is a Binomial count, so Binomial is the default here
        self.sym = StackedStimuli(S, [f"{prefix}_sym_{s}" for s in self.symbols], k,
                                  self.n_arc, p, beta=beta, w_max=w_max,
                                  norm_init=norm_init, max_rounds=max_potentiations,
                                  device=device, zero_or_size=zero_or_size)
        self.state_arc = DenseOrganFiber(pair_seeds(S, self.state_area, self.arc_area),
                                         self.n_state, self.n_arc, p, beta=beta,
                                         w_max=w_max, norm_init=norm_init,
                                         max_rounds=max_potentiations, device=device)
        self.arc_state = DenseOrganFiber(pair_seeds(S, self.arc_area, self.state_area),
                                         self.n_arc, self.n_state, p, beta=beta,
                                         w_max=w_max, norm_init=norm_init,
                                         max_rounds=max_potentiations, device=device)
        # the assigned code: [n_states, k] compact indices, one block per state
        self.blocks = torch.arange(len(self.states) * self.k, device=device,
                                   dtype=torch.int64).view(len(self.states), self.k)

    # -- helpers ---------------------------------------------------------------
    def _idx(self, x, index):
        if isinstance(x, str):
            return torch.full((self.B,), index[x], dtype=torch.int64, device=self.device)
        return torch.as_tensor(x, dtype=torch.int64, device=self.device)

    def cue_state(self, state) -> None:
        """Set STATE's winners to the assigned block of `state` (a name, or
        [B] indices) -- `activate_assembly` + `fix_assembly`."""
        self.state.winners = self.blocks[self._idx(state, self.state_index)]

    def read_state(self) -> torch.Tensor:
        """[B] index of the block STATE's current winners overlap most."""
        w = self.state.winners
        hit = (w.unsqueeze(1) // self.k).eq(
            torch.arange(len(self.states), device=self.device).view(1, -1, 1))
        # winners outside every block (if any) count for no state
        inside = (w < len(self.states) * self.k).unsqueeze(1)
        return (hit & inside).sum(2).argmax(1)

    # -- the write --------------------------------------------------------------
    def train_transition(self, symbol, from_state, to_state) -> None:
        """One presentation of ``(from, sym) -> to`` on every brain."""
        self.arc.inhibit()
        self.cue_state(from_state)
        self.sym.set_words(self._idx(symbol, self.symbol_index))
        self.arc.project(1, [self.state_arc, self.sym],
                         rows_for={id(self.state_arc): self.state.winners})
        # teacher-forced: ARC -> STATE onto the pinned target block
        self.cue_state(to_state)
        self.arc_state.begin_episode()
        self.arc_state.observe(self.arc.winners, self.state.winners)
        self.arc_state.end_episode()
        self.state.ever.scatter_(1, self.state.winners, True)

    def train(self, presentations: int = 1) -> None:
        for _ in range(presentations):
            for (fr, sym), to in self.table.items():
                self.train_transition(sym, fr, to)

    # -- running ------------------------------------------------------------------
    def step(self, symbol, freeze: bool = True) -> torch.Tensor:
        """Advance one symbol ([B] indices or a name) from whatever STATE
        holds; returns [B] state indices read out of the assembly."""
        self.sym.set_words(self._idx(symbol, self.symbol_index))
        self.arc.project(1, [self.state_arc, self.sym],
                         rows_for={id(self.state_arc): self.state.winners}, freeze=freeze)
        self.state.project(1, [self.arc_state],
                           rows_for={id(self.arc_state): self.arc.winners}, freeze=freeze)
        return self.read_state()

    def run(self, symbols: torch.Tensor, start_state) -> torch.Tensor:
        """`symbols` [B, L] symbol indices (-1 = idle); returns [B, L] state
        indices, -1 where idle. Frozen throughout, from an inhibited arc."""
        self.arc.inhibit()
        self.cue_state(start_state)
        out = torch.full_like(symbols, -1)
        for t in range(symbols.shape[1]):
            got = self.step(symbols[:, t])
            live = symbols[:, t] >= 0
            out[:, t] = torch.where(live, got, out[:, t])
        return out

    def check(self) -> None:
        self.state_arc.check()
        self.arc_state.check()
