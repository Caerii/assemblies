"""The sequence transducer on the hashed substrate (DESIGN_sequence_port.md).

`programs.sequence_transducer.SequenceTransducer`, area for area and fiber
for fiber, with B brains in one launch:

    s[w]  -> LEX                 the current word            (StimulusFiber)
    g[w]  -> OUT                 grounding signature         (StimulusFiber)
    LEX + STATE -> ARC           the REFRACTED conjunction   (two AreaFibers)
    ARC -> STATE                 state update, feed-forward  (AreaFiber)
    ARC -> OUT                   prediction                  (AreaFiber)

The clock is the numpy organ's: one `tick` per input word recomputes the
arc from (LEX, STATE); `write` repeats onto that FIXED arc, so its rounds
strengthen the fibers without advancing the machine; `emit` updates the
state and reads OUT with no teacher, frozen (the numpy experiments read
under `brain.probe()`: no plasticity, no refraction charged --
[[probe-isolation-required]]).

Fibers are the STORE fiber (`AreaFiber`): the organ's regime is absolute
pricing with a weight clip and no column scaling, which the max-relative
present-only fiber does not price. Per-fiber density: `organ_p` on the
four fibers the organ drives, the ambient `p` on the stimuli -- exactly the
numpy organ's `add_connectivity` scoping. Seeds are the engine's own pair
seeds, so a brain's connectome here IS the engine's for the same brain seed
(GATE-1 of DESIGN_sequence_port.md replays the engine's winners through
these fibers and compares the drive every projection).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from ._hashed import AreaFiber, HashedArea, StimulusFiber
from ._hashed_aligner import pair_seeds


class HashedTransducer:
    def __init__(self, brain_seeds, vocab: Sequence[str], *, n: int,
                 n_arc: int | None = None, n_state: int | None = None,
                 k: int, p: float, beta: float = 0.10,
                 organ_p: float | None = None,
                 refracted_strength: float = 0.1,
                 state_refracted_strength: float = 0.0,
                 w_max: float | None = 20.0, norm_init: bool = True,
                 max_potentiations: int = 4096, prefix: str = "_seq",
                 tie_jitter: float = 0.0, device: str = "cuda"):
        self.seeds = [int(s) for s in brain_seeds]
        self.B = len(self.seeds)
        self.vocab = list(vocab)
        self.k, self.n = k, n
        self.n_arc = n_arc or n
        self.n_state = n_state or n
        self.p, self.beta, self.w_max = float(p), float(beta), w_max
        self.organ_p = float(organ_p) if organ_p is not None else float(p)
        self.device = device
        self.lex_area = f"{prefix}_lex"
        self.arc_area = f"{prefix}_arc"
        self.state_area = f"{prefix}_state"
        self.out_area = f"{prefix}_out"
        S = self.seeds

        def area(name, size, strength=0.0):
            return HashedArea(size, k, pair_seeds(S, name, name), device=device,
                              refracted_strength=strength, tie_jitter=tie_jitter)

        self.lex = area(self.lex_area, n)
        self.arc = area(self.arc_area, self.n_arc, refracted_strength)
        self.state = area(self.state_area, self.n_state, state_refracted_strength)
        self.out = area(self.out_area, n)

        # stimuli keep the AMBIENT density and the area's beta (the engine
        # potentiates stimulus weights with the target's beta) and clip at
        # w_max like any weight
        def stim(name, size_post):
            return StimulusFiber(pair_seeds(S, name, name), k, size_post, self.p,
                                 beta=beta, w_max=w_max, norm_init=norm_init,
                                 max_rounds=max_potentiations, device=device)

        self.s: Dict[str, StimulusFiber] = {}
        self.g: Dict[str, StimulusFiber] = {}
        for w in self.vocab:
            self.s[w] = stim(f"{prefix}_s_{w}", n)
            self.g[w] = stim(f"{prefix}_g_{w}", n)

        def fiber(src, dst, n_pre, n_post):
            return AreaFiber(pair_seeds(S, src, dst), n_pre, n_post, self.organ_p,
                             beta=beta, w_max=w_max, norm_init=norm_init,
                             synaptic_scaling=False, max_rounds=max_potentiations,
                             device=device)

        self.lex_arc = fiber(self.lex_area, self.arc_area, n, self.n_arc)
        self.state_arc = fiber(self.state_area, self.arc_area, self.n_state, self.n_arc)
        self.arc_state = fiber(self.arc_area, self.state_area, self.n_arc, self.n_state)
        self.arc_out = fiber(self.arc_area, self.out_area, self.n_arc, n)
        self.out_signature: Dict[str, torch.Tensor] = {}

    # -- grounding ------------------------------------------------------------
    def ground(self, rounds: int = 5) -> None:
        for w in self.vocab:
            self.reset()
            self.lex.project(rounds, [self.s[w]])
            self.out.inhibit()
            self.out.project(rounds, [self.g[w]])
            self.out_signature[w] = self.out.winners.clone()

    # -- the clock ------------------------------------------------------------
    def reset(self) -> None:
        for a in (self.lex, self.arc, self.state, self.out):
            a.inhibit()

    def tick(self, word: str, rounds: int = 3) -> None:
        self.lex.project(rounds, [self.s[word]])
        self.arc.project(1, [self.lex_arc, self.state_arc],
                         rows_for={id(self.lex_arc): self.lex.winners,
                                   id(self.state_arc): self.state.winners})

    def write(self, target: str, rounds: int = 3) -> None:
        rows = {id(self.arc_state): self.arc.winners,
                id(self.arc_out): self.arc.winners}
        for _ in range(rounds):
            # both targets read the SAME arc: the numpy organ's simultaneous
            # update, done in sequence
            self.state.project(1, [self.arc_state], rows_for=rows)
            self.out.project(1, [self.arc_out, self.g[target]], rows_for=rows)

    def emit(self) -> torch.Tensor:
        """Update the state and read OUT with no teacher, FROZEN."""
        rows = {id(self.arc_state): self.arc.winners,
                id(self.arc_out): self.arc.winners}
        self.state.project(1, [self.arc_state], rows_for=rows, freeze=True)
        return self.out.project(1, [self.arc_out], rows_for=rows, freeze=True)

    # -- readout --------------------------------------------------------------
    def overlaps(self, emitted: torch.Tensor) -> torch.Tensor:
        """[B, V] overlap of `emitted` with each word's OUT signature."""
        B, V = self.B, len(self.vocab)
        A = torch.zeros(B, self.n, device=self.device)
        A.scatter_(1, emitted, 1.0)
        out = torch.zeros(B, V, device=self.device)
        for i, w in enumerate(self.vocab):
            out[:, i] = A.gather(1, self.out_signature[w]).sum(1) / self.k
        return out

    def rank(self, emitted: torch.Tensor, rng) -> List[List[str]]:
        """Per brain, the vocabulary ranked by overlap, ties broken randomly
        (the numpy organ's rule: vocabulary order is not neutral)."""
        ov = self.overlaps(emitted).cpu().numpy()
        ranked = []
        for b in range(self.B):
            keyed = [(-float(ov[b, i]), rng.random(), w)
                     for i, w in enumerate(self.vocab)]
            keyed.sort()
            ranked.append([w for _o, _t, w in keyed])
        return ranked

    # -- training -------------------------------------------------------------
    def train_sentence(self, sentence: Sequence[str], rounds: int = 3) -> None:
        self.reset()
        for a, nxt in zip(sentence, sentence[1:]):
            self.tick(a, rounds=rounds)
            self.write(nxt, rounds=rounds)
