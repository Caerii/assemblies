"""The sequence transducer on the hashed substrate (DESIGN_sequence_port.md).

`programs.sequence_transducer.SequenceTransducer`, area for area and fiber
for fiber, with B brains in one launch -- each brain on ITS OWN schedule:

    s[w]  -> LEX                 the current word            (StackedStimuli)
    g[w]  -> OUT                 grounding signature         (StackedStimuli)
    LEX + STATE -> ARC           the REFRACTED conjunction   (two DenseOrganFibers)
    ARC -> STATE                 state update, feed-forward  (DenseOrganFiber)
    ARC -> OUT                   prediction                  (DenseOrganFiber)

The clock is the numpy organ's: one `tick` per input word recomputes the
arc from (LEX, STATE); `write` repeats onto that FIXED arc, so its rounds
strengthen the fibers without advancing the machine; `emit` updates the
state and reads OUT with no teacher, frozen (the numpy experiments read
under `brain.probe()`: no plasticity, no refraction charged --
[[probe-isolation-required]]).

SCHEDULES. `tick`, `write` and `reset` take a word (or target) per brain,
-1 meaning "this brain is idle this step". An idle brain's stimuli
contribute nothing and potentiate nothing, its fibers see -1 rows and skip
them, and its refraction charge is of a zero drive; its winners are junk
until its next `reset`, which is where every sentence starts. `reset` with
a mask inhibits only the brains starting a sentence (winners set to -1: the
kernels' "no source" convention).

Fibers are `DenseOrganFiber`: the organ's regime is organ_p ~ 0.2 with a
weight clip and no column scaling, which the count matrix carries and the
present-only lists do not (DESIGN_sequence_port.md, Progress). Per-fiber
density: `organ_p` on the four fibers the organ drives, the ambient `p` on
the stimuli -- the numpy organ's `add_connectivity` scoping. Seeds are the
engine's own pair seeds, so a brain's connectome here IS the engine's for
the same brain seed (GATE-1 replays the engine's winners through these
fibers and compares the drive every projection).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from ._hashed import DenseOrganFiber, HashedArea, StimulusFiber, _gain_table
from ._hashed_aligner import pair_seeds


class StackedStimuli:
    """V stimulus fibers into one area, stacked and indexed per brain by word.

    `StimulusFiber`'s arithmetic exactly -- base * gain[pot], clip, / d_j --
    on tensors [V, B, n]; `widx` [B] picks each brain's word, -1 idles it.
    """

    def __init__(self, brain_seeds, names, size, n_post, p, *, beta, w_max,
                 norm_init, max_rounds, device):
        fibers = [StimulusFiber(pair_seeds(brain_seeds, nm, nm), size, n_post, p,
                                beta=beta, w_max=w_max, norm_init=norm_init,
                                max_rounds=max_rounds, device=device)
                  for nm in names]
        self.V, self.B, self.n = len(fibers), len(brain_seeds), n_post
        # the area's tie-jitter salt XORs each afferent's per-brain seeds; a
        # stack salts by its first name, the same for a brain alone or batched
        self.seeds = torch.as_tensor(pair_seeds(brain_seeds, names[0], names[0]),
                                     dtype=torch.int32, device=device)
        self.base = torch.stack([f.base for f in fibers])            # [V, B, n]
        self.dj = (torch.stack([f.dj for f in fibers]) if norm_init else None)
        self.learns = bool(beta)
        self.pot = (torch.zeros(self.V, self.B, n_post, dtype=torch.int64, device=device)
                    if self.learns else None)
        self.gain = torch.from_numpy(_gain_table(beta, max_rounds)).to(device)
        self.hi = fibers[0].hi
        self.device = device
        self.widx = torch.zeros(self.B, dtype=torch.int64, device=device)
        self._ar = torch.arange(self.B, device=device)
        del fibers

    def set_words(self, widx):
        self.widx = widx.to(self.device)

    def contribute(self, drive, rows=None):
        w = self.widx.clamp_min(0)
        d = self.base[w, self._ar]                                    # [B, n]
        if self.learns:
            d = d * self.gain[self.pot[w, self._ar].clamp_max(self.gain.numel() - 1)]
        if self.hi != float("inf"):
            d = d.clamp_max(self.hi)
        if self.dj is not None:
            d = d / self.dj[w, self._ar]
        drive += d * (self.widx >= 0).to(d.dtype).view(-1, 1)

    def begin_episode(self):
        pass

    def observe(self, prev, new):
        if not (self.learns and new.shape[1]):
            return
        live = (self.widx >= 0).view(-1, 1) & (new >= 0)
        rows = (self.widx.clamp_min(0) * self.B + self._ar).view(-1, 1)
        flat = (rows * self.n + new.clamp_min(0)).view(-1)
        self.pot.view(-1).index_add_(0, flat, live.view(-1).to(torch.int64))

    def end_episode(self):
        pass


class HashedTransducer:
    def __init__(self, brain_seeds, vocab: Sequence[str], *, n: int,
                 n_arc: int | None = None, n_state: int | None = None,
                 k: int, p: float, beta: float = 0.10,
                 organ_p: float | None = None,
                 refracted_strength: float = 0.1,
                 state_refracted_strength: float = 0.0,
                 w_max: float | None = 20.0, norm_init: bool = True,
                 max_potentiations: int = 4096, prefix: str = "_seq",
                 tie_jitter: float = 1e-6, device: str = "cuda"):
        self.seeds = [int(s) for s in brain_seeds]
        self.B = len(self.seeds)
        self.vocab = list(vocab)
        self.word_index = {w: i for i, w in enumerate(self.vocab)}
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
        self.S = StackedStimuli(S, [f"{prefix}_s_{w}" for w in self.vocab], k, n,
                                self.p, beta=beta, w_max=w_max, norm_init=norm_init,
                                max_rounds=max_potentiations, device=device)
        self.G = StackedStimuli(S, [f"{prefix}_g_{w}" for w in self.vocab], k, n,
                                self.p, beta=beta, w_max=w_max, norm_init=norm_init,
                                max_rounds=max_potentiations, device=device)

        def fiber(src, dst, n_pre, n_post):
            return DenseOrganFiber(pair_seeds(S, src, dst), n_pre, n_post, self.organ_p,
                                   beta=beta, w_max=w_max, norm_init=norm_init,
                                   max_rounds=max_potentiations, device=device)

        self.lex_arc = fiber(self.lex_area, self.arc_area, n, self.n_arc)
        self.state_arc = fiber(self.state_area, self.arc_area, self.n_state, self.n_arc)
        self.arc_state = fiber(self.arc_area, self.state_area, self.n_arc, self.n_state)
        self.arc_out = fiber(self.arc_area, self.out_area, self.n_arc, n)
        self.out_signature: Dict[str, torch.Tensor] = {}

    # -- words as tensors -----------------------------------------------------
    def _widx(self, word) -> torch.Tensor:
        if isinstance(word, str):
            return torch.full((self.B,), self.word_index[word], dtype=torch.int64,
                              device=self.device)
        return torch.as_tensor(word, dtype=torch.int64, device=self.device)

    # -- grounding ------------------------------------------------------------
    def ground(self, rounds: int = 5) -> None:
        for w in self.vocab:
            self.reset()
            self.S.set_words(self._widx(w))
            self.lex.project(rounds, [self.S])
            self.out.inhibit()
            self.G.set_words(self._widx(w))
            self.out.project(rounds, [self.G])
            self.out_signature[w] = self.out.winners.clone()

    # -- the clock ------------------------------------------------------------
    def reset(self, mask=None) -> None:
        """Sentence boundary for every brain, or for the brains in `mask`."""
        for a in (self.lex, self.arc, self.state, self.out):
            if mask is None:
                a.inhibit()
            else:
                a.inhibit_rows(mask)

    def _rows(self):
        return {id(self.lex_arc): self.lex.winners,
                id(self.state_arc): self.state.winners,
                id(self.arc_state): self.arc.winners,
                id(self.arc_out): self.arc.winners}

    def tick(self, word, rounds: int = 3) -> None:
        self.S.set_words(self._widx(word))
        self.lex.project(rounds, [self.S])
        self.arc.project(1, [self.lex_arc, self.state_arc], rows_for=self._rows())

    def write(self, target, rounds: int = 3) -> None:
        self.G.set_words(self._widx(target))
        rows = self._rows()
        for _ in range(rounds):
            # both targets read the SAME arc: the numpy organ's simultaneous
            # update, done in sequence
            self.state.project(1, [self.arc_state], rows_for=rows)
            self.out.project(1, [self.arc_out, self.G], rows_for=rows)

    def emit(self) -> torch.Tensor:
        """Update the state and read OUT with no teacher, FROZEN."""
        rows = self._rows()
        self.state.project(1, [self.arc_state], rows_for=rows, freeze=True)
        return self.out.project(1, [self.arc_out], rows_for=rows, freeze=True)

    # -- readout --------------------------------------------------------------
    def overlaps(self, emitted: torch.Tensor) -> torch.Tensor:
        """[B, V] overlap of `emitted` with each word's OUT signature."""
        B, V = self.B, len(self.vocab)
        A = torch.zeros(B, self.n, device=self.device)
        A.scatter_(1, emitted.clamp_min(0), (emitted >= 0).float())
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
        """One sentence for EVERY brain (the numpy organ's call)."""
        self.reset()
        for a, nxt in zip(sentence, sentence[1:]):
            self.tick(a, rounds=rounds)
            self.write(nxt, rounds=rounds)

    def train_schedules(self, words, targets, starts, rounds: int = 3) -> None:
        """Per-brain schedules: `words`, `targets` [B, S] word indices (-1
        idle), `starts` [B, S] bool -- a sentence begins at this step."""
        words, targets = self._widx(words), self._widx(targets)
        starts = starts.to(self.device)
        for s in range(words.shape[1]):
            if not bool((words[:, s] >= 0).any()):
                break
            self.reset(starts[:, s])
            self.tick(words[:, s], rounds=rounds)
            self.write(targets[:, s], rounds=rounds)
