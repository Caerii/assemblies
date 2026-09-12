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

from ._arc_core import HashedArcCore
from ._hashed import DenseOrganFiber, HashedArea, StimulusFiber, _gain_table
from ._hashed_aligner import pair_seeds


class StackedStimuli:
    """V stimulus fibers into one area, stacked and indexed per brain by word.

    `StimulusFiber`'s arithmetic exactly -- base * gain[pot], clip, / d_j --
    on tensors [V, B, n]; `widx` [B] picks each brain's word, -1 idles it.
    """

    def __init__(self, brain_seeds, names, size, n_post, p, *, beta, w_max,
                 norm_init, max_rounds, device, zero_or_size=True):
        fibers = [StimulusFiber(pair_seeds(brain_seeds, nm, nm), size, n_post, p,
                                beta=beta, w_max=w_max, norm_init=norm_init,
                                max_rounds=max_rounds, device=device,
                                zero_or_size=zero_or_size)
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
        del rows  # The transducer selects its own hashed row from the word index.
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
        del prev  # Learning counts only current winners for the selected word.
        if not (self.learns and new.shape[1]):
            return
        live = (self.widx >= 0).view(-1, 1) & (new >= 0)
        rows = (self.widx.clamp_min(0) * self.B + self._ar).view(-1, 1)
        flat = (rows * self.n + new.clamp_min(0)).view(-1)
        self.pot.view(-1).index_add_(0, flat, live.view(-1).to(torch.int64))

    def end_episode(self):
        pass


class HashedTransducer:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-transducer"""

    def __init__(self, brain_seeds, vocab: Sequence[str], *, n: int,
                 n_arc: int | None = None, n_state: int | None = None,
                 k: int, p: float, beta: float = 0.10,
                 organ_p: float | None = None,
                 refracted_strength: float = 0.1,
                 state_refracted_strength: float = 0.0,
                 w_max: float | None = 20.0, norm_init: bool = True,
                 max_potentiations: int = 4096, prefix: str = "_seq",
                 tie_jitter: float = 1e-6, device: str = "cuda",
                 zero_or_size: bool = True, horizon: int = 0,
                 successor_gain: float = 1.0, state_mode: str = "induced",
                 predict_gain: float = 0.0, features=None, feature_of=None,
                 organ_semantics=None):
        from ..semantics import OrganSemantics, describe_hashed_transducer

        actual_semantics = describe_hashed_transducer(
            w_max=w_max, norm_init=norm_init,
            refracted_strength=refracted_strength,
            state_refracted_strength=state_refracted_strength,
            tie_jitter=tie_jitter, zero_or_size=zero_or_size,
            horizon=horizon, successor_gain=successor_gain,
            state_mode=state_mode,
            predict_gain=predict_gain, feature_register=bool(features),
        )
        if organ_semantics is not None:
            required = OrganSemantics.normalize(organ_semantics)
            mismatch = required.mismatch(actual_semantics)
            if mismatch:
                raise ValueError(f"organ_semantics mismatch: {mismatch}")
        self.organ_semantics = actual_semantics
        #: stimuli follow the ENGINE's zero-or-size model by default (see
        #: StimulusFiber); False gives Binomial counts, the aligner's choice
        self.zero_or_size = bool(zero_or_size)
        self.seeds = [int(s) for s in brain_seeds]
        self.B = len(self.seeds)
        self.vocab = list(vocab)
        self.word_index = {w: i for i, w in enumerate(self.vocab)}
        self.k, self.n = k, n
        self.n_arc = n_arc or n
        self.n_state = n_state or n
        #: TEMPORAL MEMORY (PREREG_temporal_memory.md). `state_mode="copy"`:
        #: the state is the PREVIOUS ARC (its winners copied after each
        #: tick; no arc -> state projection), so the state -> arc fiber is a
        #: lateral arc(t-1) -> arc(t) fiber learned Hebbian prev x new.
        #: `predict_gain` g > 0: before the arc's k-WTA the lateral drive
        #: alone names a predicted set (its top k above half its maximum)
        #: and those neurons' arc drive is multiplied by (1 + g) -- the
        #: temporal-memory rule that predicted cells win.
        self.state_mode = state_mode
        self.predict_gain = float(predict_gain)
        if state_mode not in ("induced", "copy"):
            raise ValueError(state_mode)
        if state_mode == "copy" and self.n_state != self.n_arc:
            raise ValueError("state_mode='copy' needs n_state == n_arc")
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
        # the refracted arc-and-state core is shared with HashedArcFSM
        self.core = HashedArcCore(S, prefix=prefix, n_arc=self.n_arc, n_state=self.n_state,
                                  k=k, p=self.organ_p, beta=beta,
                                  refracted_strength=refracted_strength,
                                  state_refracted_strength=state_refracted_strength,
                                  w_max=w_max, norm_init=norm_init,
                                  max_potentiations=max_potentiations,
                                  tie_jitter=tie_jitter, device=device)
        self.arc, self.state = self.core.arc, self.core.state
        self.out = area(self.out_area, n)
        # stimuli keep the AMBIENT density and the area's beta (the engine
        # potentiates stimulus weights with the target's beta) and clip at
        # w_max like any weight
        self.S = StackedStimuli(S, [f"{prefix}_s_{w}" for w in self.vocab], k, n,
                                self.p, beta=beta, w_max=w_max, norm_init=norm_init,
                                max_rounds=max_potentiations, device=device,
                                zero_or_size=self.zero_or_size)
        self.G = StackedStimuli(S, [f"{prefix}_g_{w}" for w in self.vocab], k, n,
                                self.p, beta=beta, w_max=w_max, norm_init=norm_init,
                                max_rounds=max_potentiations, device=device,
                                zero_or_size=self.zero_or_size)
        #: SUCCESSOR STATE (PREREG_successor_state.md): with `horizon` h > 0
        #: the state area is teacher-forced, during the write, toward the
        #: groundings of the next h words -- one stacked stimulus per offset
        #: into STATE -- so two prefixes with the same next h words are pushed
        #: to the same state code. h = 0 is the registered transducer: the
        #: state is induced by the arc alone.
        self.horizon = int(horizon)
        self.Gs = [StackedStimuli(S, [f"{prefix}_gs{j}_{w}" for w in self.vocab], k,
                                  self.n_state, self.p, beta=beta, w_max=w_max,
                                  norm_init=norm_init, max_rounds=max_potentiations,
                                  device=device, zero_or_size=self.zero_or_size)
                   for j in range(self.horizon)]
        #: the forcing's weight against the arc's own drive into STATE; 1.0
        #: is a full stimulus, below it the induced content survives
        for g in self.Gs:
            g.base = g.base * float(successor_gain)

        def fiber(src, dst, n_pre, n_post):
            return DenseOrganFiber(pair_seeds(S, src, dst), n_pre, n_post, self.organ_p,
                                   beta=beta, w_max=w_max, norm_init=norm_init,
                                   max_rounds=max_potentiations, device=device)

        #: FEATURE REGISTER (PREREG_feature_register.md): `features` names the
        #: feature stimuli (e.g. ["sg", "pl"]); `feature_of` maps a word to a
        #: feature index or -1. A word with a feature writes REG from that
        #: feature's stimulus on its tick; a word without one leaves REG as
        #: it is. REG -> ARC is an organ fiber, learned like the others, so
        #: the arc is the conjunction of LEX, STATE and REG.
        self.features = list(features) if features else []
        self.reg = None
        if self.features:
            self.reg_area = f"{prefix}_reg"
            self.reg = area(self.reg_area, n)
            self.F = StackedStimuli(S, [f"{prefix}_f_{f}" for f in self.features], k, n,
                                    self.p, beta=beta, w_max=w_max, norm_init=norm_init,
                                    max_rounds=max_potentiations, device=device,
                                    zero_or_size=self.zero_or_size)
            fo = [int(feature_of.get(w, -1)) for w in self.vocab]
            self.feature_of = torch.tensor(fo, dtype=torch.int64, device=device)
            self.reg_gate = True          # False: every word with a feature writes (FR-4)
            self.reg_blind = False        # True: REG held empty at test (FR-3)
        self.lex_arc = fiber(self.lex_area, self.arc_area, n, self.n_arc)
        self.state_arc, self.arc_state = self.core.state_arc, self.core.arc_state
        self.arc_out = fiber(self.arc_area, self.out_area, self.n_arc, n)
        self.reg_arc = (fiber(self.reg_area, self.arc_area, n, self.n_arc)
                        if self.reg is not None else None)
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
        for a in ((self.lex, self.arc, self.state, self.out)
                  + ((self.reg,) if self.reg is not None else ())):
            if mask is None:
                a.inhibit()
            else:
                a.inhibit_rows(mask)

    def _rows(self):
        rows = {id(self.lex_arc): self.lex.winners,
                id(self.state_arc): self.state.winners,
                id(self.arc_state): self.arc.winners,
                id(self.arc_out): self.arc.winners}
        if self.reg is not None:
            rows[id(self.reg_arc)] = self.reg.winners
        return rows

    def _write_register(self, widx, freeze):
        """Words with a feature write REG from the feature's stimulus; the
        others leave it. Ungated (FR-4): the write is the word's own feature
        for every word that has one, which is the same table -- gating is
        expressed in `feature_of` (nouns -1 gated, their number ungated)."""
        f = self.feature_of[widx.clamp_min(0)]
        f = torch.where(widx >= 0, f, torch.full_like(f, -1))
        if self.reg_blind or not bool((f >= 0).any()):
            return
        old = self.reg.winners
        self.F.set_words(f)
        new = self.reg.project(1, [self.F], freeze=freeze)
        if old.shape[1] == new.shape[1]:
            keep = (f < 0).view(-1, 1)
            self.reg.winners = torch.where(keep, old, new)
        elif old.shape[1] == 0:
            # brains without a feature this tick and no register yet: a
            # register that means nothing is worse than none; blank them
            self.reg.winners = new.masked_fill((f < 0).view(-1, 1), -1)

    def _predicted_bonus(self):
        """[B, n_arc] additive term making predicted neurons win: g x the
        full raw arc drive on the lateral fiber's top-k set (above half its
        maximum). None when nothing is predicted (an empty state)."""
        if self.predict_gain <= 0 or self.state.winners.shape[1] == 0:
            return None
        lat = torch.zeros(self.B, self.n_arc, device=self.device)
        self.state_arc.contribute(lat, self.state.winners)
        top = torch.topk(lat, self.k, dim=1)
        thresh = (0.5 * top.values[:, :1]).clamp_min(1e-12)
        mask = torch.zeros_like(lat, dtype=torch.bool)
        mask.scatter_(1, top.indices, top.values >= thresh)
        raw = lat
        self.lex_arc.contribute(raw, self.lex.winners)     # lat + lex = full raw drive
        return raw * mask.to(raw.dtype) * self.predict_gain

    def tick(self, word, rounds: int = 3, freeze: bool = False) -> None:
        """`freeze` is the numpy experiments' `probe()`: no plasticity, no
        refraction charged, for scoring."""
        widx = self._widx(word)
        self.S.set_words(widx)
        self.lex.project(rounds, [self.S], freeze=freeze)
        fibers = [self.lex_arc, self.state_arc]
        rows = {id(self.lex_arc): self.lex.winners, id(self.state_arc): self.state.winners}
        if self.reg is not None:
            self._write_register(widx, freeze)
            if self.reg.winners.shape[1]:
                fibers.append(self.reg_arc)
                rows[id(self.reg_arc)] = self.reg.winners
        bonus = self._predicted_bonus()
        self.arc.project(1, fibers, rows_for=rows, freeze=freeze, stim_drive=bonus)

    def write(self, target, rounds: int = 3, ahead=None) -> None:
        """Teacher-force OUT toward `target` (the next word) and, with a
        horizon, STATE toward `ahead` [B, h] (the next h words, -1 past the
        sentence's end)."""
        self.G.set_words(self._widx(target))
        rows = self._rows()
        state_fibers = [self.arc_state]
        if self.horizon:
            if ahead is None:
                raise ValueError("a transducer with a horizon needs `ahead`")
            ahead = self._widx(ahead)
            for j, g in enumerate(self.Gs):
                g.set_words(ahead[:, j])
            state_fibers = state_fibers + self.Gs
        for _ in range(rounds):
            # both targets read the SAME arc: the numpy organ's simultaneous
            # update, done in sequence
            if self.state_mode == "copy":
                pass                                   # the state is set by emit/copy below
            elif self.horizon:
                self.state.project(1, state_fibers, rows_for=self.core.rows())
            else:
                self.core.advance()
            self.out.project(1, [self.arc_out, self.G], rows_for=rows)
        if self.state_mode == "copy":
            self.state.winners = self.arc.winners.clone()

    def emit(self) -> torch.Tensor:
        """Update the state and read OUT with no teacher, FROZEN."""
        if self.state_mode == "copy":
            out = self.out.project(1, [self.arc_out], rows_for=self._rows(), freeze=True)
            self.state.winners = self.arc.winners.clone()
            return out
        self.core.advance(freeze=True)
        return self.out.project(1, [self.arc_out], rows_for=self._rows(), freeze=True)

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
        for i in range(len(sentence) - 1):
            self.tick(sentence[i], rounds=rounds)
            self.write(sentence[i + 1], rounds=rounds)

    def train_schedules(self, words, targets, starts, rounds: int = 3) -> None:
        """Per-brain schedules: `words`, `targets` [B, S] word indices (-1
        idle), `starts` [B, S] bool -- a sentence begins at this step."""
        words, targets = self._widx(words), self._widx(targets)
        starts = starts.to(self.device)
        S_ = words.shape[1]
        ahead_all = None
        if self.horizon:
            # word s + j, or -1 when a sentence starts in (s, s + j]
            ahead_all = torch.full((self.B, S_, self.horizon), -1, dtype=torch.int64,
                                   device=self.device)
            for j in range(1, self.horizon + 1):
                if j >= S_:
                    break
                blocked = torch.zeros(self.B, S_ - j, dtype=torch.bool, device=self.device)
                for d in range(1, j + 1):
                    blocked |= starts[:, d:S_ - j + d]
                ahead_all[:, :S_ - j, j - 1] = torch.where(blocked, torch.full_like(words[:, j:], -1),
                                                           words[:, j:])
        for s in range(S_):
            if not bool((words[:, s] >= 0).any()):
                break
            self.reset(starts[:, s])
            self.tick(words[:, s], rounds=rounds)
            self.write(targets[:, s], rounds=rounds,
                       ahead=(ahead_all[:, s] if ahead_all is not None else None))
