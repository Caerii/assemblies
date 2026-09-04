"""The cross-situational aligner on the GENERATED-connectome substrate, batched.

WHY THIS EXISTS. `research/experiments/unaligned_scenes.Aligner` passed its
registration on `numpy_sparse` at ~1.4 ms of n-independent overhead per
projection plus ~0.2 us per neuron -- a serial term that no kernel can touch
and that turns a capacity sweep into hours (DESIGN_hashed_aligner.md). Here
the same learner is composed from the hashed pieces (`_hashed.py`): two
`HashedArea`s, one `StimulusFiber` per word and per feature, one `AreaFiber`
for LEX -> FEAT with column scaling. B independent brains -- seeds, in a
study -- run in one launch, and every round is a few kernels.

THE PROTOCOL, exactly the numpy learner's (its Amendments 2-5):

    per (word w, bundle b) co-presentation:
        inhibit LEX and FEAT
        round 0    LEX <- phon(w)          FEAT <- feat(f) for f in b
        rounds 1.. LEX <- phon(w)          FEAT <- feat(f) + LEX -> FEAT
    both areas update from the SAME pre-round winners (Brain.project batches
    its targets), so FEAT's cross-fiber drive uses LEX's winners from the
    previous round, and the Hebbian write is prev(LEX) x new(FEAT).

    readout, frozen:
        bundle_assembly(b)  FEAT <- feat(f) for f in b, one round
        reconstruct(w)      LEX <- phon(w); then FEAT <- LEX -> FEAT alone

Stimulus potentiation, norm_init pricing, w_max caps and column scaling all
live in the fibers, verified against the engine at < 5e-6 relative drive
(`test_hashed_aligner_parity.py`, the parity gate of this unit).

THE PINNED-WINNER DIAGNOSTIC. If FEAT's winners under stimulus + cross-fiber
equal its winners under the stimulus alone at every training round, the
whole training collapses to one GEMM, `count = X^T C Y` ([[HEBB-OUTER-
PRODUCT]]), and the readout to another. `track_pinned=True` measures that
overlap every cross-fiber round; the shortcut is exact only where it reads
1.000, and this is how far it may be trusted.
"""
from __future__ import annotations

import torch

from ..numpy_engine import _seeding
from ._hashed import AreaFiber, DenseAreaFiber, HashedArea, PresentFiber, StimulusFiber

LEX, FEAT = "LEX", "FEAT"


def _i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def pair_seeds(brain_seeds, src, dst):
    """The engine's own content-addressed fiber seeds, one per brain."""
    return [_i32(_seeding.fnv1a_pair_seed(int(s), src, dst))
            for s in brain_seeds]


class HashedAligner:
    """B brains, one learner each, one launch per round."""

    def __init__(self, brain_seeds, words, features, *, n, k, feat_n, feat_k,
                 stim_size=None, p=0.05, beta=0.1, w_max=None,
                 norm_init=True, scaling=True, rounds_word=5,
                 max_potentiations=4096, device="cuda", track_pinned=False,
                 tie_jitter=1e-6, stim_beta=0.0, stim_gain=None,
                 store="present"):
        self.seeds = [int(s) for s in brain_seeds]
        self.B = len(self.seeds)
        self.n, self.k, self.feat_n, self.feat_k = n, k, feat_n, feat_k
        self.p, self.beta, self.w_max = p, beta, w_max
        self.rounds_word = rounds_word
        self.device = device
        stim_size = k if stim_size is None else int(stim_size)
        # UNCLIPPED BY DEFAULT. Column scaling on the cross fiber bounds the
        # weights, and a clip would make the factored scale inexact (see
        # AreaFiber). The numpy learner's registration holds without the clip
        # -- U1 reads 0.985 and 1.000 on its first two seeds at w_max=None --
        # so nothing scientific rests on it.
        # TABLES ARE SIZED BY POTENTIATION COUNT, NOT BY EPISODE. A cell's
        # chain table (`max_rounds` on a fiber) must extend past the most
        # times that cell can ever co-fire; the first build sized the cross
        # fiber's by its 1-round episodes and every count above 1 clamped to
        # tab[1] -- caught by the parity gate. The cross table is a per-step
        # clip, so a long one costs nothing. The stimulus gain table has no
        # clip and must only reach the cap: base * (1+beta)^c >= w_max *
        # max(1, size p) is guaranteed once (1+beta)^c >= that cap, since
        # base >= 1 wherever the stimulus connects at all.
        import math
        if stim_beta <= 0:
            stim_table = 1                     # anchors do not learn
        elif w_max is None:
            # unclipped: the gain table can only reach the float32-finite
            # depth, ln(3.4e38)/ln(1+beta) ~ 930 at beta=0.1; a stimulus
            # potentiated more often than that has an infinite weight in the
            # engine too, so the cap is the representable regime, not a loss
            stim_table = min(max_potentiations,
                             int(88.0 / math.log1p(beta)))
        else:
            stim_cap = w_max * max(1.0, stim_size * p)
            feat_cap = w_max * max(1.0, feat_k * p)
            stim_table = int(math.ceil(math.log(max(stim_cap, feat_cap))
                                       / math.log1p(beta))) + 2
        # Stimulus-driven areas tie at the bar; see `HashedArea.tie_jitter`.
        self.lex = HashedArea(n, k, pair_seeds(self.seeds, LEX, LEX),
                              device=device, tie_jitter=tie_jitter)
        self.feat = HashedArea(feat_n, feat_k,
                               pair_seeds(self.seeds, FEAT, FEAT),
                               device=device, tie_jitter=tie_jitter)
        # STIMULI ARE ANCHORS AND DO NOT LEARN (stim_beta=0 by default). With
        # stimulus potentiation on, a feature shared by two bundles (ANIMAL in
        # DOG and CAT) wins in both bundles' scenes, potentiates far more than
        # the distinctive feature, and swamps FEAT's k-WTA: measured, the DOG
        # and CAT bundle assemblies MERGED during training (stability 15/50)
        # while every LEX assembly stayed 50/50 -- the superordinate-swallows-
        # identity mechanism the reference counters with refraction. The numpy
        # learner never met it because its 0-or-size stimulus weights start AT
        # the w_max cap, so its stimulus potentiation was a no-op by accident.
        # Making the anchor explicit puts all learning in the cross fiber,
        # which is where the registration said alignment lives. The parity
        # test passes stim_beta=beta to match the engine it replays.
        self.phon = {
            w: StimulusFiber(pair_seeds(self.seeds, f"phon_{w}", LEX),
                             stim_size, n, p, beta=stim_beta, w_max=w_max,
                             norm_init=norm_init, max_rounds=stim_table,
                             device=device)
            for w in words}
        self.featf = {
            f: StimulusFiber(pair_seeds(self.seeds, f"feat_{f}", FEAT),
                             feat_k, feat_n, p, beta=stim_beta, w_max=w_max,
                             norm_init=norm_init, max_rounds=stim_table,
                             device=device)
            for f in features}
        # One 1-round episode per training round keeps the mask narrow; the
        # store's LSM absorbs the appends. `max_rounds` here is the chain
        # table's reach in POTENTIATIONS (see above), not the episode.
        if store == "present":
            # store only what exists; a warp per brain (DESIGN_present_only.md)
            if w_max is not None:
                raise ValueError("the present-only fiber is the unclipped regime")
            self.cross = PresentFiber(pair_seeds(self.seeds, LEX, FEAT), n,
                                      feat_n, p, beta=beta, norm_init=norm_init,
                                      synaptic_scaling=scaling,
                                      max_rounds=max_potentiations, device=device)
        elif store == "dense":
            # The count matrix fits at study sizes: one launch per drive, one
            # per write, no store walk (DESIGN_dense_cross_fiber.md).
            if w_max is not None:
                raise ValueError("the dense fiber is the unclipped regime")
            self.cross = DenseAreaFiber(pair_seeds(self.seeds, LEX, FEAT), n,
                                        feat_n, p, beta=beta,
                                        norm_init=norm_init,
                                        synaptic_scaling=scaling,
                                        max_rounds=max_potentiations,
                                        device=device)
        else:
            self.cross = AreaFiber(pair_seeds(self.seeds, LEX, FEAT), n,
                                   feat_n, p, beta=beta, w_max=w_max,
                                   norm_init=norm_init,
                                   synaptic_scaling=scaling,
                                   max_rounds=max_potentiations,
                                   device=device)
        # ANCHORED ASSEMBLIES ARE CONSTANTS: a non-learning stimulus gives a
        # constant drive and deterministic ties, so LEX(word) and FEAT's
        # stimulus-only assembly for a bundle never change. Computed once.
        self._lex_cache = {}
        self._feat_cache = {}
        # THE ANCHOR SHARE. Perceived features must decide FEAT's winners
        # during training or the cross fiber binds each word to a FEAT set of
        # its own making: measured with gain 1, FEAT's winners overlapped the
        # stimulus-only assembly by 0.14 and alignment reached 0.7 against the
        # numpy learner's 0.99. The numpy learner's anchor was ~1/p stronger
        # by the 0-or-size accident; here it is a parameter, default 1/p, and
        # `track_pinned` measures whether it is enough.
        self.stim_gain = (1.0 / p) if stim_gain is None else float(stim_gain)
        for f in list(self.phon.values()) + list(self.featf.values()):
            f.drive_gain = self.stim_gain
        self.track_pinned = track_pinned
        self.pinned = []          # per cross-fiber round: mean overlap over B

    # -- training -----------------------------------------------------------
    def _stims(self, bundle):
        return [self.featf[f] for f in bundle]

    def _anchored(self, area, cache, key, fibers):
        """The area's winners under these non-learning stimuli, cached."""
        w = cache.get(key)
        if w is None:
            area.inhibit()
            w = area.project(1, fibers, freeze=True).clone()
            cache[key] = w
        area.winners = w
        return w

    def step(self, word, bundle):
        """One (word, bundle) co-presentation: its cross rounds.

        Round 0 (stimuli only) and every LEX round are anchored constants
        (see `_anchored`), so the step is exactly `rounds_word` FEAT rounds.
        """
        phon = self.phon[word]
        stims = self._stims(bundle)
        if phon.learns or any(s.learns for s in stims):
            raise ValueError("cached anchors need non-learning stimuli")
        self._anchored(self.lex, self._lex_cache, word, [phon])
        self._anchored(self.feat, self._feat_cache, tuple(bundle), stims)
        # The cross rounds are ONE episode of the cross fiber (one mask, one
        # fold into the store), managed here because the rounds interleave
        # two areas; per-round episodes cost a store append and merge each.
        self.cross.begin_episode()
        for _ in range(self.rounds_word):
            lex_prev = self.lex.winners                 # anchored, constant
            if self.track_pinned:
                pinned = self._feat_cache[tuple(bundle)]
            new = self.feat.project(1, stims + [self.cross],
                                    rows_for={id(self.cross): lex_prev},
                                    manage_episodes=False)
            if self.track_pinned:
                m = torch.zeros(self.B, self.feat_n, dtype=torch.bool,
                                device=self.device)
                m.scatter_(1, new, True)
                ov = torch.gather(m, 1, pinned.long()).sum(1).float()
                self.pinned.append(float((ov / self.feat_k).mean()))
        self.cross.end_episode()

    def train(self, exp, rng):
        """`exp`: list of (words, bundles); every word x every bundle."""
        # Size the cross fiber's pricing table by the rounds this corpus can
        # deliver: one potentiation per cross round is the worst case.
        total = sum(len(ws) * len(bs) for ws, bs in exp) * self.rounds_word
        self.cross.ensure_depth(total)
        order = list(range(len(exp)))
        rng.shuffle(order)
        for i in order:
            words, bundles = exp[i]
            for w in words:
                for b in bundles:
                    self.step(w, b)
        if hasattr(self.cross, "check"):
            self.cross.check()

    # -- frozen readouts -----------------------------------------------------
    def bundle_assembly(self, bundle):
        self.feat.inhibit()
        return self.feat.project(1, self._stims(bundle), freeze=True)

    def reconstruct(self, word):
        self.lex.inhibit()
        self.lex.project(1, [self.phon[word]], freeze=True)
        rows = self.lex.winners
        self.feat.inhibit()
        return self.feat.project(1, [self.cross],
                                 rows_for={id(self.cross): rows}, freeze=True)

    # -- scoring -------------------------------------------------------------
    def overlap_table(self, words, inventory):
        """[V, I, B] overlap counts / feat_k between reconstruct(w) and
        bundle_assembly(b), all brains, by mask GEMM."""
        V, I = len(words), len(inventory)
        R = torch.zeros(V, self.B, self.feat_n, dtype=torch.float32,
                        device=self.device)
        A = torch.zeros(I, self.B, self.feat_n, dtype=torch.float32,
                        device=self.device)
        for vi, w in enumerate(words):
            R[vi].scatter_(1, self.reconstruct(w), 1.0)
        for ii, b in enumerate(inventory):
            A[ii].scatter_(1, self.bundle_assembly(b), 1.0)
        out = torch.einsum("vbn,ibn->vib", R, A) / self.feat_k
        return out                                    # [V, I, B]
