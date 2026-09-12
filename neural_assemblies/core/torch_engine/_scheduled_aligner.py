"""The aligner as a SCHEDULE: B independent tasks with identical area shapes.

DESIGN_scheduled_training.md, layer 1. `HashedAligner` batches seeds of one
corpus; here every brain carries its own vocabulary, its own bundle
inventory and its own step schedule, and any tasks sharing (n, k, feat_n,
feat_k, stim_size) run in one launch -- every V value and every seed of a
capacity cell together. A round's cost is flat in B, so width is the first
and cheapest lever.

WHAT A BRAIN CARRIES (all padded with -1):

    words     [B, S]          word index per step
    bundles   [B, S]          bundle index per step
    features  [B, I, F_per]   feature indices of bundle j
    targets   [B, V]          the scorer's answer: bundle index of word i

Word i of brain b is brain b's own word. Its phon fiber is seeded by
(seed_b, name_i) where the name defaults to "phon_i"; a caller reproducing
`HashedAligner` passes the same names so the connectomes are the same.

ANCHORS ARE CONSTANTS. LEX winners per (brain, word) and FEAT's stimulus
drive, jitter and winners per (brain, bundle) are computed once and gathered
by the schedule. A step is exactly its cross rounds. The tie jitter follows
`HashedArea._jitter` exactly: salted by the seeds of the fibers that fire,
which for a cross round includes the cross fiber.
"""
from __future__ import annotations

import torch
from typing import Any

from ._hashed import HashedArea, PresentFiber, StimulusFiber, _fused_cuda
from ._hashed_aligner import FEAT, LEX, pair_seeds


def schedule_of(exp, word_index, bundle_index, order):
    """One brain's schedule from its experience, in `HashedAligner.train`'s
    order: sentences in `order`, then every word x every bundle."""
    ws, bs = [], []
    for i in order:
        words, bundles = exp[i]
        for w in words:
            for b in bundles:
                ws.append(word_index[w])
                bs.append(bundle_index[b])
    return ws, bs


def pad_schedules(per_brain, device="cuda"):
    """List of (words, bundles) lists -> [B, S] int64 tensors, -1 padded."""
    S = max(len(w) for w, _ in per_brain)
    B = len(per_brain)
    W = torch.full((B, S), -1, dtype=torch.int64)
    Bd = torch.full((B, S), -1, dtype=torch.int64)
    for b, (w, bb) in enumerate(per_brain):
        W[b, :len(w)] = torch.tensor(w)
        Bd[b, :len(bb)] = torch.tensor(bb)
    return W.to(device), Bd.to(device)


def _hash_jitter(salt, n, jitter, device, slab=16):
    """`HashedArea._jitter`'s arithmetic on a [B, I] salt -> [B, I, n].

    Computed `slab` bundles at a time: the int64 temporaries of the whole
    [B, I, n] block were three times the result and, at V = 1024 and a
    wide FEAT, most of the card. Same ops per element, so the same floats."""
    cols = torch.arange(n, dtype=torch.int64, device=device)
    B, I = salt.shape
    out = torch.empty(B, I, n, dtype=torch.float32, device=device)
    for i0 in range(0, I, slab):
        h = (cols.view(1, 1, -1) ^ salt[:, i0:i0 + slab].view(B, -1, 1)) * 0x9E3779B1
        h = (h ^ (h >> 15)) * 0x85EBCA6B
        h = (h ^ (h >> 13)) & 0xFFFFFFFF
        out[:, i0:i0 + slab] = h.to(torch.float32) * (jitter / 4294967296.0)
    return out


class ScheduledAligner:
    """Schedule-batched alignment with the same checked model relation.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-hashed-aligner
    """

    mod: Any
    phon: Any
    featf: Any
    cross: Any

    def __init__(self, brain_seeds, *, n, k, feat_n, feat_k, n_words,
                 n_features, word_names=None, feature_names=None,
                 stim_size=None, p=0.05, beta=0.1, norm_init=True,
                 scaling=True, rounds_word=2, stim_gain=None, tie_jitter=1e-6,
                 max_potentiations=4096, device="cuda",
                 aligner_semantics=None):
        from ..semantics import AlignerSemantics, describe_hashed_aligner
        actual_semantics = describe_hashed_aligner(
            p=p, beta=beta, w_max=None, norm_init=norm_init,
            scaling=scaling, rounds_word=rounds_word,
            tie_jitter=tie_jitter, stim_beta=0.0,
            stim_gain=stim_gain, store="present",
        )
        if aligner_semantics is not None:
            required = AlignerSemantics.normalize(aligner_semantics)
            mismatch = required.mismatch(actual_semantics)
            if mismatch:
                raise ValueError(f"aligner_semantics mismatch: {mismatch}")
        self.aligner_semantics = actual_semantics
        self.mod = _fused_cuda.load()
        self.seeds = [int(s) for s in brain_seeds]
        self.B = len(self.seeds)
        self.n, self.k, self.feat_n, self.feat_k = n, k, feat_n, feat_k
        self.V, self.F = int(n_words), int(n_features)
        self.p, self.beta = p, beta
        self.rounds_word = rounds_word
        self.device = device
        self.tie_jitter = float(tie_jitter)
        stim_size = k if stim_size is None else int(stim_size)
        gain = (1.0 / p) if stim_gain is None else float(stim_gain)
        wnames = word_names or [f"phon_{i}" for i in range(self.V)]
        fnames = feature_names or [f"feat_{f}" for f in range(self.F)]
        self.lex = HashedArea(n, k, pair_seeds(self.seeds, LEX, LEX),
                              device=device, tie_jitter=tie_jitter)
        self.feat = HashedArea(feat_n, feat_k,
                               pair_seeds(self.seeds, FEAT, FEAT),
                               device=device, tie_jitter=tie_jitter)
        self.phon = [StimulusFiber(pair_seeds(self.seeds, wn, LEX), stim_size,
                                   n, p, beta=0.0, w_max=None,
                                   norm_init=norm_init, max_rounds=1,
                                   device=device) for wn in wnames]
        self.featf = [StimulusFiber(pair_seeds(self.seeds, fn, FEAT), feat_k,
                                    feat_n, p, beta=0.0, w_max=None,
                                    norm_init=norm_init, max_rounds=1,
                                    device=device) for fn in fnames]
        for f in self.phon + self.featf:
            f.drive_gain = gain
        self.cross = PresentFiber(pair_seeds(self.seeds, LEX, FEAT), n,
                                  feat_n, p, beta=beta, norm_init=norm_init,
                                  synaptic_scaling=scaling,
                                  max_rounds=max_potentiations,
                                  device=device)
        self._prepared = False
        #: brains per block in the persistent kernel (a warp each); None picks
        #: the most that fit the kernel's shared-memory budget
        self.warps_per_block = None

    def _warps_per_block(self):
        """Mirror of the kernel's per-warp shared bytes (pr_warp_bytes)."""
        if self.warps_per_block is not None:
            return int(self.warps_per_block)
        N, K, KW = self.feat_n, self.k, self.feat_k
        W = (N + 31) // 32
        per = KW * 8 + N * 4 + 256 * 4 + KW * 4 + K * 4 + KW * 4 + 16 + W * 4 + 512 * 4 + N * 2 + N * 2
        per = (per + 7) & ~7
        budget = 96 * 1024 - 4 * 1024                  # minus the staged prices
        return max(1, min(4, budget // per))

    # -- anchors -------------------------------------------------------------
    def _select(self, area, drive, fibers):
        ranked = drive + area._jitter(fibers) if area.tie_jitter > 0 else drive
        sel, ovf = self.mod.topk_select(ranked, area.k)
        if int(ovf.max()):
            raise RuntimeError("k-WTA candidate set overflowed")
        return sel.to(torch.int64)

    def prepare(self, features):
        """Cache every anchor. `features`: [B, I, F_per] int64, -1 padded."""
        assert self.phon is not None and self.featf is not None
        B, dev = self.B, self.device
        self.features = features.to(dev)
        I, Fper = self.features.shape[1], self.features.shape[2]
        ar = torch.arange(B, device=dev)
        # LEX winners per (brain, word)
        self.lex_cache = torch.full((B, self.V, self.k), -1, dtype=torch.int64,
                                    device=dev)
        for i, ph in enumerate(self.phon):
            d = torch.zeros(B, self.n, device=dev)
            ph.contribute(d)
            self.lex_cache[:, i] = self._select(self.lex, d, [ph])
        # feature constants [B, F+1, feat_n] (slot F is the zero pad) -> bundle drive
        consts = torch.zeros(B, self.F + 1, self.feat_n, device=dev)
        for f, ff in enumerate(self.featf):
            ff.contribute(consts[:, f])
        idx = torch.where(self.features < 0,
                          torch.full_like(self.features, self.F),
                          self.features)                     # [B, I, Fper]
        self.bundle_drive = consts[ar.view(B, 1, 1), idx].sum(dim=2)   # [B, I, feat_n]
        # salts: the seeds of the fibers that fire, XORed as HashedArea does
        fseeds = torch.stack([ff.seeds.to(torch.int64) & 0xFFFFFFFF
                              for ff in self.featf] +
                             [torch.zeros(B, dtype=torch.int64, device=dev)],
                             dim=1)                          # [B, F+1]
        salt = torch.zeros(B, I, dtype=torch.int64, device=dev)
        for s in range(Fper):
            salt = salt ^ fseeds[ar.view(B, 1), idx[:, :, s]]
        cross_salt = (self.cross.seeds.to(torch.int64) & 0xFFFFFFFF).view(B, 1)
        self.jit_anchor = _hash_jitter(salt, self.feat_n, self.tie_jitter, dev)
        self.jit_cross = _hash_jitter(salt ^ cross_salt, self.feat_n,
                                      self.tie_jitter, dev)
        # the reconstruction readout fires the cross fiber ALONE, so its ties
        # break on a salt of the cross seeds only -- as HashedAligner's does
        self.jit_recon = _hash_jitter(cross_salt.view(B, 1), self.feat_n,
                                      self.tie_jitter, dev)[:, 0]   # [B, feat_n]
        # FEAT winners per (brain, bundle) under the stimulus alone
        self.feat_cache = torch.full((B, I, self.feat_k), -1,
                                     dtype=torch.int64, device=dev)
        for j in range(I):
            ranked = self.bundle_drive[:, j] + self.jit_anchor[:, j]
            sel, _ = self.mod.topk_select(ranked, self.feat_k)
            self.feat_cache[:, j] = sel.to(torch.int64)
        # the anchors are cached: the fibers (one per word and per feature)
        # and the prepare-time tensors are dead -- at V = 1024 and a wide
        # FEAT they were most of the card
        del consts, idx, fseeds, salt
        self.phon, self.featf, self.jit_anchor = None, None, None
        torch.cuda.empty_cache()
        self._prepared = True

    # -- training ------------------------------------------------------------
    def train(self, words, bundles, device_loop=False):
        """`words`, `bundles`: [B, S] int64 schedules, -1 past the end.

        ``device_loop=True`` runs the whole schedule in ONE launch
        (`present_train_kernel`, DESIGN_present_only.md): one WARP per
        brain walks its rows in order over the present-only lists; the
        selection is a warp-level radix select on `topk_select`'s own key;
        no block barrier inside a round. Gated to give IDENTICAL tables to
        the python loop below.
        """
        assert self._prepared, "call prepare(features) first"
        B, dev = self.B, self.device
        S = words.shape[1]
        self.cross.ensure_depth(S * self.rounds_word)
        words, bundles = words.to(dev), bundles.to(dev)
        if device_loop:
            cf = self.cross
            nnz, nsh = cf.price_head()
            self.mod.present_train(
                words, bundles, self.lex_cache, self.bundle_drive,
                self.jit_cross, cf.ent, cf.cmax, cf.mass, cf.scale,
                cf.invdj, cf.rel, float(cf.setpoint), self.rounds_word,
                self.feat_k, cf.err, nsh, nnz, self._warps_per_block(),
                1 if cf.absolute else 0)
            torch.cuda.synchronize()
            cf.check()
            return
        ar = torch.arange(B, device=dev)
        neg_rows = torch.full((B, self.k), -1, dtype=torch.int64, device=dev)
        neg_new = torch.full((B, self.feat_k), -1, dtype=torch.int64, device=dev)
        for s in range(S):
            w, bid = words[:, s], bundles[:, s]
            live = ((w >= 0) & (bid >= 0)).view(B, 1)
            if not bool(live.any()):
                break
            rows = torch.where(live, self.lex_cache[ar, w.clamp_min(0)], neg_rows)
            stim = self.bundle_drive[ar, bid.clamp_min(0)]
            jit = self.jit_cross[ar, bid.clamp_min(0)]
            for _ in range(self.rounds_word):
                d = stim.clone()
                self.cross.contribute(d, rows)
                sel, _ = self.mod.topk_select(d + jit, self.feat_k)
                new = torch.where(live, sel.to(torch.int64), neg_new)
                self.cross.observe(rows, new)
        self.cross.check()

    # -- readout -------------------------------------------------------------
    def overlap_table(self):
        """[B, V, I] overlaps between reconstruct(word) and each bundle's
        stimulus-only assembly, per brain."""
        B, dev = self.B, self.device
        I = self.feat_cache.shape[1]
        A = torch.zeros(B, I, self.feat_n, device=dev)
        A.scatter_(2, self.feat_cache.clamp_min(0), (self.feat_cache >= 0).float())
        R = torch.zeros(B, self.V, self.feat_n, device=dev)
        for i in range(self.V):
            d = torch.zeros(B, self.feat_n, device=dev)
            self.cross.contribute(d, self.lex_cache[:, i])
            sel, _ = self.mod.topk_select(d + self.jit_recon, self.feat_k)
            R[:, i].scatter_(1, sel.to(torch.int64), 1.0)
        return torch.einsum("bvn,bin->bvi", R, A) / self.feat_k

    def type_accuracy(self, targets, n_bundles, exposures, min_exposures):
        """Per-brain accuracy over words with enough exposures, argmax over
        that brain's own bundles. `targets`: [B, V] bundle index or -1;
        `n_bundles`: [B]; `exposures`: [B, V]."""
        tab = self.overlap_table()                                  # [B, V, I]
        B, V, I = tab.shape
        dev = tab.device
        valid = torch.arange(I, device=dev).view(1, 1, I) < n_bundles.view(B, 1, 1).to(dev)
        best = tab.masked_fill(~valid, -1.0).argmax(dim=2)          # [B, V]
        targets, exposures = targets.to(dev), exposures.to(dev)
        scored = (targets >= 0) & (exposures >= min_exposures)
        hit = (best == targets) & scored
        return (hit.sum(1).float() / scored.sum(1).clamp_min(1).float(),
                scored.sum(1))
