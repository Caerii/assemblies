"""The AREA: its winners, its afferent fibers, the k-WTA reading them, refraction and
long-range inhibition.

Moved from _hashed.py unchanged; _hashed.py re-exports every name."""
from __future__ import annotations

from ._torch_ops import torch_ops
from typing import Any
from . import _fused_cuda
from ._hashed_common import per_brain, _raise_overflow


class HashedArea:
    """One area: its winners, its afferent fibers, and the k-WTA reading them.

    The projection loop is the algebra, once:

        for each round
            drive = SUM over fibers of fiber.contribute(...)
            winners = k-WTA(drive)
            each fiber observes (its source winners, the new winners)

    Nothing in here knows about `norm_init`, `w_max` or stimulus pricing --
    those live in the fiber that owns them, which is why adding an input is
    adding a fiber rather than a flag.
    """

    mod: Any

    def __init__(self, n, k, seeds, device="cuda", refracted_strength=0.0,
                 tie_jitter=0.0):
        self.mod = _fused_cuda.load()
        if self.mod is None:
            raise RuntimeError(f"fused kernels unavailable: "
                               f"{_fused_cuda.last_error()}")
        self.n, self.k, self.B = n, k, len(seeds)
        self.device = device
        self.winners = torch_ops.zeros(self.B, 0, dtype=torch_ops.int64, device=device)
        #: neurons that have EVER fired -- what `rows/n` reads. Tracked here
        #: because the round masks are per-episode and get dropped.
        self.ever = torch_ops.zeros(self.B, n, dtype=torch_ops.bool, device=device)
        self.rounds_seen = 0
        #: REFRACTION, the engine's `refracted` mode (`core/_homeostasis.py`).
        #: A per-NEURON bias subtracted from drive before k-WTA and charged at
        #: the winners as `raw_drive * strength`, gated on plasticity exactly
        #: like the Hebbian write. It belongs to the area, not to a fiber: the
        #: reference charges against the TOTAL raw input of the winner, and
        #: the bias persists across `inhibit()` -- `RefractedArea.inhibit`
        #: does not touch it, only `reset()` does. At strength == beta the
        #: bias is the exact anti-Hebbian counterweight on a neuron's own
        #: repeated input (net drive stays at its base value), and a handicap
        #: on every other input; see PREREG_refraction_capacity.md.
        strengths = per_brain(refracted_strength, self.B)
        if strengths is None:
            self.refracted_strength: Any = float(refracted_strength or 0.0)
            charged = self.refracted_strength > 0
        else:
            # one strength per brain, [B, 1]: a swept learning rate carries
            # its own 0.5 beta
            self.refracted_strength = torch_ops.tensor(
                strengths, dtype=torch_ops.float32, device=device).view(-1, 1)
            charged = any(v > 0 for v in strengths)
        self.bias = (torch_ops.zeros(self.B, n, dtype=torch_ops.float32, device=device)
                     if charged else None)

        #: LONG RANGE INHIBITION. PROVENANCE, checked against the clones and
        #: NOT what I first wrote here: neither reference implements a
        #: refractory period. In `mdabagia-nemo/brain.py` `inhibit()` is a
        #: RESET (`clear_input(); activations = []`), and in
        #: `dmitropolsky-assemblies` "inhibit" is AREA GATING for control flow.
        #: Grepping both clones for refractory/steps_ago/recently-fired returns
        #: nothing. So LRI-with-decay is THIS REPOSITORY'S OWN construction,
        #: the same status as `ordered_recall` and `sequence_memorize`, and
        #: porting it is not parity work. It is kept because RECOVERY is the
        #: one ingredient the substrate lacks -- without it there is no decay
        #: term anywhere, so idle spacing is a no-op by construction and the
        #: refraction bias is not a relative refractory period in the
        #: biological sense.
        #:
        #: It is NOT the refraction bias with a decay bolted on -- the two are
        #: orthogonal and compose:
        #:
        #:   refraction   charges per WIN, in proportion to the raw drive,
        #:                never recovers, and is gated on plasticity
        #:   LRI          charges per STEP at a FIXED strength, recovers
        #:                linearly over `period` steps, and applies on frozen
        #:                reads too, exactly as the reference does
        #:
        #: A ring buffer of the last `period` winner sets; the penalty a neuron
        #: carries is `strength * (1 - (age - 1) / period)` summed over every
        #: slot it appears in, subtracted from drive before k-WTA.
        self.lri_period, self.lri_strength = 0, 0.0
        self._lri_hist = None
        self._lri_pos, self._lri_filled = 0, 0
        #: MASKED READOUT, the engine's `masked_readout`: a frozen projection
        #: ranks the raw drive. `project(mask_bias=...)` overrides per call.
        self.masked_readout = False
        #: A refraction that RECOVERS: at the start of every non-frozen round
        #: the bias is multiplied by this factor (exp(-1 / tau) for a recovery
        #: time of tau rounds). None -- the reference's cumulative bias, which
        #: never decays -- leaves every operation untouched.
        self.bias_decay = None
        #: TIE JITTER, opt-in. The selector breaks exact ties by smallest
        #: index -- canonical, and what every capacity result was measured
        #: with. But a STIMULUS-driven area under norm_init has a drive with a
        #: handful of distinct levels, so the k-WTA bar sits inside a tie
        #: class ([[KWTA-TIE-FRAGILE]]): at n=1000 k=50 p=0.05 the numpy
        #: engine's stimulus base leaves 41 neurons above the bar and 959 tied
        #: at zero, so 9 of 50 winners are tie-fill. A canonical rule then
        #: gives EVERY input the same low-index tie-fill, correlating assemblies
        #: that should be independent -- measured on the aligner as word
        #: assemblies that overlap across words and alignment that WORSENS with
        #: training. With `tie_jitter > 0` the ranking adds a deterministic
        #: per-(input pattern, column) offset below any real drive gap, so ties
        #: break pseudo-randomly, differently for different inputs, and
        #: reproducibly. The drive itself is untouched (parity replays still
        #: compare the true drive); only the ORDER among exact ties changes.
        self.tie_jitter = float(tie_jitter or 0.0)
        self._cols = torch_ops.arange(n, dtype=torch_ops.int64, device=device)
        self._no_f32 = torch_ops.zeros(0, dtype=torch_ops.float32, device=device)
        #: the largest deferred k-WTA overflow not yet checked, on the device
        self._overflow = torch_ops.zeros((), dtype=torch_ops.int32, device=device)

    def _jitter(self, fibers):
        """[B, n] offsets in [0, tie_jitter), keyed by the active fibers.

        Cached per fiber combination: the value depends only on WHICH fibers
        fire, and recomputing it was 24% of an aligner run (six launches per
        round for a constant).
        """
        key = tuple(sorted(id(f) for f in fibers))
        cache = self.__dict__.setdefault("_jitter_cache", {})
        hit = cache.get(key)
        if hit is not None:
            return hit
        salt = torch_ops.zeros(self.B, dtype=torch_ops.int64, device=self.device)
        for f in fibers:
            salt = salt ^ (f.seeds.to(torch_ops.int64) & 0xFFFFFFFF)
        h = (self._cols.view(1, -1) ^ salt.view(-1, 1)) * 0x9E3779B1
        h = (h ^ (h >> 15)) * 0x85EBCA6B
        h = (h ^ (h >> 13)) & 0xFFFFFFFF
        out = h.to(torch_ops.float32) * (self.tie_jitter / 4294967296.0)
        cache[key] = out
        return out

    def apply_bias(self, raw):
        """Net drive the k-WTA ranks: raw minus the accumulated bias."""
        return raw if self.bias is None else raw - self.bias

    def set_lri(self, refractory_period: int, inhibition_strength: float) -> None:
        """Configure long range inhibition, or switch it off with a zero.

        Mirrors `Brain.set_lri`, including its validation, so the two paths
        cannot drift in what they accept.
        """
        from .._homeostasis import validate_lri_parameters
        period, strength = validate_lri_parameters(refractory_period,
                                                   inhibition_strength)
        self.lri_period, self.lri_strength = period, strength
        self._lri_pos, self._lri_filled = 0, 0
        self._lri_hist = (
            torch_ops.zeros(self.B, period, self.k, dtype=torch_ops.int64,
                            device=self.device)
            if period > 0 and strength > 0 else None)

    def apply_lri(self, drive):
        """Subtract the linearly decaying penalty of recently fired neurons.

        Age 1 is the most recent step and pays the full strength; age `period`
        pays `1 / period` of it; older is forgotten entirely. A neuron that
        fired on several of those steps pays for EACH, which is what the
        reference's per-slot loop does.
        """
        if self._lri_hist is None or self._lri_filled == 0:
            return drive
        penalty = torch_ops.zeros_like(drive)
        for age in range(1, self._lri_filled + 1):
            slot = (self._lri_pos - age) % self.lri_period
            decay = 1.0 - (age - 1) / self.lri_period
            penalty.scatter_add_(
                1, self._lri_hist[:, slot],
                torch_ops.full_like(penalty[:, :self.k],
                                    self.lri_strength * decay))
        return drive - penalty

    def prime_lri(self, winners) -> None:
        """Enter ``winners`` [B, k] into the long-range-inhibition history as if
        they had just fired. A recall CUE is placed, not fired, so it is not in
        the history unless primed; priming the state a replay CAME FROM vetoes
        it, which sets the direction of travel on a chain linked both ways."""
        if self._lri_hist is None:
            raise ValueError("set_lri before priming its history")
        self._push_lri(winners)

    def _push_lri(self, winners) -> None:
        """Record this step's winners. Not gated on `freeze`: the reference
        appends its history on every step, so a frozen read still inhibits."""
        if self._lri_hist is None:
            return
        self._lri_hist[:, self._lri_pos] = winners
        self._lri_pos = (self._lri_pos + 1) % self.lri_period
        self._lri_filled = min(self._lri_filled + 1, self.lri_period)

    def charge(self, raw, new):
        """Charge the winners `raw * strength`, as `refraction_increment` does
        (it reconstructs raw as net + bias; here raw is still at hand)."""
        if self.bias is None:
            return
        self.bias.scatter_add_(
            1, new, torch_ops.gather(raw, 1, new) * self.refracted_strength)

    def _charge_and_record(self, raw, sel):
        """``charge(raw, winners)`` and the ever-fired record in one pass
        (the fused `charge` kernel: bias += raw * strength at the winners,
        then ever = True), the same float operations as the two steps."""
        strength = self.refracted_strength
        per_brain = isinstance(strength, torch_ops.Tensor)
        if self.bias is not None and not self.bias.is_contiguous():
            self.bias = self.bias.contiguous()
        self.mod.charge(self.bias if self.bias is not None else self._no_f32,
                        raw.contiguous(), sel,
                        strength.view(-1) if per_brain else self._no_f32,
                        0.0 if per_brain else float(strength), self.ever)

    def inhibit(self):
        """Clear the assembly. The next round is driven by afferents alone."""
        self.winners = torch_ops.zeros(self.B, 0, dtype=torch_ops.int64,
                                   device=self.device)

    def inhibit_rows(self, mask):
        """Clear the assembly of the brains in `mask` [B] only: their winner
        rows become -1, which every fiber reads as "no source" -- the
        per-brain sentence boundary of a scheduled organ."""
        if self.winners.shape[1] == 0:
            return
        mask = torch_ops.as_tensor(mask, dtype=torch_ops.bool, device=self.device)
        self.winners = self.winners.masked_fill(mask.view(-1, 1), -1)

    def check_overflow(self):
        """Raise if any deferred projection's k-WTA overflowed (one host
        read). Call it before anything reads the winners as a result."""
        bad = int(self._overflow)
        if bad:
            self._overflow.zero_()
            _raise_overflow(bad)

    def select(self, keep):
        """Keep only the brains ``keep`` [B'] (indices), in that order. The
        per-brain state goes with them; the round counters are shared."""
        keep = torch_ops.as_tensor(keep, dtype=torch_ops.int64, device=self.device)
        self.winners = self.winners.index_select(0, keep)
        self.ever = self.ever.index_select(0, keep)
        if self.bias is not None:
            self.bias = self.bias.index_select(0, keep)
        if isinstance(self.refracted_strength, torch_ops.Tensor):
            self.refracted_strength = self.refracted_strength.index_select(0, keep)
        if self._lri_hist is not None:
            self._lri_hist = self._lri_hist.index_select(0, keep)
        self.__dict__.pop("_jitter_cache", None)
        self.B = int(keep.numel())

    def project(self, rounds, fibers, *, rows_for=None, freeze=False,
                stim_drive=None, return_drive=False, mask_bias=None,
                manage_episodes=True, stop_when_stable=False, defer_overflow=False,
                observe=True, record=None, write=None):
        """Run ``rounds`` rounds with ``fibers`` afferent.

        ``rows_for`` maps a fiber to its source winners; a fiber absent from it
        is driven by THIS area's winners, i.e. recurrently.

        ``mask_bias`` reads the SYNAPTIC memory alone: the refraction bias is
        neither subtracted nor charged. Only meaningful with ``freeze`` -- it
        is a probe of what the synapses hold with the intrinsic veto removed
        (PREREG_refraction_capacity P1), not a mode the reference has.

        ``manage_episodes=False`` leaves the fibers' episodes to the caller,
        so several one-round calls can share ONE episode (one mask, one GEMM
        fold into the store) -- a training step whose rounds interleave two
        areas cannot be a single multi-round call here, and per-round episodes
        cost a store append and merge each. Observe/charge still happen.

        ``stop_when_stable`` gates the rounds PER BRAIN on convergence
        (PREREG_refraction_memory.md Amendment 5): a brain's item is over at
        the first round whose winner set equals the previous round's; that
        round is written like any other, and from then on the brain keeps
        its winners and writes nothing -- its rows and winners go to the
        fibers as -1, the dead-brain convention, so the fibers must accept
        -1 (the organ and stimulus fibers do; the store fibers do not). The
        loop ends early when no brain is active. ``rounds`` is then the
        CEILING T_max, and ``self.rounds_used`` [B] the rounds each brain
        spent. A brain's rounds up to its convergence are bit-identical to
        the ungated run's (tested).
        """
        rows_for = rows_for or {}
        if mask_bias is None:
            mask_bias = bool(self.masked_readout) and freeze
        if mask_bias and not freeze:
            raise ValueError("mask_bias is a READOUT option; pass freeze=True")
        if not freeze and manage_episodes:
            for f in fibers:
                f.begin_episode()
        drive = None
        active = None
        ovf_acc = None
        if stop_when_stable:
            active = torch_ops.ones(self.B, dtype=torch_ops.bool, device=self.device)
            self.rounds_used = torch_ops.zeros(self.B, dtype=torch_ops.int64,
                                           device=self.device)
        for _ in range(rounds):
            if self.bias_decay is not None and not freeze and self.bias is not None:
                self.bias.mul_(self.bias_decay)
            raw = torch_ops.zeros(self.B, self.n, dtype=torch_ops.float32,
                              device=self.device)
            for f in fibers:
                f.contribute(raw, rows_for.get(id(f), self.winners))
            if stim_drive is not None:
                raw = raw + stim_drive
            drive = raw if mask_bias else self.apply_bias(raw)
            # LRI is not the refraction bias and `mask_bias` does not lift it:
            # that option probes what the SYNAPSES hold with the intrinsic veto
            # removed, and long range inhibition is extrinsic.
            drive = self.apply_lri(drive)
            ranked = (drive + self._jitter(fibers) if self.tie_jitter > 0
                      else drive)
            sel, ovf = self.mod.topk_select(ranked, min(self.k, self.n))
            # the overflow flag is accumulated on the device and read ONCE
            # after the rounds: a host sync per round was most of a step
            ovf_acc = ovf if ovf_acc is None else torch_ops.maximum(ovf_acc, ovf)
            new = sel.to(torch_ops.int64)
            prev = self.winners
            if active is not None and prev.shape[1] == new.shape[1]:
                # a converged brain keeps its winners
                new = torch_ops.where(active.view(-1, 1), new, prev)
            if record is not None:
                record.append(new)
            if not freeze:
                if active is None:
                    # observe=False: the rounds run (refraction charged, the
                    # ever-fired record kept) but no fiber writes; a caller that
                    # writes once per item (the burst write) does so after.
                    # `write(prev, new)` returns the (pre, post) rows this
                    # round writes -- a rule that gates who may be written
                    if observe:
                        src, dst = (prev, new) if write is None else write(prev, new)
                        for f in fibers:
                            f.observe(rows_for.get(id(f), src), dst)
                    self._charge_and_record(raw, sel)
                else:
                    off = ~active.view(-1, 1)
                    new_m = new.masked_fill(off, -1)
                    for f in fibers:
                        src = rows_for.get(id(f), prev)
                        f.observe(src.masked_fill(off, -1) if src.shape[1] else src,
                                  new_m)
                    if self.bias is not None:
                        self.bias.scatter_add_(
                            1, new, torch_ops.gather(raw, 1, new)
                            * (self.refracted_strength * active.view(-1, 1)))
                    self.rounds_used += active.to(torch_ops.int64)
                if active is not None:
                    self.ever.scatter_(1, new, True)
                self.rounds_seen += 1
            self.winners = new
            self._push_lri(new)
            if active is not None:
                if prev.shape[1] == new.shape[1]:
                    same = (torch_ops.sort(new, dim=1).values
                            == torch_ops.sort(prev, dim=1).values).all(dim=1)
                    active = active & ~same
                if not bool(active.any()):
                    break
        if ovf_acc is not None:
            if defer_overflow:
                # kept on the device until `check_overflow`: reading it here is
                # a host sync per call, which stalls a store loop on every item.
                # Updated IN PLACE, so a replayed CUDA graph accumulates into it.
                torch_ops.maximum(self._overflow, ovf_acc.max(), out=self._overflow)
            else:
                _raise_overflow(int(ovf_acc.max()))
        if not freeze and manage_episodes:
            for f in fibers:
                f.end_episode()
        return (self.winners, drive) if return_drive else self.winners

    @property
    def fill(self):
        """``rows/n``: the fraction of the area that has ever fired."""
        return self.ever.sum(dim=1).float() / self.n
