"""The STIMULUS fiber: pre-summed input, one weight per target neuron.

Moved from _hashed.py unchanged; _hashed.py re-exports every name."""
from __future__ import annotations

from ._torch_ops import torch_ops
from typing import Any
from . import _fused_cuda
from ._hashed_common import per_brain, _device_gain


class StimulusFiber:
    """A stimulus -> area projection. PRE-SUMMED: one weight per target.

    TWO PRICES THAT ARE EASY TO GET WRONG, both taken from the engine:

    * ``norm_init`` divides by ``d_j = deg_j + p * (tgt.n - stim_size)`` --
      note ``tgt.n``, NOT the stimulus size. That puts the stimulus on the same
      ``~n p`` divisor as the area fiber so the two are commensurable;
      dividing by its own in-degree would make an untrained stimulus
      contribute exactly 1.0 per touched neuron against an area contribution
      of ~0.06, and the stimulus would decide every winner.
    * ``w_max`` means "multiples of the INITIAL weight", and a stimulus weight
      starts near ``stim_size * p``, not at 1. The cap is
      ``w_max * max(1, stim_size * p)``.
    """

    mod: Any
    base: Any
    pot: Any
    gain: Any
    dj: Any
    _const: Any

    def __init__(self, seeds, size, n_post, p, *, beta=0.0, w_max=None,
                 norm_init=False, max_rounds=64, device="cuda",
                 zero_or_size=False):
        self.mod = _fused_cuda.load()
        B = len(seeds)
        self.B, self.size, self.n, self.p = B, size, n_post, float(p)
        self.seeds = torch_ops.as_tensor(seeds, dtype=torch_ops.int32, device=device)
        self.threshold = _fused_cuda.threshold_for(p)
        betas = per_brain(beta, B)
        self.learns = bool(beta) if betas is None else any(betas)
        if zero_or_size:
            # THE ENGINE'S STIMULUS MODEL ([[add-stimulus-zero-or-size]]): a
            # neuron's weight from a stimulus is `size` with probability p
            # and 0 otherwise -- one Bernoulli draw, not a Binomial(size, p)
            # count. A word's assembly is then the connected set, driven at
            # `size` against fiber drives of ~k p w: the sharpness the organ's
            # numbers rest on (materialized numpy 0.17-0.22 MRR on A3 against
            # 0.12 with Binomial stimuli). The aligner's anchor gain 1/p is
            # the same fact approximated by a scalar.
            one = torch_ops.zeros(B, 1, dtype=torch_ops.int32, device=device)
            self.base = self.mod.hashed_drive(one, self.seeds, n_post,
                                              self.threshold) * float(size)
        else:
            rows = torch_ops.arange(size, dtype=torch_ops.int32,
                                device=device).expand(B, size).contiguous()
            self.base = self.mod.hashed_drive(rows, self.seeds, n_post,
                                              self.threshold)
        # an ANCHOR (beta = 0) never potentiates: no counter, no table -- with
        # a fiber per word, the int64 counter was two thirds of the memory
        self.pot = (torch_ops.zeros(B, n_post, dtype=torch_ops.int64, device=device)
                    if self.learns else None)
        self.gain = _device_gain(float(beta) if betas is None else betas,
                                 int(max_rounds), str(device))
        self.dj = ((self.base + self.p * (n_post - size)).clamp_min(1.0)
                   if norm_init else None)
        self.hi = (w_max * max(1.0, size * self.p)
                   if w_max is not None else float("inf"))
        self._no_dj = torch_ops.zeros(0, dtype=torch_ops.float32, device=device)

    #: DRIVE GAIN: a multiplier on this stimulus's contribution, 1.0 by
    #: default. It is the ANCHOR SHARE knob ([[capacity-is-an-anchor-ratio]],
    #: [[semantic-drive-share-is-the-lever]]): what an area forms as is decided
    #: by how much of its drive the stimulus supplies against learned fibers.
    #: The numpy engine's stimulus into a materialized area is 0-or-size, i.e.
    #: a full-size weight where this fiber generates a Binomial(size, p) count
    #: -- an accidental gain of ~1/p. Set explicitly rather than inherited.
    drive_gain = 1.0

    def contribute(self, drive, rows=None):
        del rows  # Anchors use fixed hashed rows; shared hook carries rows for learned fibers.
        if not self.learns:
            # An anchor never potentiates, so its priced drive is a constant:
            # one add per round instead of six launches (15% of a run).
            const = self.__dict__.get("_const")
            if const is None or const[0] != (self.drive_gain, id(self.base),
                                             id(self.dj)):
                d = self.base.clone()
                if self.hi != float("inf"):
                    d = d.clamp_max(self.hi)
                if self.dj is not None:
                    d = d / self.dj
                if self.drive_gain != 1.0:
                    d = d * self.drive_gain
                const = ((self.drive_gain, id(self.base), id(self.dj)), d)
                self._const = const
            drive += const[1]
            return
        if self.base.is_cuda and drive.is_contiguous():
            # one fused pass, the same float operations as the chain below
            self.mod.stim_add(drive, self.base, self.gain, self.pot,
                              float(self.hi), self.dj if self.dj is not None else self._no_dj,
                              float(self.drive_gain))
            return
        top = self.gain.shape[-1] - 1
        d = self.base * (self.gain[self.pot.clamp_max(top)] if self.gain.dim() == 1
                         else torch_ops.gather(self.gain, 1, self.pot.clamp_max(top)))
        if self.hi != float("inf"):
            d = d.clamp_max(self.hi)
        if self.dj is not None:
            d = d / self.dj
        if self.drive_gain != 1.0:
            d = d * self.drive_gain
        drive += d

    def begin_episode(self):
        pass

    def observe(self, prev, new):
        del prev  # Anchor potentiation depends only on newly selected winners.
        if self.learns and new.shape[1]:
            # -1 winners are a brain whose rounds are over (a converged
            # brain under `stop_when_stable`, a dead brain): no potentiation
            self.pot.scatter_add_(1, new.clamp_min(0), (new >= 0).to(torch_ops.int64))

    def end_episode(self):
        pass
