"""AssemblyMemory: a recurrent k-WTA area as an associative memory.

The unit behind ``REFRACTION-ANTI-MERGING`` (theory.py) and
``PREREG_refraction_memory.md``, named once. Four things make the memory,
and none of them is the memory alone:

* the AREA, a k-WTA over ``n`` neurons with a Hebbian recurrent fiber in
  the organ's regime (weight clip, ``norm_init``, no column scaling --
  :class:`~._hashed.DenseOrganFiber`), or the store fiber with column
  scaling for the other arm;
* REFRACTION on the area, a per-neuron bias charged at the winners
  (``strength`` in multiples of ``beta``; 0.5 is the middle of a measured
  plateau 0.3-0.6, and 0 is the Hebbian control);
* the WRITE, ``rounds`` of stimulus + recurrence from an inhibited area per
  item, optionally GATED per brain on convergence (an item ends at its first
  repeated winner set; the ceiling moves +24-34%);
* the READ, a half-cue recall with the bias MASKED: the refracted memory is
  read through the veto or not at all (the net readout reads chance).

Measured (20 brains, arm B): in regime (k p >= 3 ln n) the refracted
ceiling is ~0.40 (n/k)^2 stored assemblies, ~25x the Hebbian control's,
the items distinct (1.000) where the control collapses into hubs. See the
register entry for the caveats; the class does not enforce the regime, it
reports it (:attr:`in_regime`).

Every brain in a launch is independent (its own seeds); the batch is the
GPU lever, and a brain's trajectory in a launch of B equals its trajectory
alone.
"""
from __future__ import annotations

import math

import torch

from ._hashed import AreaFiber, DenseOrganFiber, HashedArea, StimulusFiber


def recurrent_fiber(seeds, n, p, *, beta, w_max, norm_init, synaptic_scaling,
                    max_rounds, device):
    """THE fiber for a recurrent area, chosen once for every caller.

    The organ's regime (clip, no scaling) takes the count-matrix fiber:
    O(1) per round in stored episodes, where the store fiber grows with them
    (a 1024-item grid stalled it at 10 GB); drive parity is tolerance-based.
    Storage selection also selects normalization arithmetic; see
    neural_assemblies/ir/VERIFICATION.md#contract-hashed-normalization.
    Close drives do not guarantee identical winner trajectories.
    Column scaling takes the store fiber (the four-arm parity test licenses
    the scaling + clip opt-in there).
    """
    if not synaptic_scaling and w_max is not None:
        return DenseOrganFiber(seeds, n, n, p, beta=beta, w_max=w_max,
                               norm_init=norm_init,
                               max_rounds=min(int(max_rounds), 256),
                               device=device)
    return AreaFiber(seeds, n, n, p, beta=beta, w_max=w_max,
                     norm_init=norm_init, synaptic_scaling=synaptic_scaling,
                     max_rounds=int(max_rounds), device=device,
                     scaling_allows_clip=True)


class AssemblyMemory:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-memory"""

    def __init__(self, seeds, n, k, p, *, beta=0.1, w_max=20.0, norm_init=True,
                 synaptic_scaling=False, rounds=8, strength=0.5, gate=False,
                 max_items=4096, device="cuda", organ_semantics=None):
        from ..semantics import OrganSemantics, describe_assembly_memory

        actual_semantics = describe_assembly_memory(
            w_max=w_max, norm_init=norm_init,
            synaptic_scaling=synaptic_scaling, strength=strength, beta=beta,
            gate=gate,
        )
        if organ_semantics is not None:
            required = OrganSemantics.normalize(organ_semantics)
            mismatch = required.mismatch(actual_semantics)
            if mismatch:
                raise ValueError(f"organ_semantics mismatch: {mismatch}")
        self.organ_semantics = actual_semantics
        self.seeds = [int(s) for s in seeds]
        self.B, self.n, self.k, self.p = len(self.seeds), int(n), int(k), float(p)
        self.beta, self.w_max = float(beta), w_max
        self.norm_init, self.scaling = bool(norm_init), bool(synaptic_scaling)
        self.rounds, self.strength, self.gate = int(rounds), float(strength), bool(gate)
        self.device = device
        if self.gate and self.scaling:
            raise ValueError("the convergence gate needs the organ fiber "
                             "(-1 rows); column scaling takes the store fiber")
        self.area = HashedArea(self.n, self.k, self.seeds, device=device,
                               refracted_strength=self.strength * self.beta)
        # the memory's reads are masked by default: the store is read through
        # the veto or not at all
        self.area.masked_readout = self.strength > 0
        self.fiber = recurrent_fiber(self.seeds, self.n, self.p, beta=self.beta,
                                     w_max=w_max, norm_init=norm_init,
                                     synaptic_scaling=synaptic_scaling,
                                     max_rounds=int(max_items) * self.rounds,
                                     device=device)
        self.items = 0
        self._last_used = None

    # -- the regime -----------------------------------------------------------
    @property
    def in_regime(self):
        """``k p >= 3 ln n``: the recurrent in-degree the n/k law needs.
        Out of it the cells sit 20-32% below their n/k pairs and do not
        converge at low load."""
        return self.k * self.p >= 3.0 * math.log(self.n)

    @property
    def refracted(self):
        return self.strength > 0

    # -- the write ------------------------------------------------------------
    def store(self, stim_seeds, stim_size=None):
        """Write one item per brain: its stimulus (hash-generated from
        ``stim_seeds`` [B], ``stim_size`` neurons, default k) fires every
        round alongside recurrence from an INHIBITED area for ``rounds``
        rounds -- ``inhibit_areas([A]); project({s: [A]}, {A: [A]})`` x T.
        Gated, a brain's item ends at its first repeated winner set.
        Returns the stored assembly [B, k]."""
        size = self.k if stim_size is None else int(stim_size)
        stim = StimulusFiber(stim_seeds, size, self.n, self.p, beta=self.beta,
                             w_max=self.w_max, norm_init=self.norm_init,
                             max_rounds=self.rounds, device=self.device)
        self.area.inhibit()
        win = self.area.project(self.rounds, [self.fiber, stim],
                                stop_when_stable=self.gate)
        self._last_used = (self.area.rounds_used.clone() if self.gate else None)
        self.items += 1
        return win

    @property
    def rounds_used(self):
        """[B] rounds the last item spent per brain (gated), else None."""
        return self._last_used

    # -- the read -------------------------------------------------------------
    def recall(self, cue, *, masked=None):
        """Complete ``cue`` [B, m] (m <= k neurons of a stored assembly) by
        ``rounds`` frozen recurrent rounds -- ``probe()``: nothing is written
        and no bias is charged. ``masked`` (default: whenever refracted)
        reads the synaptic memory with the bias zeroed; ``masked=False`` is
        the net readout, which reads chance on a refracted memory."""
        self.area.winners = cue.to(torch.int64)
        return self.area.project(self.rounds, [self.fiber], freeze=True,
                                 mask_bias=(None if masked is None
                                            else bool(masked) and self.refracted))

    # -- state ------------------------------------------------------------------
    @property
    def fill(self):
        """[B] fraction of the area that has ever fired."""
        return self.area.fill

    @property
    def bias(self):
        return self.area.bias

    def check(self):
        if hasattr(self.fiber, "check"):
            self.fiber.check()
