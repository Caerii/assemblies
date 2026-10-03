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
alone. ``beta`` may be one learning rate per brain, so a learning-rate sweep
is ONE launch (each brain carries its own chain table and 0.5 beta charge);
``recall_many`` reads many cues per brain in one pass; ``select`` drops the
brains a sweep has finished with (DESIGN_memory_throughput.md).
"""
from __future__ import annotations

import math

from ._torch_ops import torch_ops
from typing import Any, cast

from ._hashed import AreaFiber, DenseOrganFiber, HashedArea, StimulusFiber, per_brain

#: bytes of float32 drive a batched recall holds per pass (two [V, n] buffers)
RECALL_BYTES = 1 << 28


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
                 max_items=4096, device="cuda", organ_semantics=None, graphs=False):
        from ..semantics import OrganSemantics, describe_assembly_memory

        self.seeds = [int(s) for s in seeds]
        self.B, self.n, self.k, self.p = len(self.seeds), int(n), int(k), float(p)
        betas = per_brain(beta, self.B)

        def describe(b):
            return describe_assembly_memory(
                w_max=w_max, norm_init=norm_init,
                synaptic_scaling=synaptic_scaling, strength=strength, beta=b,
                gate=gate,
            )
        if betas is None:
            actual_semantics = describe(beta)
            if organ_semantics is not None:
                required = OrganSemantics.normalize(organ_semantics)
                mismatch = required.mismatch(actual_semantics)
                if mismatch:
                    raise ValueError(f"organ_semantics mismatch: {mismatch}")
        else:
            # a swept memory: one description per rate, each checked against
            # the profile the caller registered for that rate
            actual_semantics = {b: describe(b) for b in sorted(set(betas))}
            if organ_semantics is not None:
                for b, actual in actual_semantics.items():
                    required = OrganSemantics.normalize(organ_semantics[b])
                    mismatch = required.mismatch(actual)
                    if mismatch:
                        raise ValueError(f"organ_semantics mismatch at beta={b:g}: {mismatch}")
        self.organ_semantics = actual_semantics
        #: a float, or a tuple of one learning rate per brain
        self.beta = float(beta) if betas is None else betas
        self.w_max = w_max
        self.norm_init, self.scaling = bool(norm_init), bool(synaptic_scaling)
        self.rounds, self.strength, self.gate = int(rounds), float(strength), bool(gate)
        self.device = device
        if self.gate and self.scaling:
            raise ValueError("the convergence gate needs the organ fiber "
                             "(-1 rows); column scaling takes the store fiber")
        charge = (self.strength * self.beta if betas is None
                  else [self.strength * b for b in betas])
        self.area = HashedArea(self.n, self.k, self.seeds, device=device,
                               refracted_strength=charge)
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
        #: replay each item's write as one CUDA graph (ungated stores only)
        self.graphs = bool(graphs) and not self.gate
        self._graph = None
        self._warm = None

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
        if self.graphs:
            return self._store_graphed(stim_seeds, size)
        return self._store(stim_seeds, size)

    def _store(self, stim_seeds, size):
        stim = StimulusFiber(stim_seeds, size, self.n, self.p, beta=self.beta,
                             w_max=self.w_max, norm_init=self.norm_init,
                             max_rounds=self.rounds, device=self.device)
        self.area.inhibit()
        # the overflow check waits for `check` (or the next read): a host
        # read per item stalled the store loop on every item
        win = self.area.project(self.rounds, [self.fiber, stim],
                                stop_when_stable=self.gate, defer_overflow=True)
        self._last_used = (self.area.rounds_used.clone() if self.gate else None)
        self.items += 1
        return win

    def _store_graphed(self, stim_seeds, size):
        """``store`` as a replayed CUDA graph (DESIGN_memory_throughput.md).

        An ungated write is the same kernels on the same shapes every item,
        about 17 launches a round; captured once, an item is one replay. The
        first item at a shape is written eagerly (it builds the cached tables
        the capture must not copy from the host); the second is captured and
        every later one replays it, with the stimulus seeds copied into the
        graph's input. Everything the write mutates -- counts, bias, the
        ever-fired record, the deferred overflow -- is updated in place, so a
        replay is the eager write (tested bit for bit)."""
        import torch
        seeds = torch.as_tensor(stim_seeds, dtype=torch.int32, device=self.device)
        key = (self.B, size)
        if self._graph is None or self._graph[0] != key:
            if self._warm != key:
                self._warm = key
                return self._store(seeds, size)
            static = seeds.clone()
            graph = torch.cuda.CUDAGraph()
            items, seen = self.items, self.area.rounds_seen
            with torch.cuda.graph(graph):
                out = self._store(static, size)
            # capture records the write without running it
            self.items, self.area.rounds_seen = items, seen
            self._graph = (key, graph, static, out)
        _, graph, static, out = self._graph
        static.copy_(seeds)
        graph.replay()
        self.items += 1
        self.area.rounds_seen += self.rounds
        return out.clone()

    @property
    def rounds_used(self):
        """[B] rounds the last item spent per brain (gated), else None."""
        return self._last_used

    # -- the read -------------------------------------------------------------
    def recall(self, cue, *, masked=None, rounds=None):
        """Complete ``cue`` [B, m] (m <= k neurons of a stored assembly) by
        ``rounds`` frozen recurrent rounds (default: the write's) --
        ``probe()``: nothing is written and no bias is charged. ``masked``
        (default: whenever refracted) reads the synaptic memory with the
        bias zeroed; ``masked=False`` is the net readout, which reads chance
        on a refracted memory. ``rounds`` lets a study vary the WRITE's depth
        while holding the read's fixed."""
        if rounds is not None and (type(rounds) is not int or rounds < 1):
            raise ValueError("recall rounds must be a positive integer")
        self.area.check_overflow()
        self.area.winners = cue.to(torch_ops.int64)
        return self.area.project(self.rounds if rounds is None else rounds,
                                 [self.fiber], freeze=True,
                                 mask_bias=(None if masked is None
                                            else bool(masked) and self.refracted))

    def recall_many(self, cues, *, masked=None, rounds=None):
        """``recall`` of S cues per brain at once: ``cues`` [B, S, m] ->
        [B, S, k]. Each cue is a VIRTUAL brain reading its brain's synapses
        (the organ fiber's brain map), so a checkpoint's reads are one pass
        instead of S. The rounds are ``recall``'s: frozen, nothing written or
        charged, the bias masked whenever refracted; a cue's result equals
        ``recall`` of that cue alone."""
        if rounds is not None and (type(rounds) is not int or rounds < 1):
            raise ValueError("recall rounds must be a positive integer")
        B, S, m = cues.shape
        self.area.check_overflow()
        if not isinstance(self.fiber, DenseOrganFiber):
            return torch_ops.stack([self.recall(cues[:, s], masked=masked, rounds=rounds)
                                    for s in range(S)], dim=1)
        area = self.area
        if area._lri_hist is not None or area.tie_jitter > 0:
            raise ValueError("recall_many reads the memory's own area: no LRI, no jitter")
        mask = (bool(area.masked_readout) if masked is None
                else bool(masked) and self.refracted)
        rounds = self.rounds if rounds is None else rounds
        per = max(1, RECALL_BYTES // (8 * self.n * B))      # cues per pass
        out = []
        for s0 in range(0, S, per):
            part = cues[:, s0:s0 + per]
            P = part.shape[1]
            brains = torch_ops.arange(B, dtype=torch_ops.int32,
                                      device=cues.device).repeat_interleave(P)
            bias = (None if mask or area.bias is None
                    else area.bias.index_select(0, brains.to(torch_ops.int64)))
            winners = part.reshape(B * P, m).to(torch_ops.int64)
            ovf_acc = None
            for _ in range(rounds):
                raw = torch_ops.zeros(B * P, self.n, dtype=torch_ops.float32,
                                      device=cues.device)
                self.fiber.contribute(raw, winners, brains)
                drive = raw if bias is None else raw - bias
                sel, ovf = area.mod.topk_select(drive, min(self.k, self.n))
                ovf_acc = ovf if ovf_acc is None else torch_ops.maximum(ovf_acc, ovf)
                winners = sel.to(torch_ops.int64)
            bad = int(ovf_acc.max()) if ovf_acc is not None else 0
            if bad:
                raise RuntimeError(
                    f"k-WTA candidate set overflowed ({bad} candidates) -- the "
                    "drive is too flat for the histogram to narrow. Refusing "
                    "to return a truncated winner set.")
            out.append(winners.view(B, P, -1))
        return torch_ops.cat(out, dim=1)

    def select(self, keep):
        """Keep only the brains ``keep`` (indices into the current brains), in
        that order: a swept store drops the rates it has finished with, so the
        launch does only the work still wanted."""
        keep = [int(i) for i in keep]
        self.area.select(keep)
        select = getattr(cast(Any, self.fiber), "select", None)
        if select is None:
            raise ValueError("only the organ fiber can drop brains")
        select(keep)
        self.seeds = [self.seeds[i] for i in keep]
        if isinstance(self.beta, tuple):
            self.beta = tuple(self.beta[i] for i in keep)
        self.B = len(keep)
        self._last_used = None
        self._graph = self._warm = None

    # -- state ------------------------------------------------------------------
    @property
    def fill(self):
        """[B] fraction of the area that has ever fired."""
        return self.area.fill

    @property
    def bias(self):
        return self.area.bias

    def check(self):
        self.area.check_overflow()
        check = getattr(cast(Any, self.fiber), "check", None)
        if check is not None:
            check()
