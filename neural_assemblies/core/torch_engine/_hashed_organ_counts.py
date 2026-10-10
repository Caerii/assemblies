"""Reading and editing the dense organ fiber's counts, in either layout.

`DenseOrganFiber` keeps its counts as a plain matrix ``C`` [B, n_pre, n] (int8 or int16), or --
opt-in, ``count_dtype="int4"`` -- PACKED, two counts a byte, in ``Cp`` [B, n_pre, NB] with
NB = ceil(n / 4) * 2 (column j in byte j >> 1, the low nibble holding even j). A packed fiber's
``C`` is None, so code that reads ``C`` as counts fails instead of misreading nibbles; this mixin
is how anything outside the kernels reads or edits counts, the same call for either layout:

    unpacked()          the counts as an int16 copy
    count_sum()         the total count held
    nonzero()           how many synapses hold a count
    clamp_counts(top)   cap every count (at the clip it changes no weight)
    decrement(...)      one count off chosen synapses (sleep's unlearning)
    downscale(...)      one count off each potentiated synapse with a probability

Every method is exact for both layouts: on the same counts, the same edit leaves the same counts.
"""
from __future__ import annotations

from typing import Any

from ._torch_ops import torch_ops


class OrganCounts:
    """Mixed into DenseOrganFiber, which sets these."""

    packed: bool          # True for count_dtype "int4"
    C: Any                # [B, n_pre, n] int8 / int16 counts, or None when packed
    Cp: Any               # [B, n_pre, NB] packed bytes, or None
    NB: int               # bytes per packed row
    n_pre: int
    n: int
    device: Any

    @property
    def storage(self):
        """the bytes the kernels read: ``C`` [B, n_pre, n], or packed ``Cp`` [B, n_pre, NB]"""
        return self.Cp if self.packed else self.C

    def _unpack(self, P):
        """packed bytes [..., NB] -> int16 counts [..., n]"""
        lo = (P & 15).to(torch_ops.int16)
        hi = (P >> 4).to(torch_ops.int16)
        return torch_ops.stack([lo, hi], dim=-1).reshape(*P.shape[:-1], 2 * self.NB)[..., :self.n]

    def _pack(self, counts):
        """counts [..., n] in 0..15 -> packed bytes [..., NB]"""
        pad = 2 * self.NB - self.n
        if pad:
            counts = torch_ops.cat([counts, torch_ops.zeros(*counts.shape[:-1], pad, dtype=counts.dtype,
                                                            device=counts.device)], dim=-1)
        c = counts.to(torch_ops.uint8).reshape(*counts.shape[:-1], self.NB, 2)
        return c[..., 0] | (c[..., 1] << 4)

    def unpacked(self):
        """the counts as int16 [B, n_pre, n] (a copy, either layout)"""
        return self._unpack(self.Cp) if self.packed else self.C.to(torch_ops.int16)

    def count_sum(self):
        """the total count held (int)"""
        if not self.packed:
            return int(self.C.sum(dtype=torch_ops.int64))
        return int((self.Cp & 15).sum(dtype=torch_ops.int64) + (self.Cp >> 4).sum(dtype=torch_ops.int64))

    def nonzero(self):
        """how many synapses hold a count (int)"""
        if not self.packed:
            return int((self.C > 0).sum())
        return int(((self.Cp & 15) > 0).sum() + ((self.Cp >> 4) > 0).sum())

    def clamp_counts(self, top):
        """cap every count at ``top``, in place (at the clip it changes no weight)"""
        if not self.packed:
            self.C.clamp_(max=int(top))
            return
        t = min(int(top), 15)
        lo = (self.Cp & 15).clamp_(max=t)
        hi = (self.Cp >> 4).clamp_(max=t)
        self.Cp.copy_(lo | (hi << 4))

    def decrement(self, b, i, j, gate):
        """One count off every synapse (b, i, j) holding a positive count, in brains whose
        ``gate`` [B] (bool) is on. b [B, 1, 1], i [B, k, 1], j [B, 1, k] broadcast to the pairs;
        the (i, j) pairs of a brain must be distinct. Returns the counts removed (a device
        scalar). The int8/int16 path is memory_sleep.dream's own indexing."""
        if not self.packed:
            old = self.C[b, i, j]
            dec = ((old > 0) & gate.view(-1, 1, 1)).to(old.dtype)
            self.C[b, i, j] = old - dec
            return dec.sum(dtype=torch_ops.int64)
        bb, ii, jj = torch_ops.broadcast_tensors(b, i, j)
        byte = ((bb * self.n_pre + ii) * self.NB + (jj >> 1)).reshape(-1)
        shift = ((jj & 1) * 4).reshape(-1).to(torch_ops.int32)
        flat = self.Cp.view(-1)
        val = (flat[byte].to(torch_ops.int32) >> shift) & 15
        g = gate.view(-1, 1, 1).expand(bb.shape).reshape(-1)
        dec = ((val > 0) & g).to(torch_ops.int32)
        # Columns j and j ^ 1 of a row share a byte, so two decrements may land on one byte.
        # Each adds its two's complement, (256 - d * 2^shift) mod 256, atomically: the byte ends
        # at byte - d_lo - 16 d_hi, exactly, with no borrow across a nibble (a nibble is
        # decremented only from a positive count). No data-dependent shapes and no host sync,
        # so a recorded CUDA graph (the fast sleep) can capture it.
        flat.index_add_(0, byte, ((256 - (dec << shift)) & 255).to(torch_ops.uint8))
        return dec.sum(dtype=torch_ops.int64)

    def downscale(self, rates, gen, chunk=1024):
        """Each potentiated synapse of brain b loses one count with probability ``rates[b]`` (a
        float for every brain, or one per brain). Per brain, the uniforms are one
        torch.rand(n_pre, n) from ``gen`` -- the draws of a dense loop over brains -- compared in
        row chunks, so the transient is [chunk, n] rather than [n_pre, n]. Returns the counts
        removed (int)."""
        B = self.storage.shape[0]
        rates = ([float(rates)] * B if not hasattr(rates, "__len__")
                 else [float(r) for r in rates])
        removed = torch_ops.zeros((), dtype=torch_ops.int64, device=self.device)
        u = torch_ops.empty(self.n_pre, self.n, dtype=torch_ops.float32, device=self.device)
        for b in range(B):
            torch_ops.rand(self.n_pre, self.n, generator=gen, device=self.device, out=u)
            if rates[b] <= 0.0:
                continue
            for r0 in range(0, self.n_pre, chunk):
                r1 = min(self.n_pre, r0 + chunk)
                rows = self._unpack(self.Cp[b, r0:r1]) if self.packed else self.C[b, r0:r1]
                cut = (rows > 0) & (u[r0:r1] < rates[b])
                if self.packed:
                    self.Cp[b, r0:r1] = self._pack(rows - cut.to(rows.dtype))
                else:
                    rows -= cut.to(rows.dtype)
                removed += cut.sum(dtype=torch_ops.int64)
        return int(removed)
