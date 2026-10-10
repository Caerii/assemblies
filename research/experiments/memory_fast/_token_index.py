"""The token index: neuron -> the stored tokens it belongs to; overlaps in k (L k / n).

Part of research.experiments.memory_fast (see its __init__ for why each path is exact)."""
from __future__ import annotations


class TokenIndex:
    """neuron -> the stored tokens it belongs to, per brain: ``table`` [B, n, cap] (token ids,
    -1 padded) and ``fill`` [B, n]. Appending a token is O(k); the overlaps of V states with
    every token cost V k cap."""

    def __init__(self, B, n, device, cap=64):
        import torch
        self.B, self.n, self.device = B, n, device
        self.table = torch.full((B, n, cap), -1, dtype=torch.int32, device=device)
        self.fill = torch.zeros(B, n, dtype=torch.int64, device=device)
        self.L = 0
        self._b = torch.arange(B, device=device).view(-1, 1)

    @classmethod
    def build(cls, allst, n, device):
        idx = cls(allst.shape[1], n, device)
        idx.extend(allst)
        return idx

    def _grow(self, need):
        import torch
        cap = self.table.shape[2]
        if need <= cap:
            return
        new = max(need, 2 * cap)
        t = torch.full((self.B, self.n, new), -1, dtype=torch.int32, device=self.device)
        t[:, :, :cap] = self.table
        self.table = t

    def extend(self, tokens):
        """append tokens [T, B, k] (ids L, L+1, ...)"""
        import torch
        T = tokens.shape[0]
        if T == 0:
            return
        k = tokens.shape[2]
        # every (brain, neuron, token) entry, ordered by brain and neuron, tokens ascending within
        neuron = tokens.long().permute(1, 0, 2).reshape(self.B, T * k)          # [B, T k]
        tok = (torch.arange(T, device=self.device).repeat_interleave(k) + self.L).view(1, -1).expand(self.B, -1)
        key = (self._b * self.n + neuron).reshape(-1)
        order = torch.sort(key, stable=True).indices
        key_s = key[order]
        tok_s = tok.reshape(-1)[order]
        start = torch.ones_like(key_s, dtype=torch.bool)
        start[1:] = key_s[1:] != key_s[:-1]
        i = torch.arange(key_s.numel(), device=self.device)
        first = torch.cummax(torch.where(start, i, torch.zeros_like(i)), dim=0).values
        rank = i - first                                   # position among this call's entries
        b_s, n_s = key_s // self.n, key_s % self.n
        pos = self.fill[b_s, n_s] + rank
        self._grow(int(pos.max()) + 1)
        self.table[b_s, n_s, pos] = tok_s.to(torch.int32)
        self.fill.view(-1).scatter_add_(0, key_s, torch.ones_like(key_s))
        self.L += T

    def overlaps(self, states, brains):
        """[V, L] float32 overlap counts of states [V, k] (brain ``brains`` [V]) with every token"""
        import torch
        V = states.shape[0]
        ids = self.table[brains.long().view(-1, 1), states.long()]          # [V, k, cap]
        ids = ids.reshape(V, -1).long()
        ids = torch.where(ids < 0, torch.full_like(ids, self.L), ids)
        out = torch.zeros(V, self.L + 1, dtype=torch.float32, device=self.device)
        out.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        return out[:, :self.L]
