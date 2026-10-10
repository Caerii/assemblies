"""Replay cues: memory_load_drift.cue's permutations for every cue at once (vectorised Philox).

Part of research.experiments.memory_fast (see its __init__ for why each path is exact)."""
from __future__ import annotations


_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


def _mulhilo(a, b):
    p = a * b                                              # < 2^64: wraps in int64, bits intact
    return (p >> 32) & _MASK, p & _MASK


def philox_uniform(seeds, k, device):
    """[N, k] float32: torch.rand(k, generator=a fresh CUDA generator seeded seeds[i]), row i."""
    import torch
    s = torch.as_tensor(seeds, dtype=torch.int64, device=device).view(-1, 1)
    N = s.shape[0]
    k0 = (s & _MASK).expand(N, k).clone()
    k1 = ((s >> 32) & _MASK).expand(N, k).clone()
    c0 = torch.zeros(N, k, dtype=torch.int64, device=device)
    c1 = torch.zeros_like(c0)
    c2 = torch.arange(k, dtype=torch.int64, device=device).view(1, -1).expand(N, k).clone()
    c3 = torch.zeros_like(c0)
    for r in range(10):
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if r < 9:
            k0 = (k0 + _W0) & _MASK
            k1 = (k1 + _W1) & _MASK
    inv = torch.tensor(2.0 ** -32, dtype=torch.float32, device=device)
    u = c0.to(torch.float32) * inv + inv / 2.0
    return torch.where(u == 1.0, torch.zeros_like(u), u)


_PHILOX_OK = None


def philox_matches_torch(device, trials=64):
    """Does philox_uniform reproduce torch.rand on this torch build? Checked once per process."""
    global _PHILOX_OK
    if _PHILOX_OK is None:
        import torch
        seeds = [i * 1_000_003 + 3472 * 100_000 + i * 7 for i in range(trials)] + [2**40 + 12345, 1]
        mine = philox_uniform(seeds, 75, device)
        ref = torch.stack([torch.rand(75, device=device, generator=torch.Generator(device=device).manual_seed(s))
                           for s in seeds])
        _PHILOX_OK = bool(torch.equal(mine, ref))
    return _PHILOX_OK


def cue_index(seeds, k, device):
    """[N, k // 2] int64: the positions memory_load_drift.cue keeps, for generator seeds `seeds`."""
    import torch
    if k >= 256 or not philox_matches_torch(device):
        out = []
        for s in seeds:
            gen = torch.Generator(device=device).manual_seed(int(s))
            out.append(torch.argsort(torch.rand(k, device=device, generator=gen))[:k // 2])
        return torch.stack(out)
    vals = philox_uniform(seeds, k, device)
    perm = torch.argsort(vals, dim=1)
    srt = torch.gather(vals, 1, perm)
    tied = (srt[:, 1:] == srt[:, :-1]).any(dim=1).nonzero().view(-1).tolist()
    for r in tied:                                        # the original 1-D sort decides tie order
        perm[r] = torch.argsort(vals[r])
    return perm[:, :k // 2]
