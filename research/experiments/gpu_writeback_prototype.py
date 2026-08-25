"""Write-back cost at REALISTIC winner reuse. Lean setup.

The previous version of this measurement built its winner sets with
`randperm(n)` per (brain, item) -- 16,384 full permutations of 20,000 elements
to pick 50 indices each -- and grew its append buffer with `cat` in a loop. It
spent all its time on setup and 8.6 GB of memory, and measured nothing. Drawing
k indices directly is O(M*B*k) instead of O(M*B*n): 819k values instead of 327M.

WHY REUSE MATTERS. Random winners make nearly every (i,j) key unique -- 20M
distinct cells, where a real M=32 run holds 117k. Assemblies stabilize, so the
same pairs recur across the T rounds that train one item. The store size is what
compaction cost scales with, so a random-winner benchmark is a pessimistic bound
by two orders of magnitude, not a measurement.
"""
import os
import time

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import torch                                              # noqa: E402

DEV = 'cuda'


def event_keys(prev, new, n):
    B, k = prev.shape
    off = (torch.arange(B, device=DEV, dtype=torch.int64)
           * (n * n)).view(B, 1, 1)
    return (off + prev.view(B, k, 1) * n + new.view(B, 1, k)).reshape(-1)


def compact(keys, counts, buf):
    """Sort + segment-reduce. No hashing, no probing, no atomics."""
    if buf.numel() == 0:
        return keys, counts
    allk = torch.cat([keys, buf])
    allv = torch.cat([counts, torch.ones_like(buf, dtype=torch.int32)])
    order = torch.argsort(allk)
    uk, inv, _ = torch.unique_consecutive(allk[order], return_inverse=True,
                                          return_counts=True)
    uv = torch.zeros(uk.numel(), dtype=torch.int32, device=DEV)
    uv.scatter_add_(0, inv, allv[order])
    return uk, uv


def run(n, k, B, M, T, churn=0.1):
    g = torch.Generator(device=DEV).manual_seed(1)
    # k indices per (brain, item), drawn directly -- no permutation of n.
    cores = torch.randint(0, n, (M, B, k), device=DEV, generator=g,
                          dtype=torch.int64)
    keys = torch.zeros(0, dtype=torch.int64, device=DEV)
    counts = torch.zeros(0, dtype=torch.int32, device=DEV)
    nj = max(1, int(k * churn))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_app = 0.0
    for a in range(M):
        parts, prev = [], cores[a]
        for _ in range(T):
            new = cores[a].clone()
            jit = torch.randint(0, k, (B, nj), device=DEV, generator=g)
            new.scatter_(1, jit, torch.randint(0, n, (B, nj), device=DEV,
                                               generator=g, dtype=torch.int64))
            ta = time.perf_counter()
            parts.append(event_keys(prev, new, n))
            t_app += time.perf_counter() - ta
            prev = new
        keys, counts = compact(keys, counts, torch.cat(parts))
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return total, t_app, int(keys.numel())


def main():
    print("append-only write-back + batched compaction, realistic reuse")
    print(f"{'n':>7} {'k':>4} {'B':>5} {'M':>4} {'T':>3} {'store cells':>12} "
          f"{'total s':>8} {'per brain/round ms':>19}")
    for n, k, M, T in ((20000, 50, 32, 8), (20000, 70, 32, 8)):
        for B in (64, 256):
            tot, app, cells = run(n, k, B, M, T)
            per = 1000 * tot / (M * T * B)
            print(f"{n:>7} {k:>4} {B:>5} {M:>4} {T:>3} {cells:>12} "
                  f"{tot:>8.2f} {per:>19.5f}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
