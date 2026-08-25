"""batched_project_hashed vs batched_project_independent.

Two regimes, and the second is the point:

  1. where BOTH can run -- a like-for-like speed comparison;
  2. organ scale -- where the block-diagonal CSR cannot be built at all. That
     row reports the nnz it WOULD need rather than attempting the allocation,
     because measuring a machine into swap is not a measurement.
"""
import os
import time

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import sys                                                  # noqa: E402
sys.path.insert(0, 'F:/Github/assemblies')
from neural_assemblies.core.torch_engine import _hash as t_hash      # noqa: E402
from neural_assemblies.core.torch_engine._batched import (           # noqa: E402
    batched_project_hashed, batched_project_independent, block_diagonal)

P = 0.05
SEED = 0x5EED1234
DEV = 'cuda'


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def both(n, k, B, rounds):
    seeds = [SEED + 17 * b for b in range(B)]
    g = np.random.default_rng(5)
    w0 = np.stack([np.sort(g.choice(n, k, replace=False)) for _ in range(B)])
    w0t = torch.from_numpy(w0).cuda()

    torch.cuda.reset_peak_memory_stats()
    mats = []
    for s in seeds:
        W = t_hash.hash_bernoulli_2d(0, n, 0, n, s, P, device=DEV).float()
        mats.append(W.to_sparse_coo())
        del W
    Wb = block_diagonal(mats, n)
    del mats
    torch.cuda.synchronize()
    mem_csr = torch.cuda.max_memory_allocated() / 1e6
    nnz = int(Wb._nnz())
    t_csr = bench(lambda: batched_project_independent(
        Wb, w0t, B, n, k, rounds))
    del Wb
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    si = [to_i32(s) for s in seeds]
    t_hsh = bench(lambda: batched_project_hashed(n, k, P, si, w0t, rounds))
    torch.cuda.synchronize()
    mem_hsh = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()
    return t_csr, mem_csr, nnz, t_hsh, mem_hsh


def main():
    print("=== where BOTH can run ===")
    hdr = (f"{'n':>6} {'k':>4} {'B':>4} {'rnds':>5} | {'blockdiag ms':>12} "
           f"{'nnz':>11} {'MB':>8} | {'hashed ms':>10} {'MB':>7} | "
           f"{'speed':>7} {'mem':>7}")
    print(hdr)
    print("-" * len(hdr))
    for n, k, B, r in ((2048, 40, 8, 3), (4096, 60, 8, 3), (4096, 60, 16, 3)):
        tc, mc, nnz, th, mh = both(n, k, B, r)
        print(f"{n:>6} {k:>4} {B:>4} {r:>5} | {1000*tc:>12.3f} {nnz:>11,} "
              f"{mc:>8.1f} | {1000*th:>10.3f} {mh:>7.1f} | "
              f"{tc/th:>6.1f}x {mc/mh:>6.1f}x")

    print("\n=== organ scale: only the hashed path exists ===")
    for n, k, B, r in ((20000, 70, 64, 3), (20000, 70, 256, 3),
                       (50000, 100, 64, 3)):
        g = np.random.default_rng(5)
        w0 = torch.from_numpy(
            np.stack([np.sort(g.choice(n, k, replace=False))
                      for _ in range(B)])).cuda()
        si = [to_i32(SEED + 17 * b) for b in range(B)]
        torch.cuda.reset_peak_memory_stats()
        t = bench(lambda: batched_project_hashed(n, k, P, si, w0, r), reps=3)
        mem = torch.cuda.max_memory_allocated() / 1e6
        would = B * n * n * P
        print(f"  n={n:>6} k={k:>4} B={B:>5} rounds={r}: "
              f"{1000*t:>8.2f} ms, {mem:>7.1f} MB "
              f"| block-diagonal would need {would/1e9:>6.2f}e9 edges "
              f"(~{would*12/1e9:.0f} GB)")
        del w0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
