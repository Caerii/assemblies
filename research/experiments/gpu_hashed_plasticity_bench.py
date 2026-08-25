"""What does plasticity cost in the hashed batched path?

beta=0 is base drive + selection. beta>0 adds the potentiation correction,
whose work is B*k*C with C = |union of winner sets so far| <= T*k -- so the
cost GROWS with the round count even though the stored state does not. That
growth is the thing to measure; the round-mask staying at [B, n] int64 is the
thing not to have to.
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
from neural_assemblies.core.torch_engine._batched import (   # noqa: E402
    batched_project_hashed)

P = 0.05
SEED = 0x5EED1234


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def bench(fn, reps=3):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    print(f"{'n':>7} {'k':>4} {'B':>5} {'T':>4} | {'beta=0 ms':>10} "
          f"{'beta>0 ms':>10} {'x':>6} | {'per br/rd ms':>12} {'MB':>7}")
    print("-" * 78)
    for n, k, B, T in ((20000, 70, 64, 8), (20000, 70, 64, 32),
                       (20000, 70, 256, 8), (50000, 100, 64, 8)):
        g = np.random.default_rng(5)
        w0 = torch.from_numpy(np.stack(
            [np.sort(g.choice(n, k, replace=False)) for _ in range(B)])).cuda()
        si = [to_i32(SEED + 17 * b) for b in range(B)]
        t0 = bench(lambda: batched_project_hashed(n, k, P, si, w0, T))
        torch.cuda.reset_peak_memory_stats()
        t1 = bench(lambda: batched_project_hashed(
            n, k, P, si, w0, T, beta=0.1, w_max=20.0))
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"{n:>7} {k:>4} {B:>5} {T:>4} | {1000*t0:>10.2f} "
              f"{1000*t1:>10.2f} {t1/t0:>5.1f}x | "
              f"{1000*t1/(T*B):>12.5f} {mem:>7.1f}")
        del w0
        torch.cuda.empty_cache()

    print("\nCPU numpy_sparse reference for one projection: ~1.5-5 ms")


if __name__ == "__main__":
    main()
