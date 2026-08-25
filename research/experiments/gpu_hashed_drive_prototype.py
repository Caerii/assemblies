"""PROTOTYPE: generate the connectome on the GPU instead of fetching it.

Implements and measures `research/notes/DESIGN_gpu_hashed_drive.md`. NOT wired
into any engine -- this computes one well-defined quantity (the base drive of a
row-set, for a batch of independent brains) and checks it against the rust
kernel the numpy engine actually uses.

Needs CUDA, nvcc and a host compiler; skips cleanly without them. On Windows
run it from a shell that has run `vcvars64.bat`, or via the batch wrapper in
the design note.

    python research/experiments/gpu_hashed_drive_prototype.py

THE POINT, in one line: a cell's weight is a pure function of its position, and
on a card with a ~100:1 arithmetic-to-bandwidth ratio, computing it is cheaper
than reading it -- but only if the intermediates never reach memory, which is
why this is a fused kernel and not a torch expression.
"""

import os
import sys
import time

VC = (r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC"
      r"\Auxiliary\Build\vcvars64.bat")
os.environ.setdefault("CUDA_HOME",
                      r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
os.environ["PATH"] = (os.environ["CUDA_HOME"] + r"\bin;" + os.environ["PATH"])

import torch                                              # noqa: E402
from torch.utils.cpp_extension import load_inline         # noqa: E402

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned int fmix32(unsigned int h) {
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16; return h;
}

// drive[b, j] = # { i in rows[b, :] : cell (i, j) is present }
// One thread per (b, j). The k hashes accumulate in a register; the only
// memory traffic is k row-ids (broadcast, cached) in and one float out.
__global__ void drive_kernel(
    const int* __restrict__ rows, int B, int K, int N,
    unsigned int seed, float p, float* __restrict__ out)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)B * N;
    if (idx >= total) return;
    int b = (int)(idx / N);
    int j = (int)(idx - (long long)b * N);

    const unsigned int MUL_ROW = 2654435761u;
    const unsigned int MUL_COL = 2246822519u;
    const unsigned int MANT    = 0x00FFFFFFu;
    const float SCALE = 16777216.0f;

    unsigned int ch = ((unsigned int)j * MUL_COL) ^ seed;
    const int* rp = rows + (long long)b * K;
    int count = 0;
    for (int t = 0; t < K; ++t) {
        unsigned int h = ((unsigned int)rp[t] * MUL_ROW) ^ ch;
        h = fmix32(h);
        if ((float)(h & MANT) / SCALE < p) ++count;
    }
    out[idx] = (float)count;
}

torch::Tensor drive_hashed_cuda(torch::Tensor rows, int64_t n,
                                int64_t seed, double p) {
    TORCH_CHECK(rows.is_cuda() && rows.dtype() == torch::kInt32);
    rows = rows.contiguous();
    int B = rows.size(0), K = rows.size(1);
    auto out = torch::empty({B, (int64_t)n},
                            torch::dtype(torch::kFloat32).device(rows.device()));
    long long total = (long long)B * n;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    drive_kernel<<<blocks, threads>>>(
        rows.data_ptr<int>(), B, K, (int)n,
        (unsigned int)seed, (float)p, out.data_ptr<float>());
    return out;
}
'''

CPP_SRC = ("torch::Tensor drive_hashed_cuda(torch::Tensor rows, int64_t n, "
           "int64_t seed, double p);")


def build():
    return load_inline(
        name="na_drive_fused",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=["drive_hashed_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


# ---- reference: the same arithmetic in torch, for correctness -------------
M32 = 0xFFFFFFFF


def _fmix32_t(h):
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & M32
    h = h ^ (h >> 16)
    return h & M32


def drive_torch(rows, n, seed, p, tile=8192):
    B, K = rows.shape
    out = torch.empty(B, n, device='cuda', dtype=torch.float32)
    rh = (rows.to(torch.int64) * 2654435761) & M32
    for c0 in range(0, n, tile):
        c1 = min(c0 + tile, n)
        cols = torch.arange(c0, c1, device='cuda', dtype=torch.int64)
        ch = ((cols * 2246822519) & M32) ^ seed
        h = _fmix32_t((rh.unsqueeze(-1) ^ ch) & M32)
        u = (h & 0x00FFFFFF).to(torch.float32) / 16777216.0
        out[:, c0:c1] = (u < p).sum(dim=1, dtype=torch.float32)
    return out


def bench(fn, reps=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    mod = build()
    print("fused kernel built\n")
    p, k = 0.05, 70

    # CORRECTNESS AGAINST THE ENGINE'S OWN KERNEL, not against my torch
    # transcription of the hash. A GPU kernel that agrees with my
    # reimplementation and disagrees with the engine would be a fast WRONG
    # answer, which is the only outcome worse than a slow right one.
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from neural_assemblies.core.numpy_engine._seeding import (
        fnv1a_pair_seed, hash_area_weights_rows,
    )
    seed = int(fnv1a_pair_seed(12345, "A", "A")) & 0xFFFFFFFF
    rr = np.sort(np.random.default_rng(0).choice(4000, k, replace=False))
    blk = np.asarray(hash_area_weights_rows(rr.astype(np.int64), 0, 4000,
                                            seed, p, 0.0, -1.0))
    ref = (blk != 0).sum(axis=0).astype(np.float32)
    got = mod.drive_hashed_cuda(
        torch.from_numpy(rr.astype(np.int32)).cuda().unsqueeze(0),
        4000, seed, p)[0].cpu().numpy()
    ok = np.array_equal(got, ref)
    print(f"fused GPU == na_kernels rust drive: {ok}"
          f"  (max|diff| {np.abs(got - ref).max():.0f})")
    print(f"   mean drive {ref.mean():.3f}, expect k*p = {k * p:.1f}")
    if not ok:
        raise SystemExit("fused kernel disagrees with the engine -- stop")

    rows = torch.randint(0, 4000, (4, k), device='cuda', dtype=torch.int32)
    a = mod.drive_hashed_cuda(rows, 4000, seed, p)
    b = drive_torch(rows, 4000, seed, p)
    print(f"   fused == torch spelling: {torch.equal(a, b)}\n")

    print(f"{'n':>7} {'B':>5} {'torch ms':>10} {'fused ms':>10} "
          f"{'fused/brain':>12} {'speedup':>9}")
    for n in (4000, 20000, 50000):
        for B in (1, 16, 64, 256):
            rows = torch.randint(0, n, (B, k), device='cuda',
                                 dtype=torch.int32)
            t_f = bench(lambda: mod.drive_hashed_cuda(rows, n, seed, p))
            try:
                t_t = bench(lambda: drive_torch(rows, n, seed, p), reps=5)
            except torch.cuda.OutOfMemoryError:
                t_t = float('nan')
            torch.cuda.empty_cache()
            print(f"{n:>7} {B:>5} {1000*t_t:>10.3f} {1000*t_f:>10.4f} "
                  f"{1000*t_f/B:>12.5f} {t_t/t_f:>8.1f}x")
        print()


if __name__ == "__main__":
    main()
