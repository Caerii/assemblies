"""Batched radix-select for k-WTA, v3: one launch, candidates in shared memory.

WHAT THE PROFILE SAID. v2 split the work into four kernels and measured

      n     B     hist   thresh  collect   refine      sum
  20000    64   0.0220   0.0286   0.0329   0.0408   0.1243

No stage dominates, and `thresh` -- which touches 4096 buckets per brain and
does nothing else -- costs as much as the stage that reads 5.1 MB. That is
launch and latency, not work. A four-kernel design has a floor near 4 x 0.025 ms
however good its algorithm is. I had been optimising bandwidth inside a
latency-bound regime.

SO: one kernel, one block per brain, 1024 threads.

  * pass 1  histogram the top 12 bits of the key                 (full data)
  * scan    3-level suffix search, 32+32+4 serial steps          (in-block)
  * pass 2  collect candidates into SHARED memory                (full data)
  * refine  bitonic sort of the block-resident candidates        (no DRAM)

Two full-data passes, one launch, 67% occupancy. The candidate set was measured
at <= 269 for every configuration tested, so 1024 shared slots carries a 3.8x
margin -- and overflow is DETECTED and reported rather than silently returning a
wrong assembly.

Keys remain `(float_bits << 16) | (65535 - j)`: unique, so "largest key" means
"largest value, ties to the smallest index", which is stable argsort by
construction. Tie-breaking is an identity here, not a policy -- and the drives
are integer-valued Bernoulli sums, so ties at the bar are the common case, not
an edge case.
"""
import os
import time

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
os.environ["PATH"] = os.environ["CUDA_HOME"] + r"\bin;" + os.environ["PATH"]

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from torch.utils.cpp_extension import load_inline           # noqa: E402

DEV = 'cuda'

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define NB    4096
#define NTH   1024
#define CAPS  1024          // shared candidate slots
#define CH    (NB / NTH)    // 4 buckets per thread

__device__ __forceinline__ unsigned long long mkkey(float v, int j) {
    return ((unsigned long long)__float_as_uint(v) << 16)
           | (unsigned long long)(65535 - j);
}

__global__ void select_kernel(const float* __restrict__ x, int N, int K,
                              int* __restrict__ out, int* __restrict__ ovf) {
    __shared__ unsigned hist[NB];
    __shared__ unsigned part[NTH];
    __shared__ unsigned sup[32];
    __shared__ unsigned long long ck[CAPS];
    __shared__ int s_d0;
    __shared__ int s_cnt;

    const int b = blockIdx.x;
    const long long base = (long long)b * N;
    const int tid = threadIdx.x;

    for (int i = tid; i < NB; i += NTH) hist[i] = 0u;
    __syncthreads();

    // ---- pass 1: histogram the top 12 bits (== float bits 31..20) ---------
    for (int j = tid; j < N; j += NTH)
        atomicAdd(&hist[__float_as_uint(x[base + j]) >> 20], 1u);
    __syncthreads();

    // ---- 3-level suffix search: 32 + 32 + 4 serial steps ------------------
    unsigned local = 0u;
    for (int i = 0; i < CH; ++i) local += hist[tid * CH + i];
    part[tid] = local;
    __syncthreads();
    if (tid < 32) {
        unsigned s = 0u;
        for (int i = 0; i < 32; ++i) s += part[tid * 32 + i];
        sup[tid] = s;
    }
    __syncthreads();
    if (tid == 0) {
        unsigned acc = 0u;
        int w = 31;
        for (; w > 0; --w) {
            if (acc + sup[w] >= (unsigned)K) break;
            acc += sup[w];
        }
        int t = w * 32 + 31;
        for (; t > w * 32; --t) {
            if (acc + part[t] >= (unsigned)K) break;
            acc += part[t];
        }
        int d = t * CH + CH - 1;
        for (; d > t * CH; --d) {
            if (acc + hist[d] >= (unsigned)K) break;
            acc += hist[d];
        }
        s_d0 = d;
        s_cnt = 0;
    }
    __syncthreads();

    // ---- pass 2: collect candidates straight into shared memory -----------
    const int thr = s_d0;
    for (int j = tid; j < N; j += NTH) {
        if ((int)(__float_as_uint(x[base + j]) >> 20) >= thr) {
            int s = atomicAdd(&s_cnt, 1);
            if (s < CAPS) ck[s] = mkkey(x[base + j], j);
        }
    }
    __syncthreads();
    const int M = s_cnt;
    if (M > CAPS) {                      // report, never return a wrong set
        if (tid == 0) ovf[b] = M;
        return;
    }
    if (tid == 0) ovf[b] = 0;
    for (int i = M + tid; i < CAPS; i += NTH) ck[i] = 0ULL;   // pad low
    __syncthreads();

    // ---- refine: bitonic sort of the block-resident candidates ------------
    for (int kk = 2; kk <= CAPS; kk <<= 1) {
        for (int jj = kk >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < CAPS; i += NTH) {
                const int ixj = i ^ jj;
                if (ixj > i) {
                    const bool up = ((i & kk) == 0);
                    if ((ck[i] > ck[ixj]) == up) {
                        const unsigned long long tmp = ck[i];
                        ck[i] = ck[ixj]; ck[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    // ascending, so the K largest are the last K
    int* ob = out + (long long)b * K;
    for (int s = tid; s < K; s += NTH)
        ob[s] = 65535 - (int)(ck[CAPS - 1 - s] & 0xFFFFULL);
}

std::vector<torch::Tensor> topk_select3(torch::Tensor x, int64_t K) {
    TORCH_CHECK(x.dim() == 2 && x.is_cuda()
                && x.scalar_type() == torch::kFloat32, "need 2-D f32 cuda");
    x = x.contiguous();
    const int B = x.size(0), N = x.size(1);
    TORCH_CHECK(N <= 65536, "n must be <= 65536 for a 16-bit index key");
    TORCH_CHECK(K <= CAPS, "K must fit the shared candidate buffer");
    auto iopt = torch::dtype(torch::kInt32).device(x.device());
    auto out = torch::empty({B, (int64_t)K}, iopt);
    auto ovf = torch::zeros({B}, iopt);
    select_kernel<<<B, NTH>>>(x.data_ptr<float>(), N, (int)K,
                              out.data_ptr<int>(), ovf.data_ptr<int>());
    return {out, ovf};
}
'''

CPP = r"""
std::vector<torch::Tensor> topk_select3(torch::Tensor x, int64_t K);
"""


def build():
    return load_inline(name="na_topk_select3", cpp_sources=[CPP],
                       cuda_sources=[CUDA_SRC], functions=["topk_select3"],
                       verbose=False, extra_cuda_cflags=["-O3"])


def drive_like(B, n, k, p=0.05, dev_frac=0.06, seed=0):
    """Drive with the real shape: integer Bernoulli base plus potentiation."""
    g = np.random.default_rng(seed)
    base = g.binomial(k, p, size=(B, n)).astype(np.float32)
    m = g.random((B, n)) < dev_frac
    bump = (1.1 ** g.integers(1, 12, size=(B, n))).astype(np.float32)
    return (base + np.where(m, bump, 0.0)).astype(np.float32)


def bench(fn, reps=40):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    mod = build()

    print("=== exactness vs stable argsort ===")
    for B, n, k in ((8, 20000, 70), (8, 4000, 60), (8, 50000, 100),
                    (8, 20000, 200)):
        x = drive_like(B, n, k, seed=1)
        xt = torch.from_numpy(x).cuda()
        out, ovf = mod.topk_select3(xt, k)
        got, o = out.cpu().numpy(), ovf.cpu().numpy()
        if o.any():
            print(f"  B={B} n={n} k={k}: CANDIDATE OVERFLOW {o.max()} -- "
                  f"result withheld")
            continue
        ok = all(np.array_equal(np.sort(np.argsort(-x[b], kind='stable')[:k]),
                                np.sort(got[b])) for b in range(B))
        print(f"  B={B} n={n} k={k}: exact vs stable argsort {ok}")
        del xt
    torch.cuda.empty_cache()

    print("\n=== cost: torch.topk vs v3 ===")
    hdr = (f"{'n':>7} {'k':>4} {'B':>5} {'topk ms':>9} {'v3 ms':>9} "
           f"{'speedup':>8} {'GB/s':>7}")
    print(hdr)
    print("-" * len(hdr))
    for n, k in ((20000, 70), (50000, 100)):
        for B in (64, 256, 1024):
            x = torch.from_numpy(drive_like(min(B, 64), n, k, seed=2)).cuda()
            if B > 64:
                x = x.repeat(B // 64, 1).contiguous()
            t_tk = bench(lambda: torch.topk(x, k))
            t_v3 = bench(lambda: mod.topk_select3(x, k))
            gbs = 2 * B * n * 4 / t_v3 / 1e9        # two full-data passes
            print(f"{n:>7} {k:>4} {B:>5} {1000*t_tk:>9.4f} "
                  f"{1000*t_v3:>9.4f} {t_tk/t_v3:>7.1f}x {gbs:>7.0f}")
            del x
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
