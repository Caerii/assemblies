"""Base drive + the FULL deviation store on the GPU.

The first attempt got this wrong in an instructive way: it derived a deviation
cell's starting value from the hash. That is not what a fiber holds. Recruitment
OVERRIDES cells -- it assigns 1.0 to a cell whose base may have been ABSENT --
so `present(i,j)` is the wrong test and the drive came out 23 units low.

The correct decomposition, straight from `VirtualWeights.row_sum`:

    w[i,j] = chain( eff[i,j], count[i,j] )          eff = 1.0 if overridden
                                                          else the raw base
    drive  = base_drive + sum over deviation cells of ( chain(eff, c) - raw )

and the crucial simplification is that the base is BERNOULLI 0/1, so `eff` and
`raw` are each 0 or 1. Therefore

    chain(eff, c) = eff * tab[c]      tab[c] = chain(1.0, c)

with `tab` replayed on the host using the dense engine's own per-step
multiply-and-clip. A deviation cell is then (col, count, eff_bit, raw_bit) and
the kernel needs NO hash at all for the correction -- two bits and a table
lookup. That is why this is cheap.
"""
import os
import sys
import time

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
os.environ["PATH"] = os.environ["CUDA_HOME"] + r"\bin;" + os.environ["PATH"]

import numpy as np                                        # noqa: E402
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

__global__ void base_kernel(const int* __restrict__ rows, int B, int K, int N,
                            unsigned int seed, float p,
                            float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    int b = (int)(idx / N), j = (int)(idx - (long long)b * N);
    unsigned int ch = ((unsigned int)j * 2246822519u) ^ seed;
    const int* rp = rows + (long long)b * K;
    int c = 0;
    for (int t = 0; t < K; ++t) {
        unsigned int h = fmix32(((unsigned int)rp[t] * 2654435761u) ^ ch);
        if ((float)(h & 0x00FFFFFFu) / 16777216.0f < p) ++c;
    }
    out[idx] = (float)c;
}

// One thread per (brain, active row). No hashing: a deviation cell carries its
// own eff/raw bits, because recruitment can override a cell the hash calls
// absent and deriving the start value from the hash is simply wrong.
__global__ void dev_kernel(const int* __restrict__ rows,
                           const long long* __restrict__ indptr,
                           const int* __restrict__ dcol,
                           const int* __restrict__ dcnt,
                           const unsigned char* __restrict__ dflag,
                           const float* __restrict__ tab, int ntab,
                           int B, int K, int N, float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K) return;
    int b = (int)(idx / K), t = (int)(idx - (long long)b * K);
    int row = rows[(long long)b * K + t];
    if (row < 0 || row >= N) return;
    long long lo = indptr[(long long)b * (N + 1) + row];
    long long hi = indptr[(long long)b * (N + 1) + row + 1];
    float* ob = out + (long long)b * N;
    for (long long e = lo; e < hi; ++e) {
        unsigned char f = dflag[e];
        int c = dcnt[e];
        float eff = (f & 1u) ? ((c < ntab) ? tab[c] : tab[ntab - 1]) : 0.0f;
        float raw = (f & 2u) ? 1.0f : 0.0f;
        float d = eff - raw;
        if (d != 0.0f) atomicAdd(ob + dcol[e], d);
    }
}

torch::Tensor drive_full(torch::Tensor rows, int64_t n, int64_t seed, double p,
                         torch::Tensor indptr, torch::Tensor dcol,
                         torch::Tensor dcnt, torch::Tensor dflag,
                         torch::Tensor tab) {
    rows = rows.contiguous();
    int B = rows.size(0), K = rows.size(1);
    auto out = torch::empty({B, (int64_t)n},
                            torch::dtype(torch::kFloat32).device(rows.device()));
    long long tot = (long long)B * n; int th = 256;
    base_kernel<<<(tot + th - 1) / th, th>>>(
        rows.data_ptr<int>(), B, K, (int)n, (unsigned int)seed, (float)p,
        out.data_ptr<float>());
    if (dcol.numel() > 0) {
        long long t2 = (long long)B * K;
        dev_kernel<<<(t2 + th - 1) / th, th>>>(
            rows.data_ptr<int>(), indptr.data_ptr<long long>(),
            dcol.data_ptr<int>(), dcnt.data_ptr<int>(),
            dflag.data_ptr<unsigned char>(), tab.data_ptr<float>(),
            (int)tab.numel(), B, K, (int)n, out.data_ptr<float>());
    }
    return out;
}
'''
CPP = ("torch::Tensor drive_full(torch::Tensor rows, int64_t n, int64_t seed,"
       " double p, torch::Tensor indptr, torch::Tensor dcol,"
       " torch::Tensor dcnt, torch::Tensor dflag, torch::Tensor tab);")


def build():
    return load_inline(name="na_drive_full2", cpp_sources=[CPP],
                       cuda_sources=[CUDA_SRC], functions=["drive_full"],
                       verbose=False, extra_cuda_cflags=["-O3"])


def chain_table(beta, hi, n=96):
    g = np.float32(1.0 + beta)
    out = np.ones(n, dtype=np.float32)
    v = np.float32(1.0)
    for c in range(1, n):
        v = np.float32(v * g)
        if hi is not None:
            v = np.float32(min(v, np.float32(hi)))
        out[c] = v
    return out


def store_from_vw(vw, n):
    """(indptr, col, cnt, flag) for one fiber -- exponents plus override-only
    cells, exactly the two sets `row_sum` walks."""
    lens = np.zeros(n, dtype=np.int64)
    C, N_, F = [], [], []
    for r in range(n):
        cols_r, cnt_r, flag_r = [], [], []
        exp_cols = None
        e = vw._exp.get(r)
        if e is not None and len(e[0]):
            c, cnt, eff, raw = e
            c = np.asarray(c, dtype=np.int64)
            keep = c < n
            if keep.any():
                c = c[keep]
                cols_r.append(c)
                cnt_r.append(np.asarray(cnt, dtype=np.int64)[keep])
                flag_r.append(((np.asarray(eff)[keep] != 0).astype(np.uint8))
                              | ((np.asarray(raw)[keep] != 0).astype(np.uint8) << 1))
                exp_cols = c
        if vw._ovr.get(r):
            oarr, ovals = vw._ovr_pair(r)
            keep = oarr < n
            oarr, ovals = oarr[keep], ovals[keep]
            if exp_cols is not None and len(oarr):
                m = ~vw._sorted_isin(oarr, exp_cols)
                oarr, ovals = oarr[m], ovals[m]
            if len(oarr):
                cols_r.append(oarr.astype(np.int64))
                cnt_r.append(np.zeros(len(oarr), dtype=np.int64))
                # override => eff bit set; raw bit from the stored base
                flag_r.append(np.uint8(1)
                              | ((np.asarray(ovals) != 0).astype(np.uint8) << 1))
        if cols_r:
            cc = np.concatenate(cols_r)
            C.append(cc.astype(np.int32))
            N_.append(np.concatenate(cnt_r).astype(np.int32))
            F.append(np.concatenate(flag_r).astype(np.uint8))
            lens[r] = len(cc)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lens)
    z32 = np.zeros(0, np.int32)
    return (indptr,
            np.concatenate(C) if C else z32,
            np.concatenate(N_) if N_ else z32,
            np.concatenate(F) if F else np.zeros(0, np.uint8))


def bench(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    mod = build()
    sys.path.insert(0, '.')
    sys.path.insert(0, 'research/experiments')
    import random
    os.environ['ASSEMBLIES_VIRTUAL_WEIGHTS'] = '1'
    import _substrate_arms as A

    print("=== correctness vs the CPU VirtualWeights row_sum ===")
    for M in (8, 32):
        cfg = A.Cfg(n=2000, k=50, p=0.5, beta=0.1, T=8, M=M)
        random.seed(42); np.random.seed(42)
        brain, stims, stored = A.train(cfg, 'B', 42)
        vw = brain._engine_for(brain.areas['A'])._area_conns['A']['A'].weights
        n, k = vw.n_cols, cfg.k
        ip, dc, dn, df = store_from_vw(vw, n)
        tab = chain_table(cfg.beta, vw.w_hi)
        worst_d, bad = 0.0, 0
        for a_i in range(min(M, 8)):
            rows_np = np.sort(np.asarray(stored[a_i].winners)[:k]).astype(np.int32)
            rows_np = rows_np[rows_np < n]
            ref = np.asarray(vw.row_sum(rows_np.astype(np.int64), n),
                             dtype=np.float64)
            got = mod.drive_full(
                torch.from_numpy(rows_np).cuda().unsqueeze(0), n,
                int(vw.pair_seed) & 0xFFFFFFFF, vw.p,
                torch.from_numpy(ip).cuda().unsqueeze(0).contiguous(),
                torch.from_numpy(dc).cuda(), torch.from_numpy(dn).cuda(),
                torch.from_numpy(df).cuda(), torch.from_numpy(tab).cuda(),
            )[0].cpu().numpy().astype(np.float64)
            worst_d = max(worst_d, float(np.abs(got - ref).max()))
            kk = min(k, len(rows_np))
            if set(np.argsort(-ref, kind='stable')[:kk].tolist()) != \
               set(np.argsort(-got, kind='stable')[:kk].tolist()):
                bad += 1
        print(f"  M={M:>3}  cells={len(dc):>7}  max|GPU-CPU| = {worst_d:.3e}"
              f"   top-k set differs on {bad}/8 assemblies")

    print("\n=== cost: base only vs base + deviations (dev/row from real runs)")
    print(f"{'n':>7} {'B':>5} {'dev/row':>8} {'base ms':>9} {'full ms':>9} "
          f"{'full/brain':>11} {'dev adds':>9}")
    tabg = torch.from_numpy(chain_table(0.1, None)).cuda()
    e_ip = torch.zeros(1, 1, dtype=torch.int64).cuda()
    e_i32 = torch.zeros(0, dtype=torch.int32).cuda()
    e_u8 = torch.zeros(0, dtype=torch.uint8).cuda()
    for N, dpr in ((20000, 48), (20000, 208)):
        for B in (64, 256):
            rows = torch.randint(0, N, (B, 50), device='cuda',
                                 dtype=torch.int32)
            rng = np.random.default_rng(0)
            ip = np.zeros((B, N + 1), dtype=np.int64)
            ip[:, 1:] = np.cumsum(np.full(N, dpr, dtype=np.int64))[None, :]
            tot = int(ip[0, -1])
            dc = torch.from_numpy(rng.integers(0, N, B * tot, dtype=np.int64)
                                  .astype(np.int32)).cuda()
            dn = torch.from_numpy(rng.integers(1, 12, B * tot).astype(np.int32)
                                  ).cuda()
            df = torch.from_numpy(rng.integers(1, 4, B * tot).astype(np.uint8)
                                  ).cuda()
            ipg = torch.from_numpy(ip).cuda().contiguous()
            tb = bench(lambda: mod.drive_full(rows, N, 12345, 0.05, e_ip,
                                              e_i32, e_i32, e_u8, tabg))
            tf = bench(lambda: mod.drive_full(rows, N, 12345, 0.05, ipg, dc,
                                              dn, df, tabg))
            print(f"{N:>7} {B:>5} {dpr:>8} {1000*tb:>9.4f} {1000*tf:>9.4f} "
                  f"{1000*tf/B:>11.5f} {tf/tb:>8.2f}x")
            del dc, dn, df, ipg, rows
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
