"""Fused CUDA kernels: generate the connectome, and select without sorting.

Two kernels, both optional -- everything here degrades to ``None`` if nvcc, a
host compiler or ninja is missing, and callers must check :func:`available`.

WHY THESE EXIST. See ``research/notes/DESIGN_gpu_hashed_drive.md``. A cell's
weight is a pure function of its position, and on a card with a ~100:1
arithmetic-to-bandwidth ratio, computing it is cheaper than reading it -- but
only if the intermediates never reach memory, which is why this is a fused
kernel and not a torch expression (the torch spelling measured 0.7-0.9x against
the stored SpMM it was meant to beat, because every elementwise op writes DRAM).

THE HASH IS THE ENGINE'S, NOT A RE-DERIVATION. `drive` must agree cell-for-cell
with ``_hash.hash_bernoulli_2d``, including murmur3's fmix32 finalizer and the
INTEGER threshold convention ``(h & 0xFFFFFF) < int(p * 2**24)``. A float
comparison ``(h & 0xFFFFFF) / 2**24 < p`` is NOT the same predicate -- it
differs on the boundary cell whenever ``p * 2**24`` is not an integer. A kernel
that agrees with a plausible transcription and disagrees with the engine is a
fast wrong answer, which is the only outcome worse than a slow right one, so
``tests/test_fused_cuda.py`` pins it against the engine's own function.

SELECTION IS A COUNTING PROBLEM. The drive is a Bernoulli sum -- an integer in
[0, k] -- so it is bounded, concentrated and massively tied. ``topk_select``
histograms the top 12 bits of the key, which narrows n=20000 to a few hundred
candidates in one pass, then sorts those in shared memory.

    ** THE TIE ORDER IS DIFFERENT FROM torch.topk, ON PURPOSE. **

The key is ``(float_bits << 16) | (65535 - j)``, so keys are unique and
"largest key" means "largest value, ties to the smallest index" -- stable
argsort, by construction. ``torch.topk``'s tie order is unspecified and the
engine's CPU selector (``heapq_select_top_k``) is argpartition+argsort, both
unstable. With 5-18 columns tied at the bar in practice, that changes WHICH
neurons fire, not merely their order. ``_kwta_prune`` states the rule for this
project: making the selector's tie-break canonical is a SCIENCE-AFFECTING
change that needs its own registration and must not be smuggled in as an
optimisation. So nothing here is used unless a caller asks for it explicitly.
"""
from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()
_MODULE = None
_TRIED = False
_ERROR: str | None = None

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define NB    4096
#define NTH   1024
// Shared candidate slots. 2048 keys x 8B = 16 KB, plus hist 16 KB and part
// 4 KB = 36 KB, inside the 48 KB default. Raised from 1024 because k=sqrt(n)
// at n=16000 (k=126) overflowed at 1551 candidates -- the guard refused rather
// than truncating, which is correct, but it blocked the measurement.
#define CAPS  2048
#define CHB   (NB / NTH)

// 2654435761u / 2246822519u below are _hash._HASH_A / _HASH_B as unsigned.
__device__ __forceinline__ unsigned int ac_fmix32(unsigned int h) {
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16; return h;
}

// drive[b, j] = #{ i in rows[b, :] : cell (i, j) is present }
// One thread per (b, j); the K hashes accumulate in a REGISTER, so the only
// traffic is K row-ids (broadcast, cached) in and one float out.
__global__ void hashed_drive_kernel(
    const int* __restrict__ rows, const int* __restrict__ seeds,
    int B, int K, int N, int threshold, float* __restrict__ out)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    int b = (int)(idx / N), j = (int)(idx - (long long)b * N);
    unsigned int ch = ((unsigned int)j * 2246822519u) ^ (unsigned int)seeds[b];
    const int* rp = rows + (long long)b * K;
    int c = 0;
    for (int t = 0; t < K; ++t) {
        unsigned int h = ac_fmix32(((unsigned int)rp[t] * 2654435761u) ^ ch);
        if ((h & 0x00FFFFFFu) < (unsigned int)threshold) ++c;
    }
    out[idx] = (float)c;
}

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
    __shared__ unsigned long long s_prefix;
    __shared__ int s_rem;
    __shared__ int s_cnt;

    const int b = blockIdx.x;
    const long long base = (long long)b * N;
    const int tid = threadIdx.x;

    // TWO radix levels, not one. A single 12-bit pass cannot narrow a drive
    // whose values cluster on INTEGERS: the base is a Bernoulli count, so at
    // k=240 every untouched column shares an exact value and lands in one
    // bucket. Measured 3900 candidates at n=16000, k=240 against a 2048-slot
    // buffer, and the buffer cannot grow past 48 KB of static shared memory.
    // Two levels give 24 bits of discrimination and the boundary group
    // collapses to tens.
    if (tid == 0) { s_prefix = 0ULL; s_rem = K; }
    __syncthreads();
    for (int lvl = 0; lvl < 2; ++lvl) {
        const int shift = 36 - 12 * lvl;
        for (int i = tid; i < NB; i += NTH) hist[i] = 0u;
        __syncthreads();
        const unsigned long long pref = s_prefix;
        for (int j = tid; j < N; j += NTH) {
            const unsigned long long key = mkkey(x[base + j], j);
            if ((key >> (shift + 12)) == pref)
                atomicAdd(&hist[(unsigned)((key >> shift) & 4095ULL)], 1u);
        }
        __syncthreads();
        unsigned local = 0u;
        for (int i = 0; i < CHB; ++i) local += hist[tid * CHB + i];
        part[tid] = local;
        __syncthreads();
        if (tid < 32) {
            unsigned sm = 0u;
            for (int i = 0; i < 32; ++i) sm += part[tid * 32 + i];
            sup[tid] = sm;
        }
        __syncthreads();
        if (tid == 0) {
            const int rem = s_rem;
            unsigned acc = 0u;
            int w = 31;
            for (; w > 0; --w) { if (acc + sup[w] >= (unsigned)rem) break; acc += sup[w]; }
            int t = w * 32 + 31;
            for (; t > w * 32; --t) { if (acc + part[t] >= (unsigned)rem) break; acc += part[t]; }
            int d = t * CHB + CHB - 1;
            for (; d > t * CHB; --d) { if (acc + hist[d] >= (unsigned)rem) break; acc += hist[d]; }
            s_rem = rem - (int)acc;
            s_prefix = (s_prefix << 12) | (unsigned long long)d;
        }
        __syncthreads();
    }
    if (tid == 0) s_cnt = 0;
    __syncthreads();

    // At or above the 24-bit prefix: the definite winners plus the boundary
    // group. Fewer than K are strictly above, so the total is K plus however
    // many share the boundary prefix.
    const unsigned long long pref24 = s_prefix;
    for (int j = tid; j < N; j += NTH) {
        const unsigned long long key = mkkey(x[base + j], j);
        if ((key >> 24) >= pref24) {
            int s = atomicAdd(&s_cnt, 1);
            if (s < CAPS) ck[s] = key;
        }
    }
    __syncthreads();
    const int M = s_cnt;
    if (M > CAPS) {                 // report; never return a wrong winner set
        if (tid == 0) ovf[b] = M;
        return;
    }
    if (tid == 0) ovf[b] = 0;
    for (int i = M + tid; i < CAPS; i += NTH) ck[i] = 0ULL;
    __syncthreads();

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
    int* ob = out + (long long)b * K;
    for (int s = tid; s < K; s += NTH)
        ob[s] = 65535 - (int)(ck[CAPS - 1 - s] & 0xFFFFULL);
}


// ---- potentiation correction ------------------------------------------
// w[i,j] = present(i,j) * chain(1, count[i,j]), and the base is Bernoulli
// 0/1, so a cell that is ABSENT stays absent however often it is potentiated
// and a present one starts at exactly 1.0. The correction is therefore
//     sum over deviation cells of ( tab[count] - 1 ) * present(i,j)
// with tab[c] = chain(1.0, c) replayed on the host using the engine's own
// per-step multiply-and-clip, so the arithmetic here is a LOOKUP.
//
// count[i,j] is not stored. From `count = SUM_t x_{t-1} x_t^T` it is
//     count[i,j] = popcount( rowmask[i] & colmask[j] )
// where bit t of rowmask[i] says "i fired at t-1" and bit t of colmask[j]
// says "j fired at t". One 64-bit AND and one __popcll -- no block to
// materialise, which is what keeps this independent of the round count.
__global__ void dev_correct_kernel(
    const int* __restrict__ S,             // [B, K] current row set
    const long long* __restrict__ rowmask, // [B, N] bit t = fired at t-1
    const int* __restrict__ colids,        // [B, C] touched columns
    const long long* __restrict__ colmask, // [B, C] bit t = fired at t
    const float* __restrict__ tab, int ntab,
    const int* __restrict__ seeds,
    int B, int K, int C, int N, int W, int threshold,
    float* __restrict__ out)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K * C) return;
    int cj = (int)(idx % C);
    long long q = idx / C;
    int s = (int)(q % K), b = (int)(q / K);

    int i = S[(long long)b * K + s];
    // Masks are [B, W, n] / [B, W, C]: W 64-bit words per entry, so the
    // history is not capped at 64 rounds. count[i,j] = sum_w popcount(&).
    int c = 0;
    for (int w = 0; w < W; ++w) {
        unsigned long long rm =
            (unsigned long long)rowmask[((long long)b * W + w) * N + i];
        if (rm == 0ULL) continue;
        unsigned long long cm =
            (unsigned long long)colmask[((long long)b * W + w) * C + cj];
        c += __popcll(rm & cm);
    }
    if (c == 0) return;

    int j = colids[(long long)b * C + cj];
    unsigned int h = ac_fmix32(((unsigned int)i * 2654435761u)
                               ^ (((unsigned int)j * 2246822519u)
                                  ^ (unsigned int)seeds[b]));
    if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) return;   // absent cell
    float v = (c < ntab) ? tab[c] : tab[ntab - 1];
    atomicAdd(out + (long long)b * N + j, v - 1.0f);
}


// ---- norm_init's divisor, EXACTLY ------------------------------------
// `_pricing.inverse_indegree` computes  d_j = deg_j + p * (n_pre - rows_known)
// because engines materialise neurons lazily and neuron j's full incoming
// column does not exist yet; the second term is an unbiased estimate of the
// rows that have not appeared. THAT TERM IS IDENTICALLY ZERO HERE. A generated
// connectome has every row from the start, so rows_known == n_pre and
//     d_j = #{ i in [0, n) : cell (i, j) is present }
// is the TRUE in-degree, not an estimate. The whole defect class that lives on
// the estimate -- pricing unknown rows at the brain's p instead of the fiber's
// (79fba4f, a 6.15x over-scale) -- cannot occur in this path.
//
// One thread per (b, j), n hashes each. This is O(B * n^2) and is computed
// ONCE per brain set, not per round: the connectome does not change.
__global__ void indegree_kernel(const int* __restrict__ seeds, int B, int N,
                                int threshold, float floor_,
                                float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    int b = (int)(idx / N), j = (int)(idx - (long long)b * N);
    unsigned int ch = ((unsigned int)j * 2246822519u) ^ (unsigned int)seeds[b];
    int d = 0;
    for (int i = 0; i < N; ++i) {
        unsigned int h = ac_fmix32(((unsigned int)i * 2654435761u) ^ ch);
        if ((h & 0x00FFFFFFu) < (unsigned int)threshold) ++d;
    }
    out[idx] = (float)d < floor_ ? floor_ : (float)d;
}


// ---- substrate C: a scaled column's UNSCALED mass ---------------------
// `_scale_columns_now` sets  w[:, j] *= setpoint / mass_j  on winner columns
// each round, with setpoint = rows * p_fiber. Column scaling is per-COLUMN
// multiplicative and potentiation is per-CELL multiplicative, so they commute
// and the accumulated scale factors out of the sum:
//     mass_j = S_j * M_j,   M_j = sum_i present(i,j) * chain(count[i,j])
// hence  S_j^new = S_j^old * setpoint / (S_j^old * M_j) = setpoint / M_j.
// The old scale CANCELS, so the state is one float per column and the drive
// correction is a single elementwise multiply.
//
// That factorisation is exact only while the w_max CLIP never binds -- min()
// does not commute with a column multiply. The caller checks a conservative
// bound (tab[T] * max_j S_j) and refuses rather than silently diverging.
//
// One block per (b, winner), reduced over all n rows.
__global__ void colmass_kernel(const int* __restrict__ cols,
                               const long long* __restrict__ rowmask,
                               const long long* __restrict__ colmask,
                               const float* __restrict__ tab, int ntab,
                               const int* __restrict__ seeds,
                               int K, int N, int W, int threshold,
                               float* __restrict__ out,
                               float* __restrict__ outmax) {
    __shared__ float red[256];
    __shared__ float rmx[256];
    const int b = blockIdx.x / K, s = blockIdx.x % K;
    const int j = cols[(long long)b * K + s];
    const unsigned int ch =
        ((unsigned int)j * 2246822519u) ^ (unsigned int)seeds[b];

    float acc = 0.0f, mx = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        unsigned int h = ac_fmix32(((unsigned int)i * 2654435761u) ^ ch);
        if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) continue;
        int c = 0;
        for (int w = 0; w < W; ++w) {
            unsigned long long rm =
                (unsigned long long)rowmask[((long long)b * W + w) * N + i];
            if (rm == 0ULL) continue;
            c += __popcll(rm & (unsigned long long)
                          colmask[((long long)b * W + w) * N + j]);
        }
        float v = (c < ntab) ? tab[c] : tab[ntab - 1];
        acc += v;
        if (v > mx) mx = v;              // the DEEPEST cell in this column
    }
    red[threadIdx.x] = acc;
    rmx[threadIdx.x] = mx;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            red[threadIdx.x] += red[threadIdx.x + st];
            if (rmx[threadIdx.x + st] > rmx[threadIdx.x])
                rmx[threadIdx.x] = rmx[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(long long)b * K + s] = red[0];
        outmax[(long long)b * K + s] = rmx[0];
    }
}


// ---- potentiation correction, CSR form --------------------------------
// [[DRIVE-SPLIT]]: the correction is a sparse matvec over D restricted to the
// |S| = k active rows, so its intrinsic cost is the number of stored
// deviations in those rows. The bitmask form below cannot say WHICH cells are
// nonzero, so it must visit every (row, column) pair: O(k n W) against this
// O(sum nnz_i), a ratio n^2 T / (64 k^2) that is independent of M -- 8894x at
// n=16000, k=60, T=8.
//
// The store is ONE globally sorted key array over all brains, packed as
// b*n*n + i*n + j ([[HEBB-OUTER-PRODUCT]] gives the counts). A row is a
// contiguous run, found by two binary searches.
//
// One BLOCK per (brain, active row); threads split that row's cells. B*k
// blocks is ample parallelism and the walk is coalesced within a row.
__device__ __forceinline__ long long lb(const long long* __restrict__ a,
                                        long long n, long long v) {
    long long lo = 0, hi = n;
    while (lo < hi) {
        long long mid = (lo + hi) >> 1;
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void dev_csr_kernel(const int* __restrict__ S,
                               const long long* __restrict__ keys,
                               const int* __restrict__ cnts,
                               const long long* __restrict__ offs, int nruns,
                               const float* __restrict__ tab, int ntab,
                               const int* __restrict__ seeds,
                               int K, int N, int threshold,
                               float* __restrict__ out) {
    const int b = blockIdx.x / K, s = blockIdx.x - b * K;
    const int i = S[(long long)b * K + s];
    const long long base = (long long)b * N * N + (long long)i * N;
    const unsigned int ch = ((unsigned int)i * 2654435761u)
                            ^ (unsigned int)seeds[b];
    float* ob = out + (long long)b * N;
    // THE STORE IS A SET OF SORTED RUNS of geometrically increasing size, not
    // one sorted array. Re-sorting the whole store on every episode is O(M^2)
    // over a study -- 15e9 sorted elements at M=255, B=16 against 0.94e9 for
    // O(M log M). A run is searched exactly like the single array was; there
    // are only ~log2(M) of them.
    for (int r = 0; r < nruns; ++r) {
        const long long a = offs[r], z = offs[r + 1];
        const long long lo = a + lb(keys + a, z - a, base);
        const long long hi = a + lb(keys + a, z - a, base + N);
        for (long long e = lo + threadIdx.x; e < hi; e += blockDim.x) {
            const int j = (int)(keys[e] - base);
            const int c = cnts[e];
            const unsigned int h =
                ac_fmix32(((unsigned int)j * 2246822519u) ^ ch);
            if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) continue;
            const float v = (c < ntab) ? tab[c] : tab[ntab - 1];
            atomicAdd(ob + j, v - 1.0f);
        }
    }
}

torch::Tensor hashed_drive(torch::Tensor rows, torch::Tensor seeds,
                           int64_t n, int64_t threshold) {
    TORCH_CHECK(rows.dim() == 2 && rows.is_cuda()
                && rows.scalar_type() == torch::kInt32, "rows: [B,K] i32 cuda");
    rows = rows.contiguous();
    seeds = seeds.contiguous();
    const int B = rows.size(0), K = rows.size(1);
    auto out = torch::empty({B, (int64_t)n},
                            torch::dtype(torch::kFloat32).device(rows.device()));
    const long long tot = (long long)B * n;
    const int th = 256;
    hashed_drive_kernel<<<(tot + th - 1) / th, th>>>(
        rows.data_ptr<int>(), seeds.data_ptr<int>(), B, K, (int)n,
        (int)threshold, out.data_ptr<float>());
    return out;
}


torch::Tensor hashed_indegree(torch::Tensor seeds, int64_t n, int64_t threshold, double floor_);
std::vector<torch::Tensor> column_mass(torch::Tensor cols, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor tab, torch::Tensor seeds, int64_t threshold);
void dev_correct(torch::Tensor S, torch::Tensor rowmask, torch::Tensor colids,
                 torch::Tensor colmask, torch::Tensor tab,
                 torch::Tensor seeds, int64_t threshold, torch::Tensor out) {
    S = S.contiguous(); rowmask = rowmask.contiguous();
    colids = colids.contiguous(); colmask = colmask.contiguous();
    tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = S.size(0), K = S.size(1), C = colids.size(1);
    const int N = out.size(1);
    if (C == 0 || K == 0) return;
    const long long tot = (long long)B * K * C;
    const int th = 256;
    dev_correct_kernel<<<(tot + th - 1) / th, th>>>(
        S.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
        colids.data_ptr<int>(), colmask.data_ptr<int64_t>(),
        tab.data_ptr<float>(), (int)tab.numel(), seeds.data_ptr<int>(),
        B, K, C, N, (int)(rowmask.numel() / ((long long)B * N)),
        (int)threshold, out.data_ptr<float>());
}


torch::Tensor hashed_indegree(torch::Tensor seeds, int64_t n,
                              int64_t threshold, double floor_) {
    seeds = seeds.contiguous();
    const int B = seeds.size(0);
    auto out = torch::empty({B, (int64_t)n},
                            torch::dtype(torch::kFloat32).device(seeds.device()));
    const long long tot = (long long)B * n;
    const int th = 256;
    indegree_kernel<<<(tot + th - 1) / th, th>>>(
        seeds.data_ptr<int>(), B, (int)n, (int)threshold, (float)floor_,
        out.data_ptr<float>());
    return out;
}


std::vector<torch::Tensor> column_mass(torch::Tensor cols,
                          torch::Tensor rowmask,
                          torch::Tensor colmask, torch::Tensor tab,
                          torch::Tensor seeds, int64_t threshold) {
    cols = cols.contiguous(); rowmask = rowmask.contiguous();
    colmask = colmask.contiguous(); tab = tab.contiguous();
    seeds = seeds.contiguous();
    const int B = cols.size(0), K = cols.size(1);
    const int N = (int)colmask.size(-1);
    const int W = (int)(rowmask.numel() / ((long long)B * N));
    auto opt = torch::dtype(torch::kFloat32).device(cols.device());
    auto out = torch::empty({B, (int64_t)K}, opt);
    auto omax = torch::empty({B, (int64_t)K}, opt);
    colmass_kernel<<<B * K, 256>>>(
        cols.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
        colmask.data_ptr<int64_t>(), tab.data_ptr<float>(),
        (int)tab.numel(), seeds.data_ptr<int>(), K, N, W, (int)threshold,
        out.data_ptr<float>(), omax.data_ptr<float>());
    return {out, omax};
}


void dev_correct_csr(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts,
                     torch::Tensor offs, torch::Tensor tab,
                     torch::Tensor seeds, int64_t threshold,
                     torch::Tensor out) {
    S = S.contiguous(); keys = keys.contiguous(); cnts = cnts.contiguous();
    offs = offs.contiguous(); tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = S.size(0), K = S.size(1), N = out.size(1);
    if (K == 0 || keys.numel() == 0) return;
    dev_csr_kernel<<<B * K, 128>>>(
        S.data_ptr<int>(), keys.data_ptr<int64_t>(), cnts.data_ptr<int>(),
        offs.data_ptr<int64_t>(), (int)offs.numel() - 1,
        tab.data_ptr<float>(), (int)tab.numel(),
        seeds.data_ptr<int>(), K, N, (int)threshold, out.data_ptr<float>());
}

std::vector<torch::Tensor> topk_select(torch::Tensor x, int64_t K) {
    TORCH_CHECK(x.dim() == 2 && x.is_cuda()
                && x.scalar_type() == torch::kFloat32, "x: [B,N] f32 cuda");
    x = x.contiguous();
    const int B = x.size(0), N = x.size(1);
    TORCH_CHECK(N <= 65536, "n must be <= 65536: the key packs a 16-bit index");
    TORCH_CHECK(K <= CAPS, "k must fit the shared candidate buffer");
    auto iopt = torch::dtype(torch::kInt32).device(x.device());
    auto out = torch::empty({B, (int64_t)K}, iopt);
    auto ovf = torch::zeros({B}, iopt);
    select_kernel<<<B, NTH>>>(x.data_ptr<float>(), N, (int)K,
                              out.data_ptr<int>(), ovf.data_ptr<int>());
    return {out, ovf};
}
'''

_CPP = r"""
torch::Tensor hashed_drive(torch::Tensor rows, torch::Tensor seeds, int64_t n, int64_t threshold);
torch::Tensor hashed_indegree(torch::Tensor seeds, int64_t n, int64_t threshold, double floor_);
std::vector<torch::Tensor> column_mass(torch::Tensor cols, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor tab, torch::Tensor seeds, int64_t threshold);
void dev_correct(torch::Tensor S, torch::Tensor rowmask, torch::Tensor colids, torch::Tensor colmask, torch::Tensor tab, torch::Tensor seeds, int64_t threshold, torch::Tensor out);
void dev_correct_csr(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts, torch::Tensor offs, torch::Tensor tab, torch::Tensor seeds, int64_t threshold, torch::Tensor out);
std::vector<torch::Tensor> topk_select(torch::Tensor x, int64_t K);
"""

# Visual Studio ships ninja, but only puts it on PATH inside a developer shell.
# torch's `load_inline` needs it importable OR on PATH; adding the standard
# locations is a best effort that costs nothing when they are absent.
_VS_NINJA = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE"
    r"\CommonExtensions\Microsoft\CMake\Ninja",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE"
    r"\CommonExtensions\Microsoft\CMake\Ninja",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE"
    r"\CommonExtensions\Microsoft\CMake\Ninja",
]


def _augment_path() -> None:
    for d in _VS_NINJA:
        if os.path.isdir(d) and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def load() -> object | None:
    """Build (once) and return the fused module, or ``None`` if unavailable.

    Never raises: a missing toolchain is a capability question, not an error.
    The first failure's message is kept in :func:`last_error` for diagnosis --
    silently returning ``None`` with no way to ask why is how a GPU path
    quietly stops being used.
    """
    global _MODULE, _TRIED, _ERROR
    with _LOCK:
        if _TRIED:
            return _MODULE
        _TRIED = True
        try:
            import torch
            if not torch.cuda.is_available():
                _ERROR = "no CUDA device"
                return None
            from torch.utils.cpp_extension import load_inline
            _augment_path()
            _MODULE = load_inline(
                name="na_fused_cuda", cpp_sources=[_CPP],
                cuda_sources=[_CUDA_SRC],
                functions=["hashed_drive", "hashed_indegree", "dev_correct", "dev_correct_csr",
                                  "column_mass", "topk_select"],
                verbose=False, extra_cuda_cflags=["-O3"])
        except Exception as exc:                       # noqa: BLE001
            _MODULE = None
            _ERROR = f"{type(exc).__name__}: {exc}"
        return _MODULE


def available() -> bool:
    """True if the fused kernels built and can be called."""
    return load() is not None


def last_error() -> str | None:
    """Why :func:`load` returned ``None``, or ``None`` if it did not."""
    load()
    return _ERROR


def threshold_for(p: float) -> int:
    """The engine's integer Bernoulli threshold.

    ``_hash.hash_bernoulli_2d`` tests ``(h & 0xFFFFFF) < int(p * 2**24)``. This
    must be spelled the same way here; comparing ``(h & 0xFFFFFF) / 2**24 < p``
    instead flips the boundary cell whenever ``p * 2**24`` is not an integer.
    """
    return int(p * 16777216.0)
