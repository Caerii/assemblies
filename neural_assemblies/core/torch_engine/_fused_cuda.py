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


// ---- exact split-count correction -------------------------------------
// Potentiation is MULTIPLICATIVE, so the correction tab[c]-1 is NOT additive
// across a split count: a cell with c0 events in the store and c1 in the
// current episode needs tab[c0+c1]-1, and (tab[c0]-1)+(tab[c1]-1) is wrong by
// the cross term. COUNTS are additive ([[HEBB-OUTER-PRODUCT]]), so the exact
// scheme is: accumulate integer counts per active cell into a scratch
// [B, K, n], then apply tab ONCE per cell. Caught by engine parity at
// rel 2e-3 -- reference-based tests shared the flawed structure and passed.

__global__ void devcnt_csr_kernel(const int* __restrict__ S,
                                  const long long* __restrict__ keys,
                                  const int* __restrict__ cnts,
                                  const long long* __restrict__ offs,
                                  int nruns, int K, int N,
                                  int* __restrict__ scratch) {
    const int b = blockIdx.x / K, sl = blockIdx.x - b * K;
    const int i = S[(long long)b * K + sl];
    const long long base = (long long)b * N * N + (long long)i * N;
    int* sc = scratch + ((long long)b * K + sl) * N;
    for (int r = 0; r < nruns; ++r) {
        const long long a = offs[r], z = offs[r + 1];
        const long long lo = a + lb(keys + a, z - a, base);
        const long long hi = a + lb(keys + a, z - a, base + N);
        for (long long e = lo + threadIdx.x; e < hi; e += blockDim.x)
            atomicAdd(sc + (int)(keys[e] - base), cnts[e]);
    }
}

__global__ void devcnt_mask_kernel(const int* __restrict__ S,
                                   const long long* __restrict__ rowmask,
                                   const long long* __restrict__ colmask,
                                   int B, int K, int N, int W,
                                   int* __restrict__ scratch) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K * N) return;
    const int j = (int)(idx % N);
    const long long q = idx / N;
    const int sl = (int)(q % K), b = (int)(q / K);
    const int i = S[(long long)b * K + sl];
    int c = 0;
    for (int w = 0; w < W; ++w) {
        unsigned long long rm =
            (unsigned long long)rowmask[((long long)b * W + w) * N + i];
        if (rm == 0ULL) continue;
        c += __popcll(rm & (unsigned long long)
                      colmask[((long long)b * W + w) * N + j]);
    }
    if (c) atomicAdd(scratch + idx, c);
}

__global__ void devapply_kernel(const int* __restrict__ S,
                                const int* __restrict__ scratch,
                                const float* __restrict__ tab, int ntab,
                                const int* __restrict__ seeds,
                                int B, int K, int N, int threshold,
                                float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K * N) return;
    const int c = scratch[idx];
    if (c == 0) return;
    const int j = (int)(idx % N);
    const long long q = idx / N;
    const int sl = (int)(q % K), b = (int)(q / K);
    const int i = S[(long long)b * K + sl];
    const unsigned int h = ac_fmix32(((unsigned int)i * 2654435761u)
                                     ^ (((unsigned int)j * 2246822519u)
                                        ^ (unsigned int)seeds[b]));
    if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) return;   // absent
    const float v = (c < ntab) ? tab[c] : tab[ntab - 1];
    atomicAdd(out + (long long)b * N + j, v - 1.0f);
}


// ---- exact column mass for substrate C --------------------------------
// mass_j = SUM_i present(i,j) * tab[count_total(i,j)] needs the TOTAL count
// per cell, and the correction's lesson applies verbatim: counts are additive,
// tab is not. The store is ROW-keyed (b*n*n + i*n + j), so a column query
// cannot binary-search it; instead ONE linear pass over the store filters by
// a column -> slot map. Linear and coalesced, once per rescale.

__global__ void colcnt_store_kernel(const long long* __restrict__ keys,
                                    const int* __restrict__ cnts,
                                    long long nnz,
                                    const int* __restrict__ colmap,
                                    int K, int N,
                                    int* __restrict__ scratch) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    const long long key = keys[e];
    const int b = (int)(key / ((long long)N * N));
    const long long r = key - (long long)b * N * N;
    const int i = (int)(r / N), j = (int)(r - (long long)i * N);
    const int slot = colmap[(long long)b * N + j];
    if (slot < 0) return;
    atomicAdd(scratch + ((long long)b * K + slot) * N + i, cnts[e]);
}

__global__ void colcnt_mask_kernel(const int* __restrict__ cols,
                                   const long long* __restrict__ rowmask,
                                   const long long* __restrict__ colmask,
                                   int B, int K, int N, int W,
                                   int* __restrict__ scratch) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K * N) return;
    const int i = (int)(idx % N);
    const long long q = idx / N;
    const int sl = (int)(q % K), b = (int)(q / K);
    const int j = cols[(long long)b * K + sl];
    int c = 0;
    for (int w = 0; w < W; ++w) {
        unsigned long long rm =
            (unsigned long long)rowmask[((long long)b * W + w) * N + i];
        if (rm == 0ULL) continue;
        c += __popcll(rm & (unsigned long long)
                      colmask[((long long)b * W + w) * N + j]);
    }
    if (c) atomicAdd(scratch + idx, c);
}

__global__ void colmass_apply_kernel(const int* __restrict__ cols,
                                     const int* __restrict__ scratch,
                                     const float* __restrict__ tab, int ntab,
                                     const int* __restrict__ seeds,
                                     int K, int N, int threshold,
                                     float* __restrict__ out,
                                     float* __restrict__ outmax) {
    __shared__ float red[256];
    __shared__ float rmx[256];
    const int b = blockIdx.x / K, sl = blockIdx.x - b * K;
    const int j = cols[(long long)b * K + sl];
    const unsigned int ch = ((unsigned int)j * 2246822519u)
                            ^ (unsigned int)seeds[b];
    const int* sc = scratch + ((long long)b * K + sl) * N;
    float acc = 0.0f, mx = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const unsigned int h =
            ac_fmix32(((unsigned int)i * 2654435761u) ^ ch);
        if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) continue;
        const int c = sc[i];
        const float v = (c > 0) ? ((c < ntab) ? tab[c] : tab[ntab - 1]) : 1.0f;
        acc += v;
        if (v > mx) mx = v;
    }
    red[threadIdx.x] = acc; rmx[threadIdx.x] = mx;
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
        out[(long long)b * K + sl] = red[0];
        outmax[(long long)b * K + sl] = rmx[0];
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



// ---- MAX-RELATIVE pricing (column scaling, no clip) ----------------------
// With column scaling and no clip a weight is base * (1+beta)^c * s_j with
// s_j a per-column scalar, so a column is a SHARE distribution and only count
// DIFFERENCES within it matter. (1+beta)^c overflows float32 near c ~ 900,
// which a long training run reaches routinely; (1+beta)^(c - cmax_j) lies in
// (0, 1]. `rel[d]` = (1+beta)^(-d). The drive into column j from active rows
// S is then  s_j * ( rel[cmax_j] * base_j(S) + SUM_touched base_ij *
// (rel[cmax_j - c_ij] - rel[cmax_j]) ), and this kernel adds the second term
// onto an `out` that already holds rel[cmax_j] * base_j(S).
__global__ void devapply_rel_kernel(const int* __restrict__ S,
                                    const int* __restrict__ scratch,
                                    const float* __restrict__ rel, int nrel,
                                    const int* __restrict__ cmax,
                                    const int* __restrict__ seeds,
                                    int B, int K, int N, int threshold,
                                    float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K * N) return;
    const int c = scratch[idx];
    if (c == 0) return;
    const int j = (int)(idx % N);
    const long long q = idx / N;
    const int sl = (int)(q % K), b = (int)(q / K);
    const int i = S[(long long)b * K + sl];
    const unsigned int h = ac_fmix32(((unsigned int)i * 2654435761u)
                                     ^ (((unsigned int)j * 2246822519u)
                                        ^ (unsigned int)seeds[b]));
    if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) return;   // absent
    const int cm = cmax[(long long)b * N + j];
    const int d0 = cm, d1 = cm - c;                 // d1 >= 0 by construction
    const float r0 = (d0 < nrel) ? rel[d0] : 0.0f;
    const float r1 = (d1 < nrel) ? rel[d1] : 0.0f;
    atomicAdd(out + (long long)b * N + j, r1 - r0);
}

// mass'_j = SUM_i present(i,j) * rel[cmax_j - c_ij], with cmax_j found in a
// first pass over the same column. Returns the mass and cmax per winner col.
__global__ void colmass_rel_kernel(const int* __restrict__ cols,
                                   const int* __restrict__ scratch,
                                   const float* __restrict__ rel, int nrel,
                                   const int* __restrict__ seeds,
                                   int K, int N, int threshold,
                                   float* __restrict__ out,
                                   int* __restrict__ outmax) {
    __shared__ float red[256];
    __shared__ int rmx[256];
    const int b = blockIdx.x / K, sl = blockIdx.x - b * K;
    const int j = cols[(long long)b * K + sl];
    const unsigned int ch = ((unsigned int)j * 2246822519u)
                            ^ (unsigned int)seeds[b];
    const int* sc = scratch + ((long long)b * K + sl) * N;
    int mx = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const unsigned int h =
            ac_fmix32(((unsigned int)i * 2654435761u) ^ ch);
        if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) continue;
        const int c = sc[i];
        if (c > mx) mx = c;
    }
    rmx[threadIdx.x] = mx;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st && rmx[threadIdx.x + st] > rmx[threadIdx.x])
            rmx[threadIdx.x] = rmx[threadIdx.x + st];
        __syncthreads();
    }
    const int cm = rmx[0];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const unsigned int h =
            ac_fmix32(((unsigned int)i * 2654435761u) ^ ch);
        if ((h & 0x00FFFFFFu) >= (unsigned int)threshold) continue;
        const int d = cm - sc[i];
        acc += (d < nrel) ? rel[d] : 0.0f;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(long long)b * K + sl] = red[0];
        outmax[(long long)b * K + sl] = cm;
    }
}


// ---- DENSE cross fiber (DESIGN_dense_cross_fiber.md, DESIGN_dense_floor.md)
// Per-brain int16 count matrix C[b, i, j], per-column cmax and scale, and the
// connectome as a PRESENCE BITMASK pres[b, i, j/32] built ONCE by the same
// hash the store fiber tests at apply time: a warp reads one word for 32
// columns of a row instead of hashing 32 times. Prices are MAX-RELATIVE (see
// devapply_rel_kernel). The column mass is kept incrementally in float64 by
// the writer. Counts are int16: the writer flags a count that would pass
// 32767 in `err` rather than wrap.

#define DENSE_CMAX 32767

__global__ void presence_kernel(const int* __restrict__ seeds, int B, int Npre,
                                int N, int W, int threshold,
                                unsigned int* __restrict__ pres) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * Npre * W) return;
    const int w = (int)(idx % W);
    const long long bi = idx / W;
    const int i = (int)(bi % Npre), b = (int)(bi / Npre);
    const unsigned int seed = (unsigned int)seeds[b];
    const unsigned int ri = (unsigned int)i * 2654435761u;
    unsigned int word = 0u;
    for (int t = 0; t < 32; ++t) {
        const int j = (w << 5) + t;
        if (j >= N) break;
        const unsigned int ch = ((unsigned int)j * 2246822519u) ^ seed;
        const unsigned int h = ac_fmix32(ri ^ ch);
        if ((h & 0x00FFFFFFu) < (unsigned int)threshold) word |= (1u << t);
    }
    pres[idx] = word;
}

// d[b, j] += scale[b, j] * invdj[b, j] * SUM_{sl} present(i_sl, j) * rel[cmax_j - C[b, i_sl, j]]
__global__ void dense_drive_kernel(const int* __restrict__ S, int K,
                                   const short* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   const int* __restrict__ cmax,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ invdj,
                                   const float* __restrict__ rel, int nrel,
                                   int B, int Npre, int N,
                                   float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    const int j = (int)(idx % N), b = (int)(idx / N);
    const int cm = cmax[idx];
    const short* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const int wj = j >> 5, bj = j & 31;
    float acc = 0.0f;
    for (int sl = 0; sl < K; ++sl) {
        const int i = S[(long long)b * K + sl];
        if (i < 0) continue;
        if (!((Pb[(long long)i * W + wj] >> bj) & 1u)) continue;
        const int d = cm - (int)Cb[(long long)i * N + j];
        acc += (d < nrel) ? rel[d] : 0.0f;
    }
    float v = acc * scale[idx];
    if (invdj != nullptr) v *= invdj[idx];
    out[idx] += v;
}

// One block per (b, winner column j): counts the prev rows in, updates the
// column max, the relative mass (float64, incremental) and the scale.
__global__ void dense_write_kernel(const int* __restrict__ P, int KP,
                                   const int* __restrict__ Wn, int KW,
                                   short* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   int* __restrict__ cmax,
                                   double* __restrict__ mass,
                                   float* __restrict__ scale,
                                   const float* __restrict__ rel, int nrel,
                                   int Npre, int N,
                                   float setpoint, int do_scale,
                                   int* __restrict__ err) {
    __shared__ int rmx[128];
    __shared__ double red[128];
    const int b = blockIdx.x / KW, sw = blockIdx.x - b * KW;
    const int j = Wn[(long long)b * KW + sw];
    if (j < 0) return;
    short* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const int wj = j >> 5, bj = j & 31;
    const long long cidx = (long long)b * N + j;
    const int cm_old = cmax[cidx];
    // pass 1: increment, find the new column max among the written cells
    int mx = cm_old;
    for (int sl = threadIdx.x; sl < KP; sl += blockDim.x) {
        const int i = P[(long long)b * KP + sl];
        if (i < 0) continue;
        if (!((Pb[(long long)i * W + wj] >> bj) & 1u)) continue;
        const int c = (int)Cb[(long long)i * N + j] + 1;
        if (c > DENSE_CMAX) { atomicExch(err, 1); continue; }
        Cb[(long long)i * N + j] = (short)c;
        if (c > mx) mx = c;
    }
    rmx[threadIdx.x] = mx;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st && rmx[threadIdx.x + st] > rmx[threadIdx.x])
            rmx[threadIdx.x] = rmx[threadIdx.x + st];
        __syncthreads();
    }
    const int cm_new = rmx[0];
    if (!do_scale) { if (threadIdx.x == 0) cmax[cidx] = cm_new; return; }
    // pass 2: the written cells' change in relative price, at the NEW max
    double acc = 0.0;
    for (int sl = threadIdx.x; sl < KP; sl += blockDim.x) {
        const int i = P[(long long)b * KP + sl];
        if (i < 0) continue;
        if (!((Pb[(long long)i * W + wj] >> bj) & 1u)) continue;
        const int c = (int)Cb[(long long)i * N + j];     // already incremented
        const int dn = cm_new - c, dold = cm_new - (c - 1);
        const float rn = (dn >= 0 && dn < nrel) ? rel[dn] : 0.0f;
        const float ro = (dold >= 0 && dold < nrel) ? rel[dold] : 0.0f;
        acc += (double)rn - (double)ro;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int dc = cm_new - cm_old;
        const double shrink = (dc < nrel) ? (double)rel[dc] : 0.0;
        double m = mass[cidx] * shrink + red[0];
        mass[cidx] = m;
        cmax[cidx] = cm_new;
        scale[cidx] = (m > 1e-12) ? (float)((double)setpoint / m) : 1.0f;
    }
}


// ---- LAYER 3 (DESIGN_scheduled_training.md, DESIGN_dense_floor.md): the
// training loop on the device. One 256-thread block per brain -- three or
// four blocks share an SM.
//
// LATENCY, NOT BANDWIDTH. Measured by cycle counter: a drive that tested
// presence first and loaded a count only for present cells cost 84k cycles
// a round, the same as the hash version it replaced. A warp waits on a load
// whenever ANY of its 32 lanes has a present cell -- 81% of rows at p=0.05
// -- and the branch made the loads issue one at a time, so a round was ~160
// serialized memory round-trips per warp. The count loads are independent:
// issue them UNCONDITIONALLY, unrolled, ten in flight, and multiply by the
// presence bit. The bytes are the probe's bytes (the lines are touched
// either way); the time is one round-trip per ten rows. The price table is
// staged in shared memory so the dependent lookup costs a shared load.
//
// Shared memory holds the round's drive as 64-bit keys (float bits << 16 |
// (65535 - j)) -- the SAME key the histogram selector uses, so ties break
// identically. The k winners are the keys at or above the k-th largest,
// found by RADIX SELECT (8-bit digits from the top, a 256-bin histogram per
// pass -- double-buffered, so a pass is two barriers -- early exit when a
// bin's count equals the remainder); keys are unique, so exactly k qualify,
// and the winner set is the sort's first k. Winners are collected UNORDERED:
// the write is per column and commutative. The write is dense_write_kernel's
// arithmetic, one thread per winner column, its loads pipelined the same way.
//
// HOW THE LOADS ARE KEPT IN FLIGHT. nvcc sinks a load whose only use is under
// a condition INTO that condition (it saw "if (present) use(c)" and made the
// load conditional again, serializing them -- read the SASS, not the source).
// So: a chunk of SCHED_CH counts is loaded into registers by one unrolled
// loop, and consumed by a second whose adds are SELECTS, `acc += present ?
// v : 0`, so every loaded value has an unconditional use. `x + 0.0f == x`
// bit-for-bit here: acc is a sum of non-negative prices from +0, never -0.
#define SCHED_MAXK 128
#define SCHED_TH 256
#define SCHED_RELSH 2048          // staged price entries (the rest are read from global)
#define SCHED_CH 10               // count loads in flight per thread

// rel[d], from the staged head when it is there; 0 past the table; d < 0 is
// not a valid depth and prices at 0 (the write's guard).
__device__ __forceinline__ float sched_price(int d, const float* srel, int nsh,
                                             const float* __restrict__ rel, int nrel) {
    float v = srel[(d >= 0 && d < nsh) ? d : 0];
    if (d < 0 || d >= nsh) v = (d >= 0 && d < nrel) ? rel[d] : 0.0f;
    return v;
}

__device__ __forceinline__ unsigned long long sched_key(float v, int j) {
    unsigned int u = __float_as_uint(v);
    // map float to an order-preserving unsigned key
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 16) | (unsigned long long)(65535 - (j & 0xFFFF));
}

// The KW-th largest of keys[0..N) (unique keys). Block-wide; every thread
// returns the same threshold. `hist`: 2 x 256 ints, both ZERO at entry and
// left zero at exit; `sh`: 2 x 3 ints. Two barriers per pass.
//
// `kand`/`kor` are the AND and OR of every key's float half: the bits where
// they agree are shared by ALL keys and need no pass. Positive drives within
// a few binades share sign and most of the exponent, so the first digit
// starts ~9 bits down and the float's 32 bits are decided in three passes
// instead of four; identical floats (rare, jitter) fall through to the
// column bits like any other tie.
__device__ unsigned long long sched_select(const unsigned long long* keys, int N,
                                           int KW, int* hist, int* sh,
                                           unsigned int kand, unsigned int kor) {
    const int lead = (kand ^ kor) ? __clz(kand ^ kor) : 32;      // shared top bits
    const unsigned long long lmask = lead ? (~0ull << (64 - lead)) : 0ull;
    unsigned long long prefix = ((unsigned long long)kand << 32) & lmask, pmask = lmask;
    int rem = KW, pass = 0, last = 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int shift = 56 - lead; if (shift < 0) shift = 0;
    // the last digit clamps to bit 0 (overlapping decided bits is harmless:
    // they are equal in every valid key) so the column bits are always covered
    for (; shift >= 0; shift = (shift >= 8) ? shift - 8 : (shift > 0 ? 0 : -1), ++pass) {
        int* h = hist + (pass & 1) * 256;
        int* out = sh + (pass & 1) * 3;
        for (int base = 0; base < N; base += blockDim.x) {
            const int j = base + threadIdx.x;
            unsigned long long key = 0ull;
            bool valid = false;
            if (j < N) { key = keys[j]; valid = ((key & pmask) == prefix); }
            const unsigned int d = valid ? (unsigned int)((key >> shift) & 0xFFull) : 0u;
            const unsigned int act = __ballot_sync(0xFFFFFFFFu, valid);
            if (valid) {
                // one shared atomic per distinct digit per warp
                const unsigned int m = __match_any_sync(act, d);
                if ((__ffs(m) - 1) == lane) atomicAdd(&h[d], __popc(m));
            }
        }
        __syncthreads();
        if (warp == 0) {
            // lane l owns bins 255-8l .. 248-8l; suffix scan from the top
            int s = 0;
#pragma unroll
            for (int t = 0; t < 8; ++t) s += h[255 - 8 * lane - t];
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
                if (lane >= o) incl += v;
            }
            const int excl = incl - s;
            const bool here = (excl < rem) && (rem <= incl);
            const unsigned int bal = __ballot_sync(0xFFFFFFFFu, here);
            if (lane == (__ffs(bal) - 1)) {
                int acc = excl;
                for (int t = 0; t < 8; ++t) {
                    const int bin = 255 - 8 * lane - t;
                    const int c = h[bin];
                    if (acc + c >= rem) { out[0] = bin; out[1] = rem - acc; out[2] = c; break; }
                    acc += c;
                }
            }
        } else {
            // meanwhile: zero the other buffer for the next pass
            int* o = hist + ((pass + 1) & 1) * 256;
            for (int t = threadIdx.x - 32; t < 256; t += blockDim.x - 32) o[t] = 0;
        }
        __syncthreads();
        prefix |= ((unsigned long long)out[0]) << shift;
        pmask |= 0xFFull << shift;
        rem = out[1];
        last = pass;
        if (out[2] == rem) break;       // every key with this prefix wins
    }
    // leave the dirty buffer zero for the next call (no reader remains)
    for (int t = threadIdx.x; t < 256; t += blockDim.x) hist[(last & 1) * 256 + t] = 0;
    return prefix;
}

__global__ void sched_train_kernel(const long long* __restrict__ words,
                                   const long long* __restrict__ bundles, int S,
                                   const long long* __restrict__ lex_cache, int V, int K,
                                   const float* __restrict__ bundle_drive,
                                   const float* __restrict__ jit, int I,
                                   short* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   int* __restrict__ cmax,
                                   double* __restrict__ mass, float* __restrict__ scale,
                                   const float* __restrict__ invdj,
                                   const float* __restrict__ rel, int nrel, int nsh,
                                   int Npre, int N,
                                   float setpoint, int rounds, int KW,
                                   int* __restrict__ err, int sms) {
    const long long t_start = clock64();
    const int q = blockIdx.x / sms, nq = (gridDim.x + sms - 1) / sms;
    extern __shared__ unsigned long long keys[];          // N keys, then ...
    unsigned int* spres = reinterpret_cast<unsigned int*>(keys + N);   // K x W words
    float* srel = reinterpret_cast<float*>(spres + K * W);             // nsh prices
    short* wcnt = reinterpret_cast<short*>(srel + nsh);                // KW x K counts
    __shared__ int hist[512];
    __shared__ int sh[6];
    __shared__ int rows[SCHED_MAXK];
    __shared__ int win[SCHED_MAXK];
    __shared__ int colmax[SCHED_MAXK];
    __shared__ unsigned int kbits[2];
    __shared__ int nwin;
    const int nch = (K + SCHED_CH - 1) / SCHED_CH;
    if (threadIdx.x == 0) { kbits[0] = 0xFFFFFFFFu; kbits[1] = 0u; }
    const int b = blockIdx.x;
    short* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const long long cbase = (long long)b * N;
    for (int t = threadIdx.x; t < 512; t += blockDim.x) hist[t] = 0;
    for (int t = threadIdx.x; t < nsh; t += blockDim.x) srel[t] = rel[t];
    for (int s = 0; s < S; ++s) {
        const long long w = words[(long long)b * S + s];
        const long long bid = bundles[(long long)b * S + s];
        if (w < 0 || bid < 0) break;
        if (threadIdx.x < K) {
            long long r = lex_cache[((long long)b * V + w) * K + threadIdx.x];
            rows[threadIdx.x] = (int)r;
        }
        __syncthreads();
        // LOCALITY: the step's rows are fixed for its rounds, so their
        // presence words are staged in shared memory once (coalesced, a
        // row's words are contiguous) and tested there by drive and write.
        for (int t = threadIdx.x; t < K * W; t += blockDim.x) {
            const int i = rows[t / W];
            spres[t] = (i >= 0) ? Pb[(long long)i * W + (t % W)] : 0u;
        }
        __syncthreads();
        const float* stim = bundle_drive + ((long long)b * I + bid) * N;
        const float* jt = jit + ((long long)b * I + bid) * N;
        for (int r = 0; r < rounds; ++r) {
            // drive over columns -> keys
            unsigned int kand = 0xFFFFFFFFu, kor = 0u;
            for (int j = threadIdx.x; j < N; j += blockDim.x) {
                const int cm = cmax[cbase + j];
                const int wj = j >> 5, bj = j & 31;
                float acc = 0.0f;
                for (int s0 = 0; s0 < K; s0 += SCHED_CH) {
                    int cs[SCHED_CH];
#pragma unroll
                    for (int u = 0; u < SCHED_CH; ++u) {           // SCHED_CH loads in flight
                        const int sl = s0 + u;
                        const int i = (sl < K) ? rows[sl] : 0;
                        cs[u] = (sl < K) ? (int)Cb[(long long)(i < 0 ? 0 : i) * N + j] : 0;
                    }
#pragma unroll
                    for (int u = 0; u < SCHED_CH; ++u) {
                        const int sl = s0 + u;
                        const unsigned int pw = (sl < K) ? spres[sl * W + wj] : 0u;
                        const float v = sched_price(cm - cs[u], srel, nsh, rel, nrel);
                        acc += ((pw >> bj) & 1u) ? v : 0.0f;
                    }
                }
                float v = acc * scale[cbase + j];
                if (invdj != nullptr) v *= invdj[cbase + j];
                // the python path: d = stim; d += v; ranked = d + jit
                const float dd = stim[j] + v;
                const unsigned long long key = sched_key(dd + jt[j], j);
                keys[j] = key;
                kand &= (unsigned int)(key >> 32); kor |= (unsigned int)(key >> 32);
            }
            kand = __reduce_and_sync(0xFFFFFFFFu, kand);
            kor = __reduce_or_sync(0xFFFFFFFFu, kor);
            if ((threadIdx.x & 31) == 0) { atomicAnd(&kbits[0], kand); atomicOr(&kbits[1], kor); }
            if (threadIdx.x == 0) nwin = 0;
            if (threadIdx.x < KW) colmax[threadIdx.x] = 0;
            __syncthreads();
            const unsigned long long T = sched_select(keys, N, KW, hist, sh, kbits[0], kbits[1]);
            for (int j = threadIdx.x; j < N; j += blockDim.x) {
                if (keys[j] >= T) {
                    const int pos = atomicAdd(&nwin, 1);
                    if (pos < SCHED_MAXK) win[pos] = j;
                }
            }
            __syncthreads();
            if (nwin != KW) { if (threadIdx.x == 0) atomicExch(err, 2); return; }
            // every thread has read kbits (before the barrier above); reset
            // for the next drive, two barriers away
            if (threadIdx.x == 0) { kbits[0] = 0xFFFFFFFFu; kbits[1] = 0u; }
            // write, phase A (the whole block): (column, chunk) per thread --
            // KW x nch of them -- loads in flight, present cells incremented
            // and stored, the new counts staged in shared, column maxima by
            // shared atomicMax (commutative, so the old max joins unordered)
            bool over = false;
            for (int t = threadIdx.x; t < KW * nch; t += blockDim.x) {
                const int col = t / nch, s0 = (t - col * nch) * SCHED_CH;
                const int j = win[col];
                const int wj = j >> 5, bj = j & 31;
                int mx = (s0 == 0) ? cmax[cbase + j] : 0;
                int cs[SCHED_CH];
#pragma unroll
                for (int u = 0; u < SCHED_CH; ++u) {
                    const int sl = s0 + u;
                    const int i = (sl < K) ? rows[sl] : 0;
                    cs[u] = (sl < K) ? (int)Cb[(long long)(i < 0 ? 0 : i) * N + j] : 0;
                }
#pragma unroll
                for (int u = 0; u < SCHED_CH; ++u) {
                    const int sl = s0 + u;
                    const int i = (sl < K) ? rows[sl] : 0;
                    const unsigned int pw = (sl < K) ? spres[sl * W + wj] : 0u;
                    const bool present = (pw >> bj) & 1u;
                    const int c = cs[u] + 1;
                    const bool ok = present && c <= DENSE_CMAX;
                    over |= present && !ok;
                    if (ok) Cb[(long long)(i < 0 ? 0 : i) * N + j] = (short)c;
                    if (sl < K) wcnt[col * K + sl] = (short)(ok ? c : cs[u]);
                    mx = ok ? (c > mx ? c : mx) : mx;
                }
                atomicMax(&colmax[col], mx);
            }
            if (over) atomicExch(err, 1);
            __syncthreads();
            // write, phase B (one thread per winner column): the change in
            // relative price at the NEW max, summed in row order from the
            // staged counts, then mass, max and scale -- dense_write_kernel's
            // arithmetic
            if (threadIdx.x < KW) {
                const int j = win[threadIdx.x];
                const int wj = j >> 5, bj = j & 31;
                const int cm_old = cmax[cbase + j];
                const int cm_new = colmax[threadIdx.x];
                const short* wc = wcnt + threadIdx.x * K;
                double acc = 0.0;
                for (int sl = 0; sl < K; ++sl) {
                    if (!((spres[sl * W + wj] >> bj) & 1u)) continue;
                    const int c = (int)wc[sl];
                    const float rn = sched_price(cm_new - c, srel, nsh, rel, nrel);
                    const float ro = sched_price(cm_new - (c - 1), srel, nsh, rel, nrel);
                    acc += (double)rn - (double)ro;
                }
                const int dc = cm_new - cm_old;
                const double shrink = (dc < nrel) ? (double)rel[dc] : 0.0;
                const double m = mass[cbase + j] * shrink + acc;
                mass[cbase + j] = m;
                cmax[cbase + j] = cm_new;
                scale[cbase + j] = (m > 1e-12) ? (float)((double)setpoint / m) : 1.0f;
            }
            __syncthreads();
            // DE-PHASE (once, after the first round): blocks run identical
            // schedules and stay in lockstep, so every block on an SM streams
            // its drive at once and then selects at once -- memory time and
            // compute time ADD. Block q of the `nq` sharing an SM waits q/nq
            // of its own measured first round, after which its select and
            // write overlap its neighbours' drives.
            if (s == 0 && r == 0 && nq > 1) {
                const long long t1 = clock64();
                const long long wait = ((t1 - t_start) * (long long)q) / nq;
                while (clock64() - t1 < wait) { }
                __syncthreads();
            }
        }
    }
}

// The ROOFLINE PROBE (DESIGN_dense_floor.md): the drive's reads -- K rows of
// int16 counts across N columns per round, the row set shifting each round
// so the lines are not the same ones -- and nothing else, in the training
// kernel's block shape. Its time is the streamed-lines floor the kernel is
// measured against.
__global__ void stream_probe_kernel(const short* __restrict__ C,
                                    const int* __restrict__ S, int K,
                                    int Npre, int N, int rounds,
                                    float* __restrict__ out) {
    __shared__ int rows[SCHED_MAXK];
    const int b = blockIdx.x;
    const short* Cb = C + (long long)b * Npre * N;
    float acc = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        if (threadIdx.x < K) rows[threadIdx.x] = (S[(long long)b * K + threadIdx.x] + r) % Npre;
        __syncthreads();
        for (int j = threadIdx.x; j < N; j += blockDim.x)
            for (int sl = 0; sl < K; ++sl)
                acc += (float)Cb[(long long)rows[sl] * N + j];
        __syncthreads();
    }
    out[(long long)b * blockDim.x + threadIdx.x] = acc;
}

void dev_correct_exact(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts,
                       torch::Tensor offs, torch::Tensor rowmask,
                       torch::Tensor colmask, torch::Tensor scratch,
                       torch::Tensor tab, torch::Tensor seeds,
                       int64_t threshold, torch::Tensor out) {
    S = S.contiguous(); tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = S.size(0), K = S.size(1), N = out.size(1);
    if (K == 0) return;
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        offs = offs.contiguous();
        devcnt_csr_kernel<<<B * K, 128>>>(
            S.data_ptr<int>(), keys.data_ptr<int64_t>(), cnts.data_ptr<int>(),
            offs.data_ptr<int64_t>(), (int)offs.numel() - 1, K, N,
            scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * rowmask.size(-1)));
        devcnt_mask_kernel<<<(tot + th - 1) / th, th>>>(
            S.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    devapply_kernel<<<(tot + th - 1) / th, th>>>(
        S.data_ptr<int>(), scratch.data_ptr<int>(), tab.data_ptr<float>(),
        (int)tab.numel(), seeds.data_ptr<int>(), B, K, N, (int)threshold,
        out.data_ptr<float>());
}


std::vector<torch::Tensor> column_mass_exact(
        torch::Tensor cols, torch::Tensor keys, torch::Tensor cnts,
        torch::Tensor colmap, torch::Tensor rowmask, torch::Tensor colmask,
        torch::Tensor scratch, torch::Tensor tab, torch::Tensor seeds,
        int64_t n, int64_t threshold) {
    cols = cols.contiguous(); tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = cols.size(0), K = cols.size(1), N = (int)n;
    auto opt = torch::dtype(torch::kFloat32).device(cols.device());
    auto out = torch::empty({B, (int64_t)K}, opt);
    auto omax = torch::empty({B, (int64_t)K}, opt);
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        colmap = colmap.contiguous();
        const long long nnz = keys.numel();
        colcnt_store_kernel<<<(nnz + th - 1) / th, th>>>(
            keys.data_ptr<int64_t>(), cnts.data_ptr<int>(), nnz,
            colmap.data_ptr<int>(), K, N, scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * N));
        colcnt_mask_kernel<<<(tot + th - 1) / th, th>>>(
            cols.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    colmass_apply_kernel<<<B * K, 256>>>(
        cols.data_ptr<int>(), scratch.data_ptr<int>(), tab.data_ptr<float>(),
        (int)tab.numel(), seeds.data_ptr<int>(), K, N, (int)threshold,
        out.data_ptr<float>(), omax.data_ptr<float>());
    return {out, omax};
}


void dev_correct_rel(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts,
                     torch::Tensor offs, torch::Tensor rowmask,
                     torch::Tensor colmask, torch::Tensor scratch,
                     torch::Tensor rel, torch::Tensor cmax, torch::Tensor seeds,
                     int64_t threshold, torch::Tensor out) {
    S = S.contiguous(); rel = rel.contiguous(); seeds = seeds.contiguous();
    cmax = cmax.contiguous();
    const int B = S.size(0), K = S.size(1), N = out.size(1);
    if (K == 0) return;
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        offs = offs.contiguous();
        devcnt_csr_kernel<<<B * K, 128>>>(
            S.data_ptr<int>(), keys.data_ptr<int64_t>(), cnts.data_ptr<int>(),
            offs.data_ptr<int64_t>(), (int)offs.numel() - 1, K, N,
            scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * rowmask.size(-1)));
        devcnt_mask_kernel<<<(tot + th - 1) / th, th>>>(
            S.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    devapply_rel_kernel<<<(tot + th - 1) / th, th>>>(
        S.data_ptr<int>(), scratch.data_ptr<int>(), rel.data_ptr<float>(),
        (int)rel.numel(), cmax.data_ptr<int>(), seeds.data_ptr<int>(),
        B, K, N, (int)threshold, out.data_ptr<float>());
}


std::vector<torch::Tensor> column_mass_rel(
        torch::Tensor cols, torch::Tensor keys, torch::Tensor cnts,
        torch::Tensor colmap, torch::Tensor rowmask, torch::Tensor colmask,
        torch::Tensor scratch, torch::Tensor rel, torch::Tensor seeds,
        int64_t n, int64_t threshold) {
    cols = cols.contiguous(); rel = rel.contiguous(); seeds = seeds.contiguous();
    const int B = cols.size(0), K = cols.size(1), N = (int)n;
    auto optf = torch::dtype(torch::kFloat32).device(cols.device());
    auto opti = torch::dtype(torch::kInt32).device(cols.device());
    auto out = torch::empty({B, (int64_t)K}, optf);
    auto omax = torch::empty({B, (int64_t)K}, opti);
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        colmap = colmap.contiguous();
        const long long nnz = keys.numel();
        colcnt_store_kernel<<<(nnz + th - 1) / th, th>>>(
            keys.data_ptr<int64_t>(), cnts.data_ptr<int>(), nnz,
            colmap.data_ptr<int>(), K, N, scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * N));
        colcnt_mask_kernel<<<(tot + th - 1) / th, th>>>(
            cols.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    colmass_rel_kernel<<<B * K, 256>>>(
        cols.data_ptr<int>(), scratch.data_ptr<int>(), rel.data_ptr<float>(),
        (int)rel.numel(), seeds.data_ptr<int>(), K, N, (int)threshold,
        out.data_ptr<float>(), omax.data_ptr<int>());
    return {out, omax};
}


torch::Tensor hashed_presence(torch::Tensor seeds, int64_t n_pre, int64_t n_post,
                              int64_t threshold) {
    seeds = seeds.contiguous();
    const int B = seeds.size(0), W = (int)((n_post + 31) / 32);
    auto out = torch::empty({B, n_pre, (int64_t)W},
                            torch::dtype(torch::kInt32).device(seeds.device()));
    const long long tot = (long long)B * n_pre * W;
    const int th = 256;
    presence_kernel<<<(tot + th - 1) / th, th>>>(
        seeds.data_ptr<int>(), B, (int)n_pre, (int)n_post, W, (int)threshold,
        reinterpret_cast<unsigned int*>(out.data_ptr<int>()));
    return out;
}

void dense_drive(torch::Tensor S, torch::Tensor C, torch::Tensor pres,
                 torch::Tensor cmax, torch::Tensor scale, torch::Tensor invdj,
                 torch::Tensor rel, torch::Tensor out) {
    S = S.contiguous(); rel = rel.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt16, "counts are int16");
    const int B = C.size(0), Npre = C.size(1), N = C.size(2), K = S.size(1);
    const int W = pres.size(2);
    if (K == 0) return;
    const long long tot = (long long)B * N;
    const int th = 256;
    dense_drive_kernel<<<(tot + th - 1) / th, th>>>(
        S.data_ptr<int>(), K, C.data_ptr<short>(),
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        cmax.data_ptr<int>(), scale.data_ptr<float>(),
        invdj.numel() ? invdj.data_ptr<float>() : nullptr,
        rel.data_ptr<float>(), (int)rel.numel(),
        B, Npre, N, out.data_ptr<float>());
}

void dense_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor C,
                 torch::Tensor pres, torch::Tensor cmax, torch::Tensor mass,
                 torch::Tensor scale, torch::Tensor rel, double setpoint,
                 int64_t do_scale, torch::Tensor err) {
    P = P.contiguous(); Wn = Wn.contiguous(); rel = rel.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt16, "counts are int16");
    const int B = C.size(0), Npre = C.size(1), N = C.size(2);
    const int W = pres.size(2);
    const int KP = P.size(1), KW = Wn.size(1);
    if (KP == 0 || KW == 0) return;
    dense_write_kernel<<<B * KW, 128>>>(
        P.data_ptr<int>(), KP, Wn.data_ptr<int>(), KW, C.data_ptr<short>(),
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        cmax.data_ptr<int>(), mass.data_ptr<double>(), scale.data_ptr<float>(),
        rel.data_ptr<float>(), (int)rel.numel(),
        Npre, N, (float)setpoint, (int)do_scale, err.data_ptr<int>());
}


void sched_train(torch::Tensor words, torch::Tensor bundles,
                 torch::Tensor lex_cache, torch::Tensor bundle_drive,
                 torch::Tensor jit, torch::Tensor C, torch::Tensor pres,
                 torch::Tensor cmax, torch::Tensor mass, torch::Tensor scale,
                 torch::Tensor invdj, torch::Tensor rel, double setpoint,
                 int64_t rounds, int64_t kw, torch::Tensor err, int64_t nsh) {
    words = words.contiguous(); bundles = bundles.contiguous();
    lex_cache = lex_cache.contiguous(); bundle_drive = bundle_drive.contiguous();
    jit = jit.contiguous(); rel = rel.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt16, "counts are int16");
    const int B = C.size(0), Npre = C.size(1), N = C.size(2);
    const int W = pres.size(2);
    const int S = words.size(1), V = lex_cache.size(1), K = lex_cache.size(2);
    const int I = bundle_drive.size(1);
    TORCH_CHECK(K <= SCHED_MAXK && kw <= SCHED_MAXK, "k too large for the block");
    TORCH_CHECK(N <= 8192, "N too large for shared-memory selection");
    TORCH_CHECK(nsh >= 0 && nsh <= SCHED_RELSH && nsh <= rel.numel(), "nsh");
    const size_t shm = (size_t)N * sizeof(unsigned long long)
                     + (size_t)K * W * sizeof(unsigned int)
                     + (size_t)nsh * sizeof(float)
                     + (size_t)kw * K * sizeof(short);
    TORCH_CHECK(shm <= 96 * 1024, "keys + staged presence + prices + counts exceed shared memory");
    cudaFuncSetAttribute(sched_train_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    int dev = 0, sms = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    sched_train_kernel<<<B, SCHED_TH, shm>>>(
        words.data_ptr<int64_t>(), bundles.data_ptr<int64_t>(), S,
        lex_cache.data_ptr<int64_t>(), V, K,
        bundle_drive.data_ptr<float>(), jit.data_ptr<float>(), I,
        C.data_ptr<short>(),
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        cmax.data_ptr<int>(), mass.data_ptr<double>(),
        scale.data_ptr<float>(),
        invdj.numel() ? invdj.data_ptr<float>() : nullptr,
        rel.data_ptr<float>(), (int)rel.numel(), (int)nsh,
        Npre, N, (float)setpoint, (int)rounds, (int)kw, err.data_ptr<int>(), sms);
}

torch::Tensor stream_probe(torch::Tensor C, torch::Tensor S, int64_t rounds) {
    S = S.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt16, "counts are int16");
    const int B = C.size(0), Npre = C.size(1), N = C.size(2), K = S.size(1);
    TORCH_CHECK(K <= SCHED_MAXK, "k too large for the block");
    auto out = torch::zeros({B, SCHED_TH},
                            torch::dtype(torch::kFloat32).device(C.device()));
    stream_probe_kernel<<<B, SCHED_TH>>>(
        C.data_ptr<short>(), S.data_ptr<int>(), K, Npre, N, (int)rounds,
        out.data_ptr<float>());
    return out;
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
void dev_correct_exact(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts, torch::Tensor offs, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor scratch, torch::Tensor tab, torch::Tensor seeds, int64_t threshold, torch::Tensor out);
std::vector<torch::Tensor> column_mass_exact(torch::Tensor cols, torch::Tensor keys, torch::Tensor cnts, torch::Tensor colmap, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor scratch, torch::Tensor tab, torch::Tensor seeds, int64_t n, int64_t threshold);
void dev_correct_rel(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts, torch::Tensor offs, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor scratch, torch::Tensor rel, torch::Tensor cmax, torch::Tensor seeds, int64_t threshold, torch::Tensor out);
std::vector<torch::Tensor> column_mass_rel(torch::Tensor cols, torch::Tensor keys, torch::Tensor cnts, torch::Tensor colmap, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor scratch, torch::Tensor rel, torch::Tensor seeds, int64_t n, int64_t threshold);
torch::Tensor hashed_presence(torch::Tensor seeds, int64_t n_pre, int64_t n_post, int64_t threshold);
void dense_drive(torch::Tensor S, torch::Tensor C, torch::Tensor pres, torch::Tensor cmax, torch::Tensor scale, torch::Tensor invdj, torch::Tensor rel, torch::Tensor out);
void dense_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor C, torch::Tensor pres, torch::Tensor cmax, torch::Tensor mass, torch::Tensor scale, torch::Tensor rel, double setpoint, int64_t do_scale, torch::Tensor err);
void sched_train(torch::Tensor words, torch::Tensor bundles, torch::Tensor lex_cache, torch::Tensor bundle_drive, torch::Tensor jit, torch::Tensor C, torch::Tensor pres, torch::Tensor cmax, torch::Tensor mass, torch::Tensor scale, torch::Tensor invdj, torch::Tensor rel, double setpoint, int64_t rounds, int64_t kw, torch::Tensor err, int64_t nsh);
torch::Tensor stream_probe(torch::Tensor C, torch::Tensor S, int64_t rounds);
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
                functions=["hashed_drive", "hashed_indegree", "dev_correct",
                           "dev_correct_csr", "dev_correct_exact",
                           "column_mass_exact", "dev_correct_rel",
                           "column_mass_rel", "hashed_presence", "dense_drive",
                           "dense_write", "sched_train", "stream_probe",
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
