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

// ORDER-PRESERVING: the raw float bits rank a NEGATIVE drive above every
// positive one (sign bit set = largest unsigned). Drives are non-negative
// everywhere but under refraction, where net = raw - bias, and there this
// selector was never gated: a refracted area's most-biased neurons kept
// winning, the bias grew without bound (66 against a drive of 0.6), and
// the sequence organ's arc collapsed onto one assembly (DESIGN_sequence_port.md).
__device__ __forceinline__ unsigned long long mkkey(float v, int j) {
    unsigned int u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 16) | (unsigned long long)(65535 - j);
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


// ---- PRESENCE BITMASK (DESIGN_dense_floor.md) ------------------------------
// The connectome as bits, pres[b, i, j/32], built ONCE by the same hash the
// store fiber tests at apply time. It is the hash's stored form (GATE-4) and
// the source of the present-only lists below. Counts are int16 everywhere:
// a writer flags a count that would pass 32767 in `err` rather than wrap.

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

#define SCHED_MAXK 128            // rows / winner columns a warp-per-brain kernel accepts

// rel[d]: from the staged head when it is there; from global for
// nsh <= d < nnz (nnz = the table's nonzero head -- it is monotone, and past
// it the price IS zero, so no load); d < 0 is not a valid depth and prices
// at 0 (the write's guard).
__device__ __forceinline__ float sched_price(int d, const float* srel, int nsh,
                                             const float* __restrict__ rel, int nnz) {
    float v = srel[(d >= 0 && d < nsh) ? d : 0];
    if (d < 0 || d >= nsh) v = (d >= 0 && d < nnz) ? rel[d] : 0.0f;
    return v;
}

__device__ __forceinline__ unsigned long long sched_key(float v, int j) {
    unsigned int u = __float_as_uint(v);
    // map float to an order-preserving unsigned key
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 16) | (unsigned long long)(65535 - (j & 0xFFFF));
}

// ---- DENSE ORGAN fiber (DESIGN_sequence_port.md) ---------------------------
// The organ's regime: organ_p ~ 0.2, k = 200, n to 50,000. Present-only
// lists do not fit (2,000-10,000 entries per row); at this density the count
// MATRIX does: int16 counts [n_pre, n_post] per brain, the connectome as the
// presence bitmask, ABSOLUTE pricing by the engine's chain table (clip
// included), norm_init, no column scaling. A thread per column sums its K
// rows IN ROW ORDER -- the store fiber's sequence -- with presence-
// predicated loads (a predicated-off load fetches nothing, the loads stay in
// flight: DESIGN_dense_floor.md lessons 2 and 7). The write is a block per
// winner column. Rows and winners of -1 are skipped: the dead-brain and
// per-brain-inhibit convention of the scheduled organ.
#define ORGAN_CH 8

__global__ void organ_drive_kernel(const int* __restrict__ S, int K,
                                   const short* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   const float* __restrict__ invdj,
                                   const float* __restrict__ tab, int ntab,
                                   int B, int Npre, int N, float* __restrict__ out) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    const int j = (int)(idx % N), b = (int)(idx / N);
    const short* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const int* Sb = S + (long long)b * K;
    const int wj = j >> 5, bj = j & 31;
    float acc = 0.0f;
    for (int s0 = 0; s0 < K; s0 += ORGAN_CH) {
        int cs[ORGAN_CH];
        unsigned int pm = 0u;
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int sl = s0 + u;
            const int i = (sl < K) ? Sb[sl] : -1;
            const unsigned int pw = (i >= 0) ? Pb[(long long)i * W + wj] : 0u;
            const bool pr = (pw >> bj) & 1u;
            pm |= (pr ? 1u : 0u) << u;
            cs[u] = pr ? (int)Cb[(long long)i * N + j] : 0;      // predicated
        }
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int c = cs[u] < ntab ? cs[u] : ntab - 1;       // the chain saturates at the clip
            acc += ((pm >> u) & 1u) ? tab[c] : 0.0f;
        }
    }
    float v = acc;
    if (invdj != nullptr) v *= invdj[idx];
    out[idx] += v;
}

// one block per (brain, winner column): count the present rows in
__global__ void organ_write_kernel(const int* __restrict__ P, int KP,
                                   const int* __restrict__ Wn, int KW,
                                   short* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   int Npre, int N, int* __restrict__ err) {
    const int b = blockIdx.x / KW, sw = blockIdx.x - b * KW;
    const int j = Wn[(long long)b * KW + sw];
    if (j < 0) return;
    short* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const int wj = j >> 5, bj = j & 31;
    for (int sl = threadIdx.x; sl < KP; sl += blockDim.x) {
        const int i = P[(long long)b * KP + sl];
        if (i < 0) continue;
        if (!((Pb[(long long)i * W + wj] >> bj) & 1u)) continue;
        const int c = (int)Cb[(long long)i * N + j] + 1;
        if (c > DENSE_CMAX) { atomicExch(err, 1); continue; }
        Cb[(long long)i * N + j] = (short)c;
    }
}

// ---- PRESENT-ONLY cross fiber (DESIGN_present_only.md) --------------------
// The connectome is FIXED; store only what exists. Per (brain, row): the
// present columns with their counts, one packed 32-bit entry each (column
// in the low 16 bits, int16 count in the high 16), padded to DMAX with
// PR_PAD. A round reads K row lists (~200 B each) instead of K x N counts.
//
// ONE WARP PER BRAIN, ROWS IN ORDER. Lanes walk a row's entries; a row's
// columns are distinct, so `drive[j] += price` is a plain shared add with no
// race, and every column's sum accumulates in row order -- the SAME float
// sequence as the retired dense kernel's per-column loop (DESIGN_present_only.md
// gated them identical), hence identical drives.
// The write's two passes walk rows the same way into per-slot shared
// accumulators (max, then price change at the new max). Selection is the
// radix select at warp level. No block barrier inside a round.
#define PR_PAD 0xFFFFFFFFu
#define PR_ROWS 4                      // rows whose entries are loaded together
#define PR_LMAX 512                    // winner cells the write remembers (typ. ~K*KW*p)

__device__ __forceinline__ int pr_col(unsigned int e) { return (int)(e & 0xFFFFu); }
__device__ __forceinline__ int pr_cnt(unsigned int e) { return (int)(short)(e >> 16); }
__device__ __forceinline__ unsigned int pr_pack(int col, int cnt) {
    return ((unsigned int)(cnt & 0xFFFF) << 16) | ((unsigned int)col & 0xFFFFu);
}

__global__ void present_degree_kernel(const unsigned int* __restrict__ pres, int W,
                                      long long rows_total, int* __restrict__ deg) {
    const long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= rows_total) return;
    const unsigned int* p = pres + r * W;
    int d = 0;
    for (int w = 0; w < W; ++w) d += __popc(p[w]);
    deg[r] = d;
}

// a row's set bits, ascending, count 0; then PR_PAD
__global__ void present_fill_kernel(const unsigned int* __restrict__ pres, int W, int N,
                                    long long rows_total, int DMAX,
                                    unsigned int* __restrict__ ent) {
    const long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= rows_total) return;
    const unsigned int* p = pres + r * W;
    unsigned int* e = ent + r * DMAX;
    int k = 0;
    for (int w = 0; w < W; ++w) {
        unsigned int word = p[w];
        while (word) {
            const int t = __ffs(word) - 1;
            word &= word - 1;
            const int j = (w << 5) + t;
            if (j < N && k < DMAX) e[k++] = pr_pack(j, 0);
        }
    }
    for (; k < DMAX; ++k) e[k] = PR_PAD;
}

// per-warp shared buffers
struct PrShared {
    double* dmass;        // KW   price-change accumulator per winner slot
    float* drive;         // N    the round's drive, then its keys in place
    int* hist;            // 256
    int* cmx2;            // KW   new column max per slot
    int* rows;            // K
    int* win;             // KW
    int* misc;            // 4    [0] winners found
    unsigned int* wmask;  // W    winner-column bitmap
    unsigned int* wl;     // PR_LMAX  winner cells (col << 16 | new count), row order
    short* cmx;           // N    staged column max
    short* slot;          // N    winner column -> slot; the select's candidate list before that
};

__host__ __device__ __forceinline__ size_t pr_warp_bytes(int N, int W, int K, int KW) {
    size_t b = (size_t)KW * 8 + (size_t)N * 4 + 256 * 4 + (size_t)KW * 4 + (size_t)K * 4
             + (size_t)KW * 4 + 16 + (size_t)W * 4 + (size_t)PR_LMAX * 4
             + (size_t)N * 2 + (size_t)N * 2;
    return (b + 7) & ~(size_t)7;
}

__device__ __forceinline__ PrShared pr_carve(unsigned char* base, int N, int W, int K, int KW) {
    PrShared s;
    s.dmass = reinterpret_cast<double*>(base);            base += (size_t)KW * 8;
    s.drive = reinterpret_cast<float*>(base);             base += (size_t)N * 4;
    s.hist = reinterpret_cast<int*>(base);                base += 256 * 4;
    s.cmx2 = reinterpret_cast<int*>(base);                base += (size_t)KW * 4;
    s.rows = reinterpret_cast<int*>(base);                base += (size_t)K * 4;
    s.win = reinterpret_cast<int*>(base);                 base += (size_t)KW * 4;
    s.misc = reinterpret_cast<int*>(base);                base += 16;
    s.wmask = reinterpret_cast<unsigned int*>(base);      base += (size_t)W * 4;
    s.wl = reinterpret_cast<unsigned int*>(base);         base += (size_t)PR_LMAX * 4;
    s.cmx = reinterpret_cast<short*>(base);               base += (size_t)N * 2;
    s.slot = reinterpret_cast<short*>(base);
    return s;
}

// drive[j] = SUM over rows in order of price(cmax_j - count_ij), present cells
// `absolute`: the price index is the COUNT itself (the engine's chain table,
// clip included) instead of (column max - count) -- the regime without
// column scaling, which the max-relative form does not price.
template <int MAXIT>
__device__ void pr_drive(const unsigned int* __restrict__ eb, int DMAX, int K,
                         const PrShared& s, int N, const float* srel, int nsh,
                         const float* __restrict__ rel, int nnz, int absolute) {
    const int lane = threadIdx.x & 31;
    for (int j = lane; j < N; j += 32) s.drive[j] = 0.0f;
    __syncwarp();
    for (int s0 = 0; s0 < K; s0 += PR_ROWS) {
        unsigned int e[PR_ROWS][MAXIT];
#pragma unroll
        for (int r = 0; r < PR_ROWS; ++r) {                 // PR_ROWS x MAXIT loads in flight
            const int sl = s0 + r;
            const int i = (sl < K) ? s.rows[sl] : -1;
            const unsigned int* re = eb + (long long)(i < 0 ? 0 : i) * DMAX;
#pragma unroll
            for (int t = 0; t < MAXIT; ++t) {
                const int p = lane + 32 * t;
                e[r][t] = (i >= 0 && p < DMAX) ? re[p] : PR_PAD;
            }
        }
#pragma unroll
        for (int r = 0; r < PR_ROWS; ++r) {
#pragma unroll
            for (int t = 0; t < MAXIT; ++t) {
                const unsigned int v = e[r][t];
                if (v != PR_PAD) {
                    const int j = pr_col(v);
                    const int d = absolute ? pr_cnt(v) : (int)s.cmx[j] - pr_cnt(v);
                    s.drive[j] += sched_price(d, srel, nsh, rel, nnz);
                }
            }
            __syncwarp();                                   // row order
        }
    }
}

// the python path: d = stim; d += drive * scale [* invdj]; ranked = d + jit
__device__ void pr_keys(const PrShared& s, int N, const float* __restrict__ scale_b,
                        const float* __restrict__ invdj_b, const float* __restrict__ stim,
                        const float* __restrict__ jt, unsigned int& kand, unsigned int& kor) {
    const int lane = threadIdx.x & 31;
    unsigned int* uk = reinterpret_cast<unsigned int*>(s.drive);
    unsigned int a = 0xFFFFFFFFu, o = 0u;
#pragma unroll 4
    for (int j = lane; j < N; j += 32) {
        float v = s.drive[j] * scale_b[j];
        if (invdj_b != nullptr) v *= invdj_b[j];
        const float dd = stim[j] + v;
        unsigned int u = __float_as_uint(dd + jt[j]);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        uk[j] = u; a &= u; o |= u;
    }
    kand = __reduce_and_sync(0xFFFFFFFFu, a);
    kor = __reduce_or_sync(0xFFFFFFFFu, o);
    __syncwarp();
}

// RANK FINISH: once the candidates fit four per lane (<= 128), the rem-th
// largest of them (keys are unique) is the threshold, by shuffled compares
// instead of up to three more passes. Valid right after a compaction, when the list holds
// exactly the keys matching the prefix and `rem` counts the winners among them.
__device__ __forceinline__ unsigned long long pr_rank_finish(const short* cand, const unsigned int* uk,
                                                             int ncand, int rem) {
    const int lane = threadIdx.x & 31;
    unsigned long long key[4];
    int rank[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int q = lane + 32 * m;
        const int j = (q < ncand) ? (int)cand[q] : -1;
        key[m] = (j >= 0) ? ((((unsigned long long)uk[j]) << 16) | (unsigned long long)(65535 - j)) : 0ull;
        rank[m] = 0;
    }
    for (int o = 0; o < ncand; ++o) {
        unsigned long long other;
        switch (o >> 5) {                                   // uniform
            case 0: other = __shfl_sync(0xFFFFFFFFu, key[0], o & 31); break;
            case 1: other = __shfl_sync(0xFFFFFFFFu, key[1], o & 31); break;
            case 2: other = __shfl_sync(0xFFFFFFFFu, key[2], o & 31); break;
            default: other = __shfl_sync(0xFFFFFFFFu, key[3], o & 31); break;
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) rank[m] += (other > key[m]) ? 1 : 0;
    }
    unsigned long long found = 0ull;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const bool hit = (lane + 32 * m < ncand) && (rank[m] == rem - 1);
        const unsigned int bal = __ballot_sync(0xFFFFFFFFu, hit);
        if (bal) found = __shfl_sync(0xFFFFFFFFu, key[m], __ffs(bal) - 1);
    }
    return found;
}

// the KW largest of the 48-bit keys (key32 << 16 | 65535 - j), warp-level
// radix select; then the winners into win/wmask/slot, cmx2 and dmass reset.
//
// After the first pass only the keys in the crossing bin can still decide
// the threshold, so they are COMPACTED into a candidate list (the slot map's
// space, dead until the winners are known) and later passes scan tens of
// keys instead of a thousand; the compaction pass histograms the next digit
// as it goes, so it costs no extra pass.
__device__ unsigned long long pr_select(const PrShared& s, int N, int W, int KW,
                                        unsigned int kand, unsigned int kor, int& nfound) {
    const int lane = threadIdx.x & 31;
    const unsigned int* uk = reinterpret_cast<const unsigned int*>(s.drive);
    short* cand = s.slot;
    const int lead = (kand ^ kor) ? __clz(kand ^ kor) : 32;
    const unsigned long long lmask = lead ? (~0ull << (48 - lead)) : 0ull;
    unsigned long long prefix = (((unsigned long long)kand) << 16) & lmask, pmask = lmask;
    int rem = KW, ncand = -1;                                 // -1: not compacted yet
    int shift = 40 - lead; if (shift < 0) shift = 0;
    for (int t = lane; t < 256; t += 32) s.hist[t] = 0;
    __syncwarp();
    for (; shift >= 0; shift = (shift >= 8) ? shift - 8 : (shift > 0 ? 0 : -1)) {
        // histogram of this digit over the keys still matching the prefix;
        // on the pass after the first, compact them as well
        if (ncand < 0) {
            for (int base = 0; base < N; base += 32) {
                const int j = base + lane;
                unsigned long long key = 0ull;
                bool valid = false;
                if (j < N) {
                    key = (((unsigned long long)uk[j]) << 16) | (unsigned long long)(65535 - j);
                    valid = ((key & pmask) == prefix);
                }
                const unsigned int d = valid ? (unsigned int)((key >> shift) & 0xFFull) : 0u;
                if (valid) atomicAdd(&s.hist[d], 1);    // plain: a warp's digits mostly differ
            }
        } else {
            int kept = 0;
            for (int base = 0; base < ncand; base += 32) {
                const int q = base + lane;
                int j = -1;
                unsigned long long key = 0ull;
                bool valid = false;
                if (q < ncand) {
                    j = (int)cand[q];
                    key = (((unsigned long long)uk[j]) << 16) | (unsigned long long)(65535 - j);
                    valid = ((key & pmask) == prefix);
                }
                const unsigned int act = __ballot_sync(0xFFFFFFFFu, valid);
                __syncwarp();                                   // everyone has read this chunk
                if (valid) {
                    const int pos = kept + __popc(act & ((1u << lane) - 1u));
                    cand[pos] = (short)j;                       // pos <= q: in place is safe
                    const unsigned int d = (unsigned int)((key >> shift) & 0xFFull);
                    atomicAdd(&s.hist[d], 1);
                }
                kept += __popc(act);
                __syncwarp();
            }
            ncand = kept;
            if (ncand <= 128) { prefix = pr_rank_finish(cand, uk, ncand, rem); break; }
        }
        __syncwarp();
        int sum = 0;
#pragma unroll
        for (int t = 0; t < 8; ++t) sum += s.hist[255 - 8 * lane - t];
        int incl = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
            if (lane >= o) incl += v;
        }
        const int excl = incl - sum;
        const bool here = (excl < rem) && (rem <= incl);
        const unsigned int bal = __ballot_sync(0xFFFFFFFFu, here);
        const int L = __ffs(bal) - 1;
        int digit = 0, nrem = 0, cnt = 0;
        if (lane == L) {
            int acc = excl;
            for (int t = 0; t < 8; ++t) {
                const int bin = 255 - 8 * lane - t;
                const int c = s.hist[bin];
                if (acc + c >= rem) { digit = bin; nrem = rem - acc; cnt = c; break; }
                acc += c;
            }
        }
        digit = __shfl_sync(0xFFFFFFFFu, digit, L);
        nrem = __shfl_sync(0xFFFFFFFFu, nrem, L);
        cnt = __shfl_sync(0xFFFFFFFFu, cnt, L);
        prefix |= ((unsigned long long)digit) << shift;
        pmask |= 0xFFull << shift;
        rem = nrem;
        __syncwarp();
        for (int t = lane; t < 256; t += 32) s.hist[t] = 0;
        __syncwarp();
        if (cnt == rem) break;                              // every key with this prefix wins
        if (ncand < 0) {
            // compact the crossing bin's keys for the passes to come
            int kept = 0;
            for (int base = 0; base < N; base += 32) {
                const int j = base + lane;
                bool valid = false;
                if (j < N) {
                    const unsigned long long key = (((unsigned long long)uk[j]) << 16) | (unsigned long long)(65535 - j);
                    valid = ((key & pmask) == prefix);
                }
                const unsigned int act = __ballot_sync(0xFFFFFFFFu, valid);
                if (valid) cand[kept + __popc(act & ((1u << lane) - 1u))] = (short)j;
                kept += __popc(act);
            }
            ncand = kept;
            __syncwarp();
            if (ncand <= 128) { prefix = pr_rank_finish(cand, uk, ncand, rem); break; }
        }
    }
    if (lane == 0) s.misc[0] = 0;
    for (int w = lane; w < W; w += 32) s.wmask[w] = 0u;
    __syncwarp();
    for (int j = lane; j < N; j += 32) {
        const unsigned long long key = (((unsigned long long)uk[j]) << 16) | (unsigned long long)(65535 - j);
        if (key >= prefix) {
            const int pos = atomicAdd(&s.misc[0], 1);
            if (pos < KW) {
                s.win[pos] = j; s.slot[j] = (short)pos;
                s.cmx2[pos] = (int)s.cmx[j]; s.dmass[pos] = 0.0;
            }
            atomicOr(&s.wmask[j >> 5], 1u << (j & 31));
        }
    }
    __syncwarp();
    nfound = s.misc[0];
    return prefix;
}

// winners given (win/wmask/slot/cmx2/dmass prepared): count the rows in,
// then price the change at the new max -- dense_write_kernel's arithmetic.
//
// LOCKSTEP. A warp executes one instruction stream: a chain of shared loads
// and double-precision ops done by ONE lane while the others are masked
// costs the whole warp that chain, and a loop over ~200 winner cells with
// one owner each ran 200 chains in series (62k cycles). So:
//   pass 1   walks the rows in order, stores the incremented counts, and
//            APPENDS each winner cell (slot, new count) to a shared list by
//            ballot -- no atomics, no lookups beyond the slot, no per-row
//            barrier (positions grow with rows, so the list is in row order)
//   bucket   each lane collects the entries of the slots it owns
//            (slot mod 32) into its own region, in list order
//   pass 2   ALL lanes at once: lane l prices its k-th entry while every
//            other lane prices its own; per slot the entries are met in
//            list = row order, the max first and then the price change,
//            both in registers -- the row walk's exact double sequence
// If the list overflows, the original two-pass row walk runs instead.
template <int MAXIT>
__device__ bool pr_write(unsigned int* __restrict__ eb, int DMAX, int K, const PrShared& s,
                         int N, const float* srel, int nsh, const float* __restrict__ rel,
                         int nnz, int nrel, int* __restrict__ cmax_b, double* __restrict__ mass_b,
                         float* __restrict__ scale_b, float setpoint, int do_scale, int nw) {
    const int lane = threadIdx.x & 31;
    const unsigned int lt = (1u << lane) - 1u;
    bool over = false;
    int nl = 0;
    // ---- pass 1
    for (int s0 = 0; s0 < K; s0 += PR_ROWS) {
        unsigned int e[PR_ROWS][MAXIT];
#pragma unroll
        for (int r = 0; r < PR_ROWS; ++r) {
            const int sl = s0 + r;
            const int i = (sl < K) ? s.rows[sl] : -1;
            const unsigned int* re = eb + (long long)(i < 0 ? 0 : i) * DMAX;
#pragma unroll
            for (int t = 0; t < MAXIT; ++t) {
                const int p = lane + 32 * t;
                e[r][t] = (i >= 0 && p < DMAX) ? re[p] : PR_PAD;
            }
        }
#pragma unroll
        for (int r = 0; r < PR_ROWS; ++r) {
            const int sl = s0 + r;
            const int i = (sl < K) ? s.rows[sl] : -1;
            unsigned int* re = eb + (long long)(i < 0 ? 0 : i) * DMAX;
#pragma unroll
            for (int t = 0; t < MAXIT; ++t) {
                const unsigned int v = e[r][t];
                const int j = pr_col(v);
                const bool hit = (v != PR_PAD) && ((s.wmask[j >> 5] >> (j & 31)) & 1u);
                const int c = pr_cnt(v) + 1;
                const bool ok = hit && (c <= DENSE_CMAX);
                over |= hit && !ok;
                const unsigned int act = __ballot_sync(0xFFFFFFFFu, ok);
                if (ok) {
                    re[lane + 32 * t] = pr_pack(j, c);
                    const int pos = nl + __popc(act & lt);
                    if (pos < PR_LMAX) s.wl[pos] = ((unsigned int)s.slot[j] << 16) | (unsigned int)c;
                }
                nl += __popc(act);
            }
        }
    }
    __syncwarp();
    const int cap = (N < PR_LMAX) ? N : PR_LMAX;              // the bucket scratch is the keys' space
    if (nl <= cap) {
        // ---- bucket by owner lane (slot & 31), list order kept
        unsigned int* scratch = reinterpret_cast<unsigned int*>(s.drive);
        int cnt = 0;
        for (int q = 0; q < nl; ++q) cnt += (((s.wl[q] >> 16) & 31u) == (unsigned int)lane);
        int incl = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
            if (lane >= o) incl += v;
        }
        const int off = incl - cnt;
        int k = 0;
        for (int q = 0; q < nl; ++q) {
            const unsigned int e = s.wl[q];
            if (((e >> 16) & 31u) == (unsigned int)lane) scratch[off + k++] = e;
        }
        __syncwarp();
        // ---- pass 2, all lanes at once: the max of each owned slot, then
        // the price change, per slot in list order
        int maxk = cnt;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) maxk = max(maxk, __shfl_xor_sync(0xFFFFFFFFu, maxk, o));
        // a lane owns slots lane, lane+32, lane+64, lane+96 (KW <= SCHED_MAXK)
        int mx[4];
        double dm[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int jq = (lane + 32 * m < nw) ? s.win[lane + 32 * m] : -1;
            mx[m] = (jq >= 0) ? (int)s.cmx[jq] : 0;
            dm[m] = 0.0;
        }
        for (int q = 0; q < maxk; ++q) {
            const unsigned int e = (q < cnt) ? scratch[off + q] : 0u;
            const int c = (int)(e & 0xFFFFu), m_ = (int)(e >> 21);
#pragma unroll
            for (int m = 0; m < 4; ++m) if (q < cnt && m_ == m) mx[m] = max(mx[m], c);
        }
        if (do_scale) {
            for (int q = 0; q < maxk; ++q) {
                const unsigned int e = (q < cnt) ? scratch[off + q] : 0u;
                const int c = (int)(e & 0xFFFFu), m_ = (int)(e >> 21);
                int cm_new = mx[0];
#pragma unroll
                for (int m = 1; m < 4; ++m) if (m_ == m) cm_new = mx[m];
                const int dn = cm_new - c, dold = cm_new - (c - 1);
                const float rn = sched_price(dn, srel, nsh, rel, nnz);
                const float ro = sched_price(dold, srel, nsh, rel, nnz);
                const double term = (double)rn - (double)ro;
#pragma unroll
                for (int m = 0; m < 4; ++m) if (q < cnt && m_ == m) dm[m] += term;
            }
        }
        // ---- finalize the owned slots (slot q is owned by lane q & 31)
        for (int q = lane; q < nw; q += 32) {
            const int j = s.win[q];
            const int m_ = q >> 5;
            int cm_new = mx[0];
            double dsum = dm[0];
#pragma unroll
            for (int m = 1; m < 4; ++m) if (m_ == m) { cm_new = mx[m]; dsum = dm[m]; }
            const int cm_old = (int)s.cmx[j];
            if (do_scale) {
                const int dc = cm_new - cm_old;
                const double shrink = (dc >= 0 && dc < nrel) ? (double)rel[dc] : 0.0;
                const double m = mass_b[j] * shrink + dsum;
                mass_b[j] = m;
                scale_b[j] = (m > 1e-12) ? (float)((double)setpoint / m) : 1.0f;
            }
            cmax_b[j] = cm_new;
            s.cmx[j] = (short)cm_new;
        }
        __syncwarp();
        return over;
    }
    // ---- overflow fallback: the row walk, twice (max, then price change)
    for (int pass = 0; pass < (do_scale ? 2 : 1); ++pass) {
        for (int s0 = 0; s0 < K; s0 += PR_ROWS) {
            unsigned int e[PR_ROWS][MAXIT];
#pragma unroll
            for (int r = 0; r < PR_ROWS; ++r) {
                const int sl = s0 + r;
                const int i = (sl < K) ? s.rows[sl] : -1;
                const unsigned int* re = eb + (long long)(i < 0 ? 0 : i) * DMAX;
#pragma unroll
                for (int t = 0; t < MAXIT; ++t) {
                    const int p = lane + 32 * t;
                    e[r][t] = (i >= 0 && p < DMAX) ? re[p] : PR_PAD;
                }
            }
#pragma unroll
            for (int r = 0; r < PR_ROWS; ++r) {
#pragma unroll
                for (int t = 0; t < MAXIT; ++t) {
                    const unsigned int v = e[r][t];
                    if (v == PR_PAD) continue;
                    const int j = pr_col(v);
                    if (!((s.wmask[j >> 5] >> (j & 31)) & 1u)) continue;
                    const int sl_ = (int)s.slot[j];
                    const int c = pr_cnt(v);                         // already incremented
                    if (pass == 0) {
                        if (c > s.cmx2[sl_]) s.cmx2[sl_] = c;
                    } else {
                        const int cm_new = s.cmx2[sl_];
                        const int dn = cm_new - c, dold = cm_new - (c - 1);
                        const float rn = sched_price(dn, srel, nsh, rel, nnz);
                        const float ro = sched_price(dold, srel, nsh, rel, nnz);
                        s.dmass[sl_] += (double)rn - (double)ro;
                    }
                }
                __syncwarp();
            }
        }
    }
    for (int q = lane; q < nw; q += 32) {
        const int j = s.win[q];
        const int cm_old = (int)s.cmx[j], cm_new = s.cmx2[q];
        if (do_scale) {
            const int dc = cm_new - cm_old;
            const double shrink = (dc >= 0 && dc < nrel) ? (double)rel[dc] : 0.0;
            const double m = mass_b[j] * shrink + s.dmass[q];
            mass_b[j] = m;
            scale_b[j] = (m > 1e-12) ? (float)((double)setpoint / m) : 1.0f;
        }
        cmax_b[j] = cm_new;
        s.cmx[j] = (short)cm_new;
    }
    __syncwarp();
    return over;
}

// LAYER 3 on the present-only fiber: a warp per brain walks its schedule.
template <int MAXIT>
__global__ void present_train_kernel(const long long* __restrict__ words,
                                     const long long* __restrict__ bundles, int S,
                                     const long long* __restrict__ lex_cache, int V, int K,
                                     const float* __restrict__ bundle_drive,
                                     const float* __restrict__ jit, int I,
                                     unsigned int* __restrict__ ent, int DMAX,
                                     int* __restrict__ cmax, double* __restrict__ mass,
                                     float* __restrict__ scale, const float* __restrict__ invdj,
                                     const float* __restrict__ rel, int nrel, int nsh, int nnz,
                                     int Npre, int N, float setpoint, int rounds, int KW, int B,
                                     int* __restrict__ err, int absolute) {
    extern __shared__ unsigned char smem_raw[];
    const int W = (N + 31) / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, WPB = blockDim.x >> 5;
    float* srel = reinterpret_cast<float*>(smem_raw);
    const size_t srel_bytes = ((size_t)nsh * 4 + 7) & ~(size_t)7;
    PrShared s = pr_carve(smem_raw + srel_bytes + (size_t)warp * pr_warp_bytes(N, W, K, KW), N, W, K, KW);
    for (int t = threadIdx.x; t < nsh; t += blockDim.x) srel[t] = rel[t];
    __syncthreads();                                        // the only block barrier
    const int b = blockIdx.x * WPB + warp;
    if (b >= B) return;
    unsigned int* eb = ent + (long long)b * Npre * DMAX;
    int* cmax_b = cmax + (long long)b * N;
    double* mass_b = mass + (long long)b * N;
    float* scale_b = scale + (long long)b * N;
    const float* invdj_b = (invdj != nullptr) ? invdj + (long long)b * N : nullptr;
    for (int j = lane; j < N; j += 32) s.cmx[j] = (short)cmax_b[j];
    __syncwarp();
    for (int st = 0; st < S; ++st) {
        const long long w = words[(long long)b * S + st];
        const long long bid = bundles[(long long)b * S + st];
        if (w < 0 || bid < 0) break;
        for (int t = lane; t < K; t += 32)
            s.rows[t] = (int)lex_cache[((long long)b * V + w) * K + t];
        __syncwarp();
        const float* stim = bundle_drive + ((long long)b * I + bid) * N;
        const float* jt = jit + ((long long)b * I + bid) * N;
        for (int r = 0; r < rounds; ++r) {
            pr_drive<MAXIT>(eb, DMAX, K, s, N, srel, nsh, rel, nnz, absolute);
            unsigned int kand, kor;
            pr_keys(s, N, scale_b, invdj_b, stim, jt, kand, kor);
            int nfound;
            pr_select(s, N, W, KW, kand, kor, nfound);
            if (nfound != KW) { if (lane == 0) atomicExch(err, 2); return; }
            const bool over = pr_write<MAXIT>(eb, DMAX, K, s, N, srel, nsh, rel, nnz, nrel,
                                              cmax_b, mass_b, scale_b, setpoint, 1, KW);
            if (__any_sync(0xFFFFFFFFu, over) && lane == 0) atomicExch(err, 1);
        }
    }
}

// the python path's drive: out[b, j] += scale * invdj * SUM_rows price
template <int MAXIT>
__global__ void present_drive_kernel(const unsigned int* __restrict__ ent, int DMAX,
                                     const int* __restrict__ S, int K,
                                     const int* __restrict__ cmax, const float* __restrict__ scale,
                                     const float* __restrict__ invdj, const float* __restrict__ rel,
                                     int nnz, int Npre, int N, float* __restrict__ out,
                                     int absolute) {
    extern __shared__ unsigned char smem_raw[];
    __shared__ float dummy[1];
    const int W = (N + 31) / 32, lane = threadIdx.x & 31, b = blockIdx.x;
    PrShared s = pr_carve(smem_raw, N, W, K, 1);
    for (int j = lane; j < N; j += 32) s.cmx[j] = (short)cmax[(long long)b * N + j];
    for (int t = lane; t < K; t += 32) s.rows[t] = S[(long long)b * K + t];
    __syncwarp();
    pr_drive<MAXIT>(ent + (long long)b * Npre * DMAX, DMAX, K, s, N, dummy, 0, rel, nnz, absolute);
    for (int j = lane; j < N; j += 32) {
        float v = s.drive[j] * scale[(long long)b * N + j];
        if (invdj != nullptr) v *= invdj[(long long)b * N + j];
        out[(long long)b * N + j] += v;
    }
}

// the python path's write: winners GIVEN
template <int MAXIT>
__global__ void present_write_kernel(const int* __restrict__ P, int KP,
                                     const int* __restrict__ Wn, int KW,
                                     unsigned int* __restrict__ ent, int DMAX,
                                     int* __restrict__ cmax, double* __restrict__ mass,
                                     float* __restrict__ scale, const float* __restrict__ rel,
                                     int nrel, int nnz, int Npre, int N, float setpoint,
                                     int do_scale, int* __restrict__ err) {
    extern __shared__ unsigned char smem_raw[];
    __shared__ float dummy[1];
    const int W = (N + 31) / 32, lane = threadIdx.x & 31, b = blockIdx.x;
    PrShared s = pr_carve(smem_raw, N, W, KP, KW);
    for (int j = lane; j < N; j += 32) s.cmx[j] = (short)cmax[(long long)b * N + j];
    for (int t = lane; t < KP; t += 32) s.rows[t] = P[(long long)b * KP + t];
    for (int w = lane; w < W; w += 32) s.wmask[w] = 0u;
    if (lane == 0) s.misc[0] = 0;
    __syncwarp();
    for (int t = lane; t < KW; t += 32) {
        const int j = Wn[(long long)b * KW + t];
        if (j < 0) continue;
        const int pos = atomicAdd(&s.misc[0], 1);
        s.win[pos] = j; s.slot[j] = (short)pos;
        s.cmx2[pos] = (int)s.cmx[j]; s.dmass[pos] = 0.0;
        atomicOr(&s.wmask[j >> 5], 1u << (j & 31));
    }
    __syncwarp();
    const int nw = s.misc[0];
    if (nw == 0) return;
    const bool over = pr_write<MAXIT>(ent + (long long)b * Npre * DMAX, DMAX, KP, s, N, dummy, 0,
                                      rel, nnz, nrel, cmax + (long long)b * N,
                                      mass + (long long)b * N, scale + (long long)b * N,
                                      setpoint, do_scale, nw);
    if (__any_sync(0xFFFFFFFFu, over) && lane == 0) atomicExch(err, 1);
}

// the roofline probe for this layout: K row lists per round, a warp per
// brain, rows shifting each round -- the drive's reads and nothing else
template <int MAXIT>
__global__ void present_probe_kernel(const unsigned int* __restrict__ ent, int DMAX,
                                     const int* __restrict__ S, int K, int Npre,
                                     int rounds, float* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, WPB = blockDim.x >> 5;
    const int b = blockIdx.x * WPB + warp;
    const unsigned int* eb = ent + (long long)b * Npre * DMAX;
    float acc = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        for (int s0 = 0; s0 < K; s0 += PR_ROWS) {
            unsigned int e[PR_ROWS][MAXIT];
#pragma unroll
            for (int q = 0; q < PR_ROWS; ++q) {
                const int sl = s0 + q;
                const int i = (sl < K) ? (S[(long long)b * K + sl] + r) % Npre : -1;
                const unsigned int* re = eb + (long long)(i < 0 ? 0 : i) * DMAX;
#pragma unroll
                for (int t = 0; t < MAXIT; ++t) {
                    const int p = lane + 32 * t;
                    e[q][t] = (i >= 0 && p < DMAX) ? re[p] : PR_PAD;
                }
            }
#pragma unroll
            for (int q = 0; q < PR_ROWS; ++q)
#pragma unroll
                for (int t = 0; t < MAXIT; ++t) acc += (float)(e[q][t] & 0xFFu);
        }
    }
    out[(long long)b * 32 + lane] = acc;
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

void organ_drive(torch::Tensor S, torch::Tensor C, torch::Tensor pres, torch::Tensor invdj,
                 torch::Tensor tab, torch::Tensor out) {
    S = S.contiguous(); tab = tab.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt16, "counts are int16");
    const int B = C.size(0), Npre = C.size(1), N = C.size(2), K = S.size(1), W = pres.size(2);
    if (K == 0) return;
    const long long tot = (long long)B * N;
    const int th = 256;
    organ_drive_kernel<<<(tot + th - 1) / th, th>>>(
        S.data_ptr<int>(), K, C.data_ptr<short>(),
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        invdj.numel() ? invdj.data_ptr<float>() : nullptr,
        tab.data_ptr<float>(), (int)tab.numel(), B, Npre, N, out.data_ptr<float>());
}

void organ_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor C, torch::Tensor pres,
                 torch::Tensor err) {
    P = P.contiguous(); Wn = Wn.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt16, "counts are int16");
    const int B = C.size(0), Npre = C.size(1), N = C.size(2), W = pres.size(2);
    const int KP = P.size(1), KW = Wn.size(1);
    if (KP == 0 || KW == 0) return;
    organ_write_kernel<<<B * KW, 128>>>(
        P.data_ptr<int>(), KP, Wn.data_ptr<int>(), KW, C.data_ptr<short>(),
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        Npre, N, err.data_ptr<int>());
}

torch::Tensor present_degree(torch::Tensor pres) {
    pres = pres.contiguous();
    const int B = pres.size(0), Npre = pres.size(1), W = pres.size(2);
    auto out = torch::empty({B, Npre}, torch::dtype(torch::kInt32).device(pres.device()));
    const long long tot = (long long)B * Npre;
    const int th = 256;
    present_degree_kernel<<<(tot + th - 1) / th, th>>>(
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W, tot, out.data_ptr<int>());
    return out;
}

torch::Tensor present_fill(torch::Tensor pres, int64_t n_post, int64_t dmax) {
    pres = pres.contiguous();
    const int B = pres.size(0), Npre = pres.size(1), W = pres.size(2);
    auto out = torch::empty({B, Npre, dmax}, torch::dtype(torch::kInt32).device(pres.device()));
    const long long tot = (long long)B * Npre;
    const int th = 256;
    present_fill_kernel<<<(tot + th - 1) / th, th>>>(
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W, (int)n_post, tot,
        (int)dmax, reinterpret_cast<unsigned int*>(out.data_ptr<int>()));
    return out;
}

template <typename F>
static void pr_dispatch(int dmax, F&& f) {
    if (dmax <= 128) f(std::integral_constant<int, 4>{});
    else if (dmax <= 512) f(std::integral_constant<int, 16>{});
    else TORCH_CHECK(false, "row degree too large for the present-only kernels");
}

void present_drive(torch::Tensor ent, torch::Tensor S, torch::Tensor cmax,
                   torch::Tensor scale, torch::Tensor invdj, torch::Tensor rel,
                   int64_t nnz, torch::Tensor out, int64_t absolute) {
    S = S.contiguous(); rel = rel.contiguous();
    const int B = ent.size(0), Npre = ent.size(1), DMAX = ent.size(2);
    const int K = S.size(1), N = out.size(1), W = (N + 31) / 32;
    if (K == 0) return;
    const size_t shm = pr_warp_bytes(N, W, K, 1);
    pr_dispatch(DMAX, [&](auto tag) { constexpr int MAXIT = decltype(tag)::value;
        cudaFuncSetAttribute(present_drive_kernel<MAXIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        present_drive_kernel<MAXIT><<<B, 32, shm>>>(
            reinterpret_cast<const unsigned int*>(ent.data_ptr<int>()), DMAX,
            S.data_ptr<int>(), K, cmax.data_ptr<int>(), scale.data_ptr<float>(),
            invdj.numel() ? invdj.data_ptr<float>() : nullptr, rel.data_ptr<float>(),
            (int)nnz, Npre, N, out.data_ptr<float>(), (int)absolute); });
}

void present_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor ent, torch::Tensor cmax,
                   torch::Tensor mass, torch::Tensor scale, torch::Tensor rel, int64_t nnz,
                   double setpoint, int64_t do_scale, torch::Tensor err) {
    P = P.contiguous(); Wn = Wn.contiguous(); rel = rel.contiguous();
    const int B = ent.size(0), Npre = ent.size(1), DMAX = ent.size(2);
    const int KP = P.size(1), KW = Wn.size(1), N = cmax.size(1), W = (N + 31) / 32;
    if (KP == 0 || KW == 0) return;
    const size_t shm = pr_warp_bytes(N, W, KP, KW);
    pr_dispatch(DMAX, [&](auto tag) { constexpr int MAXIT = decltype(tag)::value;
        cudaFuncSetAttribute(present_write_kernel<MAXIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        present_write_kernel<MAXIT><<<B, 32, shm>>>(
            P.data_ptr<int>(), KP, Wn.data_ptr<int>(), KW,
            reinterpret_cast<unsigned int*>(ent.data_ptr<int>()), DMAX,
            cmax.data_ptr<int>(), mass.data_ptr<double>(), scale.data_ptr<float>(),
            rel.data_ptr<float>(), (int)rel.numel(), (int)nnz, Npre, N, (float)setpoint,
            (int)do_scale, err.data_ptr<int>()); });
}

void present_train(torch::Tensor words, torch::Tensor bundles, torch::Tensor lex_cache,
                   torch::Tensor bundle_drive, torch::Tensor jit, torch::Tensor ent,
                   torch::Tensor cmax, torch::Tensor mass, torch::Tensor scale,
                   torch::Tensor invdj, torch::Tensor rel, double setpoint, int64_t rounds,
                   int64_t kw, torch::Tensor err, int64_t nsh, int64_t nnz, int64_t wpb,
                   int64_t absolute) {
    words = words.contiguous(); bundles = bundles.contiguous();
    lex_cache = lex_cache.contiguous(); bundle_drive = bundle_drive.contiguous();
    jit = jit.contiguous(); rel = rel.contiguous();
    const int B = ent.size(0), Npre = ent.size(1), DMAX = ent.size(2);
    const int S = words.size(1), V = lex_cache.size(1), K = lex_cache.size(2);
    const int I = bundle_drive.size(1), N = bundle_drive.size(2), W = (N + 31) / 32;
    TORCH_CHECK(N <= 65535, "N must fit the 16-bit column field");
    TORCH_CHECK(nsh >= 0 && nsh <= nnz && nnz <= rel.numel(), "nsh/nnz");
    TORCH_CHECK(wpb >= 1 && wpb <= 32, "warps per block");
    const size_t shm = (((size_t)nsh * 4 + 7) & ~(size_t)7) + (size_t)wpb * pr_warp_bytes(N, W, K, (int)kw);
    TORCH_CHECK(shm <= 96 * 1024, "per-warp buffers exceed shared memory; fewer warps per block");
    const int blocks = (B + (int)wpb - 1) / (int)wpb;
    pr_dispatch(DMAX, [&](auto tag) { constexpr int MAXIT = decltype(tag)::value;
        cudaFuncSetAttribute(present_train_kernel<MAXIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        present_train_kernel<MAXIT><<<blocks, 32 * (int)wpb, shm>>>(
            words.data_ptr<int64_t>(), bundles.data_ptr<int64_t>(), S,
            lex_cache.data_ptr<int64_t>(), V, K, bundle_drive.data_ptr<float>(),
            jit.data_ptr<float>(), I, reinterpret_cast<unsigned int*>(ent.data_ptr<int>()), DMAX,
            cmax.data_ptr<int>(), mass.data_ptr<double>(), scale.data_ptr<float>(),
            invdj.numel() ? invdj.data_ptr<float>() : nullptr, rel.data_ptr<float>(),
            (int)rel.numel(), (int)nsh, (int)nnz, Npre, N, (float)setpoint, (int)rounds,
            (int)kw, B, err.data_ptr<int>(), (int)absolute); });
}

torch::Tensor present_probe(torch::Tensor ent, torch::Tensor S, int64_t rounds, int64_t wpb) {
    S = S.contiguous();
    const int B = ent.size(0), Npre = ent.size(1), DMAX = ent.size(2), K = S.size(1);
    TORCH_CHECK(B % wpb == 0, "B must be a multiple of warps per block");
    auto out = torch::zeros({B, 32}, torch::dtype(torch::kFloat32).device(ent.device()));
    pr_dispatch(DMAX, [&](auto tag) { constexpr int MAXIT = decltype(tag)::value;
        present_probe_kernel<MAXIT><<<B / (int)wpb, 32 * (int)wpb>>>(
            reinterpret_cast<const unsigned int*>(ent.data_ptr<int>()), DMAX,
            S.data_ptr<int>(), K, Npre, (int)rounds, out.data_ptr<float>()); });
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
torch::Tensor present_degree(torch::Tensor pres);
void organ_drive(torch::Tensor S, torch::Tensor C, torch::Tensor pres, torch::Tensor invdj, torch::Tensor tab, torch::Tensor out);
void organ_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor C, torch::Tensor pres, torch::Tensor err);
torch::Tensor present_fill(torch::Tensor pres, int64_t n_post, int64_t dmax);
void present_drive(torch::Tensor ent, torch::Tensor S, torch::Tensor cmax, torch::Tensor scale, torch::Tensor invdj, torch::Tensor rel, int64_t nnz, torch::Tensor out, int64_t absolute);
void present_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor ent, torch::Tensor cmax, torch::Tensor mass, torch::Tensor scale, torch::Tensor rel, int64_t nnz, double setpoint, int64_t do_scale, torch::Tensor err);
void present_train(torch::Tensor words, torch::Tensor bundles, torch::Tensor lex_cache, torch::Tensor bundle_drive, torch::Tensor jit, torch::Tensor ent, torch::Tensor cmax, torch::Tensor mass, torch::Tensor scale, torch::Tensor invdj, torch::Tensor rel, double setpoint, int64_t rounds, int64_t kw, torch::Tensor err, int64_t nsh, int64_t nnz, int64_t wpb, int64_t absolute);
torch::Tensor present_probe(torch::Tensor ent, torch::Tensor S, int64_t rounds, int64_t wpb);
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
                           "column_mass_rel", "hashed_presence",
                           "organ_drive", "organ_write",
                           "present_degree", "present_fill", "present_drive",
                           "present_write", "present_train", "present_probe",
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
