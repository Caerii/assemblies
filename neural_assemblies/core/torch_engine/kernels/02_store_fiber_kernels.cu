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
//     d_j = #{ i in [0, n_pre) : cell (i, j) is present }
// is the TRUE in-degree, not an estimate. The rows run over the SOURCE area,
// n_pre, and the columns over the target, n_post. Until 2026-10-07 this kernel
// took one n for both, so a fiber between areas of different sizes divided by
// the in-degree from n_post rows -- n_post / n_pre times too large on average
// and only ~0.7 correlated with the true one (PREREG_refraction_memory.md,
// Amendment 35). Square fibers were exact and are unchanged. The whole defect class that lives on
// the estimate -- pricing unknown rows at the brain's p instead of the fiber's
// (79fba4f, a 6.15x over-scale) -- cannot occur in this path.
//
// One thread per (b, j), n hashes each. This is O(B * n^2) and is computed
// ONCE per brain set, not per round: the connectome does not change.
__global__ void indegree_kernel(const int* __restrict__ seeds, int B, int Npre, int N,
                                int threshold, float floor_,
                                float* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    int b = (int)(idx / N), j = (int)(idx - (long long)b * N);
    unsigned int ch = ((unsigned int)j * 2246822519u) ^ (unsigned int)seeds[b];
    int d = 0;
    for (int i = 0; i < Npre; ++i) {
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

