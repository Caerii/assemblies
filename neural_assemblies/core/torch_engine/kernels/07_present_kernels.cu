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

