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

// BRAIN MAP AND PER-BRAIN TABLES (DESIGN_memory_throughput.md). A row of
// `S` and `out` is a VIRTUAL brain v; `bmap[v]` names the physical brain whose
// counts, connectome, in-degrees and table it reads (no map: v itself). That
// lets a frozen read run many cues against one brain's synapses in a single
// launch. `tab_stride` 0 shares one chain table; ntab gives each physical
// brain its own (a learning-rate sweep batched into the brain axis). The
// arithmetic per column is unchanged -- the same rows in the same order --
// so a mapped or swept brain reads the drive it would read alone.
// COUNT WIDTH. The organ fiber stores int8 counts (CMAX 127) wherever the
// weight clip binds by count 127 -- every rate from ~0.024 up -- and int16
// counts (CMAX 32767) below that, where a weak write's counts run on before
// the clip. The kernels are templated on the count type; the int8
// instantiation is the arithmetic the int8-only kernels did.
template <typename CT> struct OrganCount;
template <> struct OrganCount<signed char> {
    typedef char4 vec4;
    static __device__ __forceinline__ vec4 zero4() { return make_char4(0, 0, 0, 0); }
    static constexpr int cmax = 127;
};
template <> struct OrganCount<short> {
    typedef short4 vec4;
    static __device__ __forceinline__ vec4 zero4() { return make_short4(0, 0, 0, 0); }
    static constexpr int cmax = 32767;
};

template <typename CT>
__global__ void organ_drive_kernel(const int* __restrict__ S, int K,
                                   const CT* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   const float* __restrict__ invdj,
                                   const float* __restrict__ tab, int ntab, int tab_stride,
                                   const int* __restrict__ bmap,
                                   int V, int Npre, int N, float* __restrict__ out) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)V * N) return;
    const int j = (int)(idx % N), v = (int)(idx / N);
    const int b = bmap != nullptr ? bmap[v] : v;
    const CT* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const float* Tb = tab + (long long)b * tab_stride;
    const int* Sb = S + (long long)v * K;
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
            acc += ((pm >> u) & 1u) ? Tb[c] : 0.0f;
        }
    }
    float d = acc;
    if (invdj != nullptr) d *= invdj[(long long)b * N + j];
    out[idx] += d;
}

// FOUR COLUMNS PER THREAD (n a multiple of 4). One char4 load per row
// fetches four adjacent counts and one presence word covers all four (j is a
// multiple of 4, so bits j..j+3 share a word): a quarter of the load
// instructions of `organ_drive_kernel`, and 128-byte warp transactions
// instead of 32. Each column still sums its rows in row order with the same
// predicate, so every column's float sequence -- and its drive -- is the
// scalar kernel's.
template <typename CT>
__global__ void organ_drive4_kernel(const int* __restrict__ S, int K,
                                    const CT* __restrict__ C,
                                    const unsigned int* __restrict__ pres, int W,
                                    const float* __restrict__ invdj,
                                    const float* __restrict__ tab, int ntab, int tab_stride,
                                    const int* __restrict__ bmap,
                                    int V, int Npre, int N, float* __restrict__ out) {
    const int N4 = N >> 2;
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)V * N4) return;
    const int j = (int)(idx % N4) << 2, v = (int)(idx / N4);
    const int b = bmap != nullptr ? bmap[v] : v;
    const CT* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const float* Tb = tab + (long long)b * tab_stride;
    const int* Sb = S + (long long)v * K;
    const int wj = j >> 5, bj = j & 31;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int s0 = 0; s0 < K; s0 += ORGAN_CH) {
        typename OrganCount<CT>::vec4 cs[ORGAN_CH];
        unsigned int nb[ORGAN_CH];
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int sl = s0 + u;
            const int i = (sl < K) ? Sb[sl] : -1;
            const unsigned int pw = (i >= 0) ? Pb[(long long)i * W + wj] : 0u;
            nb[u] = (pw >> bj) & 0xFu;
            cs[u] = nb[u] ? *reinterpret_cast<const typename OrganCount<CT>::vec4*>(
                                Cb + (long long)i * N + j)
                          : OrganCount<CT>::zero4();                    // predicated
        }
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int c0 = cs[u].x < ntab ? cs[u].x : ntab - 1;
            const int c1 = cs[u].y < ntab ? cs[u].y : ntab - 1;
            const int c2 = cs[u].z < ntab ? cs[u].z : ntab - 1;
            const int c3 = cs[u].w < ntab ? cs[u].w : ntab - 1;
            a0 += (nb[u] & 1u) ? Tb[c0] : 0.0f;
            a1 += (nb[u] & 2u) ? Tb[c1] : 0.0f;
            a2 += (nb[u] & 4u) ? Tb[c2] : 0.0f;
            a3 += (nb[u] & 8u) ? Tb[c3] : 0.0f;
        }
    }
    const long long o = (long long)v * N + j;
    if (invdj != nullptr) {
        const float* ib = invdj + (long long)b * N + j;
        a0 *= ib[0]; a1 *= ib[1]; a2 *= ib[2]; a3 *= ib[3];
    }
    out[o] += a0; out[o + 1] += a1; out[o + 2] += a2; out[o + 3] += a3;
}

// one block per (brain, winner column): count the present rows in
template <typename CT>
__global__ void organ_write_kernel(const int* __restrict__ P, int KP,
                                   const int* __restrict__ Wn, int KW,
                                   CT* __restrict__ C,
                                   const unsigned int* __restrict__ pres, int W,
                                   int Npre, int N, int* __restrict__ err) {
    const int b = blockIdx.x / KW, sw = blockIdx.x - b * KW;
    const int j = Wn[(long long)b * KW + sw];
    if (j < 0) return;
    CT* Cb = C + (long long)b * Npre * N;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const int wj = j >> 5, bj = j & 31;
    for (int sl = threadIdx.x; sl < KP; sl += blockDim.x) {
        const int i = P[(long long)b * KP + sl];
        if (i < 0) continue;
        if (!((Pb[(long long)i * W + wj] >> bj) & 1u)) continue;
        const int c = (int)Cb[(long long)i * N + j] + 1;
        if (c > OrganCount<CT>::cmax) { atomicExch(err, 1); continue; }
        Cb[(long long)i * N + j] = (CT)c;
    }
}

