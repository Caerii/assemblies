// ---- PRESENCE BITMASK (DESIGN_dense_floor.md) ------------------------------
// The connectome as bits, pres[b, i, j/32], built ONCE by the same hash the
// store fiber tests at apply time. It is the hash's stored form (GATE-4) and
// the source of the present-only lists below. Counts are int16 everywhere:
// a writer flags a count that would pass 32767 in `err` rather than wrap.

#define DENSE_CMAX 32767
#define ORGAN_CMAX 127      // int8 counts in the dense organ fiber; the chain table saturates near 31

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

