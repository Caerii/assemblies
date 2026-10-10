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
    // Sort the smallest power of two that holds the M candidates, not all
    // CAPS slots: M is typically k plus a few ties (~60-100), and the full
    // 2048-slot network was most of the kernel. The keys are unique and
    // nonzero and the padding is 0, so the top K land in the same order at
    // the top of the shorter array: the same winners, in the same order.
    int P = 1;
    while (P < M) P <<= 1;
    for (int i = M + tid; i < P; i += NTH) ck[i] = 0ULL;
    __syncthreads();

    for (int kk = 2; kk <= P; kk <<= 1) {
        for (int jj = kk >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < P; i += NTH) {
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
        ob[s] = 65535 - (int)(ck[P - 1 - s] & 0xFFFFULL);
}


