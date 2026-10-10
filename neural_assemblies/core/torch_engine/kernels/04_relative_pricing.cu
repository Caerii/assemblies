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


