// ---- PACKED 4-BIT COUNTS for the dense organ fiber (opt-in) ----------------
// DenseOrganFiber(count_dtype="int4"): two counts per byte, column j of a row
// in byte j >> 1 (the low nibble holds even j), each row padded to RB row
// bytes, RB = ceil(N / 4) * 2 (not "NB": the prelude defines NB), so four adjacent columns are one aligned 16-bit
// load. Half the bytes of int8 and half a drive's count traffic.
//
// EXACT wherever the weight clip binds by count 15: a count is priced
// min(count, ntab - 1) exactly as the wider types price theirs, every column
// sums its rows in the same order with the same predicate, and the write
// saturates at 15 (err = 1, informational once the table is saturated there).
// The drive therefore depends only on min(count, clip) -- what int8 storage
// gives too -- so writing and reading are bit-identical to int8. Unlearning is
// NOT: a count held at 15 falls below the clip after fewer decrements than an
// int8 count that ran on toward 127 (see DenseOrganFiber).
__device__ __forceinline__ int packed_count(const unsigned char* row, int j) {
    const unsigned char b = row[j >> 1];
    return (j & 1) ? (b >> 4) : (b & 15);
}

// one thread per column, as organ_drive_kernel
__global__ void organ_drive_packed_kernel(const int* __restrict__ S, int K,
                                          const unsigned char* __restrict__ C, int RB,
                                          const unsigned int* __restrict__ pres, int W,
                                          const float* __restrict__ invdj,
                                          const float* __restrict__ tab, int ntab, int tab_stride,
                                          const int* __restrict__ bmap,
                                          int V, int Npre, int N, float* __restrict__ out) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)V * N) return;
    const int j = (int)(idx % N), v = (int)(idx / N);
    const int b = bmap != nullptr ? bmap[v] : v;
    const unsigned char* Cb = C + (long long)b * Npre * RB;
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
            cs[u] = pr ? packed_count(Cb + (long long)i * RB, j) : 0;    // predicated
        }
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int c = cs[u] < ntab ? cs[u] : ntab - 1;
            acc += ((pm >> u) & 1u) ? Tb[c] : 0.0f;
        }
    }
    float d = acc;
    if (invdj != nullptr) d *= invdj[(long long)b * N + j];
    out[idx] += d;
}

// four columns per thread, as organ_drive4_kernel: one 16-bit load per row
__global__ void organ_drive4_packed_kernel(const int* __restrict__ S, int K,
                                           const unsigned char* __restrict__ C, int RB,
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
    const unsigned char* Cb = C + (long long)b * Npre * RB;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const float* Tb = tab + (long long)b * tab_stride;
    const int* Sb = S + (long long)v * K;
    const int wj = j >> 5, bj = j & 31;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int s0 = 0; s0 < K; s0 += ORGAN_CH) {
        unsigned int cs[ORGAN_CH];
        unsigned int nb[ORGAN_CH];
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int sl = s0 + u;
            const int i = (sl < K) ? Sb[sl] : -1;
            const unsigned int pw = (i >= 0) ? Pb[(long long)i * W + wj] : 0u;
            nb[u] = (pw >> bj) & 0xFu;
            cs[u] = nb[u] ? (unsigned int)*reinterpret_cast<const unsigned short*>(
                                Cb + (long long)i * RB + (j >> 1))
                          : 0u;                                             // predicated
        }
#pragma unroll
        for (int u = 0; u < ORGAN_CH; ++u) {
            const int x0 = cs[u] & 15u, x1 = (cs[u] >> 4) & 15u;
            const int x2 = (cs[u] >> 8) & 15u, x3 = (cs[u] >> 12) & 15u;
            const int c0 = x0 < ntab ? x0 : ntab - 1;
            const int c1 = x1 < ntab ? x1 : ntab - 1;
            const int c2 = x2 < ntab ? x2 : ntab - 1;
            const int c3 = x3 < ntab ? x3 : ntab - 1;
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

// one block per (brain, winner column), as organ_write_kernel. Columns j and
// j ^ 1 of a row share a byte and are written by DIFFERENT blocks, so each
// increment is a compare-and-swap on the aligned 32-bit word that holds it.
__global__ void organ_write_packed_kernel(const int* __restrict__ P, int KP,
                                          const int* __restrict__ Wn, int KW,
                                          unsigned char* __restrict__ C, int RB,
                                          const unsigned int* __restrict__ pres, int W,
                                          int Npre, int* __restrict__ err) {
    const int b = blockIdx.x / KW, sw = blockIdx.x - b * KW;
    const int j = Wn[(long long)b * KW + sw];
    if (j < 0) return;
    unsigned char* Cb = C + (long long)b * Npre * RB;
    const unsigned int* Pb = pres + (long long)b * Npre * W;
    const int wj = j >> 5, bj = j & 31;
    for (int sl = threadIdx.x; sl < KP; sl += blockDim.x) {
        const int i = P[(long long)b * KP + sl];
        if (i < 0) continue;
        if (!((Pb[(long long)i * W + wj] >> bj) & 1u)) continue;
        const unsigned long long addr = reinterpret_cast<unsigned long long>(
            Cb + (long long)i * RB + (j >> 1));
        unsigned int* word = reinterpret_cast<unsigned int*>(addr & ~3ull);
        const int sh = (int)(addr & 3ull) * 8 + ((j & 1) ? 4 : 0);
        unsigned int old = *word, assumed;
        do {
            assumed = old;
            if (((assumed >> sh) & 15u) >= 15u) { atomicExch(err, 1); break; }
            old = atomicCAS(word, assumed, assumed + (1u << sh));
        } while (old != assumed);
    }
}

// host launchers, called by organ_drive / organ_write for uint8 (packed) counts
static void organ_drive_packed(torch::Tensor S, torch::Tensor C, torch::Tensor pres,
                               torch::Tensor invdj, torch::Tensor tab, torch::Tensor bmap,
                               torch::Tensor out) {
    const int B = C.size(0), Npre = C.size(1), RB = C.size(2), N = out.size(1);
    const int K = S.size(1), W = pres.size(2), V = S.size(0);
    TORCH_CHECK(RB == ((N + 3) / 4) * 2, "packed counts: rows of ceil(N / 4) * 2 bytes");
    TORCH_CHECK(out.size(0) == V && out.is_contiguous(), "out: contiguous [V, N]");
    TORCH_CHECK(bmap.numel() ? bmap.numel() == V : V == B,
                "without a brain map the rows are the brains");
    TORCH_CHECK(tab.dim() == 1 || (tab.dim() == 2 && tab.size(0) == B),
                "tab: [ntab] shared or [B, ntab] per brain");
    if (K == 0) return;
    const int ntab = (int)tab.size(tab.dim() - 1);
    const int stride = tab.dim() == 2 ? ntab : 0;
    const int th = 256;
    const unsigned char* Cp = C.data_ptr<unsigned char>();
    const unsigned int* pp = reinterpret_cast<const unsigned int*>(pres.data_ptr<int>());
    const float* ij = invdj.numel() ? invdj.data_ptr<float>() : nullptr;
    const int* bm = bmap.numel() ? bmap.data_ptr<int>() : nullptr;
    if (N % 4 == 0) {
        const long long tot4 = (long long)V * (N / 4);
        organ_drive4_packed_kernel<<<(tot4 + th - 1) / th, th, 0, NA_STREAM>>>(
            S.data_ptr<int>(), K, Cp, RB, pp, W, ij, tab.data_ptr<float>(), ntab, stride,
            bm, V, Npre, N, out.data_ptr<float>());
        return;
    }
    const long long tot = (long long)V * N;
    organ_drive_packed_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        S.data_ptr<int>(), K, Cp, RB, pp, W, ij, tab.data_ptr<float>(), ntab, stride,
        bm, V, Npre, N, out.data_ptr<float>());
}

static void organ_write_packed(torch::Tensor P, torch::Tensor Wn, torch::Tensor C,
                               torch::Tensor pres, torch::Tensor err) {
    const int B = C.size(0), Npre = C.size(1), RB = C.size(2), W = pres.size(2);
    const int KP = P.size(1), KW = Wn.size(1);
    if (KP == 0 || KW == 0) return;
    organ_write_packed_kernel<<<B * KW, 128, 0, NA_STREAM>>>(
        P.data_ptr<int>(), KP, Wn.data_ptr<int>(), KW, C.data_ptr<unsigned char>(), RB,
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W, Npre, err.data_ptr<int>());
}

