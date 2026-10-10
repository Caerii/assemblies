// ---- FUSED ELEMENTWISE STEPS OF A ROUND (DESIGN_memory_throughput.md) -----
// Each replaces a chain of torch ops with one pass, doing the SAME float
// operations in the same order. The _rn intrinsics forbid the compiler from
// contracting a multiply and an add into one FMA, which would round once
// where torch rounds twice.
//
// stim_add: a learned stimulus's priced drive added into `drive`:
//     drive += [min(base * gain[min(pot, top)], hi) / dj] * mult
// (StimulusFiber.contribute's chain; hi, dj and mult are optional).
__global__ void stim_add_kernel(float* __restrict__ drive, const float* __restrict__ base,
                                const float* __restrict__ gain, int gain_stride, int top,
                                const long long* __restrict__ pot,
                                float hi, int has_hi, const float* __restrict__ dj,
                                float mult, int has_mult, int B, int N) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * N) return;
    const int b = (int)(idx / N);
    long long c = pot[idx];
    if (c > top) c = top;
    float d = __fmul_rn(base[idx], gain[(long long)b * gain_stride + c]);
    if (has_hi) d = fminf(d, hi);
    if (dj != nullptr) d = __fdiv_rn(d, dj[idx]);
    if (has_mult) d = __fmul_rn(d, mult);
    drive[idx] = __fadd_rn(drive[idx], d);
}

void stim_add(torch::Tensor drive, torch::Tensor base, torch::Tensor gain, torch::Tensor pot,
              double hi, torch::Tensor dj, double mult) {
    TORCH_CHECK(drive.is_contiguous() && base.is_contiguous() && pot.is_contiguous(),
                "contiguous drive, base, pot");
    TORCH_CHECK(pot.scalar_type() == torch::kInt64, "pot is int64");
    gain = gain.contiguous();
    const int B = drive.size(0), N = drive.size(1);
    const int ntab = (int)gain.size(gain.dim() - 1);
    const int stride = gain.dim() == 2 ? ntab : 0;
    const bool has_hi = std::isfinite(hi);
    const long long tot = (long long)B * N;
    const int th = 256;
    stim_add_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        drive.data_ptr<float>(), base.data_ptr<float>(), gain.data_ptr<float>(), stride,
        ntab - 1, pot.data_ptr<long long>(), has_hi ? (float)hi : 0.0f, has_hi ? 1 : 0,
        dj.numel() ? dj.contiguous().data_ptr<float>() : nullptr,
        (float)mult, mult != 1.0 ? 1 : 0, B, N);
}

// charge: the refraction bias and the ever-fired record at a round's winners:
//     bias[b, j] += raw[b, j] * strength_b;  ever[b, j] = true
// (HashedArea.charge then ever.scatter_). A brain's winners are distinct,
// so each bias cell takes one add, as scatter_add_ gives it. Winners of -1
// are skipped.
__global__ void charge_kernel(float* __restrict__ bias, const float* __restrict__ raw,
                              const int* __restrict__ sel, int K,
                              const float* __restrict__ strength, float s0,
                              bool* __restrict__ ever, int B, int N) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * K) return;
    const int b = (int)(idx / K);
    const int j = sel[idx];
    if (j < 0) return;
    const long long o = (long long)b * N + j;
    if (bias != nullptr) {
        const float s = strength != nullptr ? strength[b] : s0;
        bias[o] = __fadd_rn(bias[o], __fmul_rn(raw[o], s));
    }
    ever[o] = true;
}

void charge(torch::Tensor bias, torch::Tensor raw, torch::Tensor sel, torch::Tensor strength,
            double s0, torch::Tensor ever) {
    sel = sel.contiguous();
    TORCH_CHECK(sel.scalar_type() == torch::kInt32, "winners are int32");
    TORCH_CHECK(raw.is_contiguous() && ever.is_contiguous() && ever.scalar_type() == torch::kBool,
                "contiguous raw, bool ever");
    const int B = sel.size(0), K = sel.size(1), N = raw.size(1);
    if (K == 0) return;
    const long long tot = (long long)B * K;
    const int th = 256;
    charge_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        bias.numel() ? bias.data_ptr<float>() : nullptr, raw.data_ptr<float>(),
        sel.data_ptr<int>(), K,
        strength.numel() ? strength.contiguous().data_ptr<float>() : nullptr, (float)s0,
        ever.data_ptr<bool>(), B, N);
}

