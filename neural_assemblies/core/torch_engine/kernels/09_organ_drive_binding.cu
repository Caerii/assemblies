template <typename CT>
static void organ_drive_t(torch::Tensor S, CT* Cp, torch::Tensor C, torch::Tensor pres,
                          torch::Tensor invdj, torch::Tensor tab, torch::Tensor bmap,
                          torch::Tensor out);

void organ_drive(torch::Tensor S, torch::Tensor C, torch::Tensor pres, torch::Tensor invdj,
                 torch::Tensor tab, torch::Tensor bmap, torch::Tensor out) {
    S = S.contiguous(); tab = tab.contiguous(); bmap = bmap.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt8 || C.scalar_type() == torch::kInt16,
                "counts are int8 or int16");
    if (C.scalar_type() == torch::kInt16) {
        organ_drive_t<short>(S, C.data_ptr<short>(), C, pres, invdj, tab, bmap, out);
        return;
    }
    organ_drive_t<signed char>(S, C.data_ptr<signed char>(), C, pres, invdj, tab, bmap, out);
}

template <typename CT>
static void organ_drive_t(torch::Tensor S, CT* Cp, torch::Tensor C, torch::Tensor pres,
                          torch::Tensor invdj, torch::Tensor tab, torch::Tensor bmap,
                          torch::Tensor out) {
    const int B = C.size(0), Npre = C.size(1), N = C.size(2), K = S.size(1), W = pres.size(2);
    const int V = S.size(0);
    TORCH_CHECK(out.size(0) == V && out.size(1) == N && out.is_contiguous(), "out: contiguous [V, N]");
    TORCH_CHECK(bmap.numel() ? bmap.numel() == V : V == B,
                "without a brain map the rows are the brains");
    TORCH_CHECK(tab.dim() == 1 || (tab.dim() == 2 && tab.size(0) == B),
                "tab: [ntab] shared or [B, ntab] per brain");
    if (K == 0) return;
    const int ntab = (int)tab.size(tab.dim() - 1);
    const int stride = tab.dim() == 2 ? ntab : 0;
    const int th = 256;
    if (N % 4 == 0) {
        const long long tot4 = (long long)V * (N / 4);
        organ_drive4_kernel<CT><<<(tot4 + th - 1) / th, th, 0, NA_STREAM>>>(
            S.data_ptr<int>(), K, Cp,
            reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
            invdj.numel() ? invdj.data_ptr<float>() : nullptr,
            tab.data_ptr<float>(), ntab, stride,
            bmap.numel() ? bmap.data_ptr<int>() : nullptr,
            V, Npre, N, out.data_ptr<float>());
        return;
    }
    const long long tot = (long long)V * N;
    organ_drive_kernel<CT><<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        S.data_ptr<int>(), K, Cp,
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        invdj.numel() ? invdj.data_ptr<float>() : nullptr,
        tab.data_ptr<float>(), ntab, stride,
        bmap.numel() ? bmap.data_ptr<int>() : nullptr,
        V, Npre, N, out.data_ptr<float>());
}

