torch::Tensor present_degree(torch::Tensor pres) {
    pres = pres.contiguous();
    const int B = pres.size(0), Npre = pres.size(1), W = pres.size(2);
    auto out = torch::empty({B, Npre}, torch::dtype(torch::kInt32).device(pres.device()));
    const long long tot = (long long)B * Npre;
    const int th = 256;
    present_degree_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W, tot, out.data_ptr<int>());
    return out;
}

torch::Tensor present_fill(torch::Tensor pres, int64_t n_post, int64_t dmax) {
    pres = pres.contiguous();
    const int B = pres.size(0), Npre = pres.size(1), W = pres.size(2);
    auto out = torch::empty({B, Npre, dmax}, torch::dtype(torch::kInt32).device(pres.device()));
    const long long tot = (long long)B * Npre;
    const int th = 256;
    present_fill_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
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
        present_drive_kernel<MAXIT><<<B, 32, shm, NA_STREAM>>>(
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
        present_write_kernel<MAXIT><<<B, 32, shm, NA_STREAM>>>(
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
        present_train_kernel<MAXIT><<<blocks, 32 * (int)wpb, shm, NA_STREAM>>>(
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
        present_probe_kernel<MAXIT><<<B / (int)wpb, 32 * (int)wpb, 0, NA_STREAM>>>(
            reinterpret_cast<const unsigned int*>(ent.data_ptr<int>()), DMAX,
            S.data_ptr<int>(), K, Npre, (int)rounds, out.data_ptr<float>()); });
    return out;
}

