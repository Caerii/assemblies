void dev_correct_exact(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts,
                       torch::Tensor offs, torch::Tensor rowmask,
                       torch::Tensor colmask, torch::Tensor scratch,
                       torch::Tensor tab, torch::Tensor seeds,
                       int64_t threshold, torch::Tensor out) {
    S = S.contiguous(); tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = S.size(0), K = S.size(1), N = out.size(1);
    if (K == 0) return;
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        offs = offs.contiguous();
        devcnt_csr_kernel<<<B * K, 128, 0, NA_STREAM>>>(
            S.data_ptr<int>(), keys.data_ptr<int64_t>(), cnts.data_ptr<int>(),
            offs.data_ptr<int64_t>(), (int)offs.numel() - 1, K, N,
            scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * rowmask.size(-1)));
        devcnt_mask_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
            S.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    devapply_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        S.data_ptr<int>(), scratch.data_ptr<int>(), tab.data_ptr<float>(),
        (int)tab.numel(), seeds.data_ptr<int>(), B, K, N, (int)threshold,
        out.data_ptr<float>());
}


std::vector<torch::Tensor> column_mass_exact(
        torch::Tensor cols, torch::Tensor keys, torch::Tensor cnts,
        torch::Tensor colmap, torch::Tensor rowmask, torch::Tensor colmask,
        torch::Tensor scratch, torch::Tensor tab, torch::Tensor seeds,
        int64_t n, int64_t threshold) {
    cols = cols.contiguous(); tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = cols.size(0), K = cols.size(1), N = (int)n;
    auto opt = torch::dtype(torch::kFloat32).device(cols.device());
    auto out = torch::empty({B, (int64_t)K}, opt);
    auto omax = torch::empty({B, (int64_t)K}, opt);
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        colmap = colmap.contiguous();
        const long long nnz = keys.numel();
        colcnt_store_kernel<<<(nnz + th - 1) / th, th, 0, NA_STREAM>>>(
            keys.data_ptr<int64_t>(), cnts.data_ptr<int>(), nnz,
            colmap.data_ptr<int>(), K, N, scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * N));
        colcnt_mask_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
            cols.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    colmass_apply_kernel<<<B * K, 256, 0, NA_STREAM>>>(
        cols.data_ptr<int>(), scratch.data_ptr<int>(), tab.data_ptr<float>(),
        (int)tab.numel(), seeds.data_ptr<int>(), K, N, (int)threshold,
        out.data_ptr<float>(), omax.data_ptr<float>());
    return {out, omax};
}


void dev_correct_rel(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts,
                     torch::Tensor offs, torch::Tensor rowmask,
                     torch::Tensor colmask, torch::Tensor scratch,
                     torch::Tensor rel, torch::Tensor cmax, torch::Tensor seeds,
                     int64_t threshold, torch::Tensor out) {
    S = S.contiguous(); rel = rel.contiguous(); seeds = seeds.contiguous();
    cmax = cmax.contiguous();
    const int B = S.size(0), K = S.size(1), N = out.size(1);
    if (K == 0) return;
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        offs = offs.contiguous();
        devcnt_csr_kernel<<<B * K, 128, 0, NA_STREAM>>>(
            S.data_ptr<int>(), keys.data_ptr<int64_t>(), cnts.data_ptr<int>(),
            offs.data_ptr<int64_t>(), (int)offs.numel() - 1, K, N,
            scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * rowmask.size(-1)));
        devcnt_mask_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
            S.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    devapply_rel_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        S.data_ptr<int>(), scratch.data_ptr<int>(), rel.data_ptr<float>(),
        (int)rel.numel(), cmax.data_ptr<int>(), seeds.data_ptr<int>(),
        B, K, N, (int)threshold, out.data_ptr<float>());
}


std::vector<torch::Tensor> column_mass_rel(
        torch::Tensor cols, torch::Tensor keys, torch::Tensor cnts,
        torch::Tensor colmap, torch::Tensor rowmask, torch::Tensor colmask,
        torch::Tensor scratch, torch::Tensor rel, torch::Tensor seeds,
        int64_t n, int64_t threshold) {
    cols = cols.contiguous(); rel = rel.contiguous(); seeds = seeds.contiguous();
    const int B = cols.size(0), K = cols.size(1), N = (int)n;
    auto optf = torch::dtype(torch::kFloat32).device(cols.device());
    auto opti = torch::dtype(torch::kInt32).device(cols.device());
    auto out = torch::empty({B, (int64_t)K}, optf);
    auto omax = torch::empty({B, (int64_t)K}, opti);
    const long long tot = (long long)B * K * N;
    const int th = 256;
    if (keys.numel() > 0) {
        keys = keys.contiguous(); cnts = cnts.contiguous();
        colmap = colmap.contiguous();
        const long long nnz = keys.numel();
        colcnt_store_kernel<<<(nnz + th - 1) / th, th, 0, NA_STREAM>>>(
            keys.data_ptr<int64_t>(), cnts.data_ptr<int>(), nnz,
            colmap.data_ptr<int>(), K, N, scratch.data_ptr<int>());
    }
    if (rowmask.numel() > 0) {
        rowmask = rowmask.contiguous(); colmask = colmask.contiguous();
        const int W = (int)(rowmask.numel() / ((long long)B * N));
        colcnt_mask_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
            cols.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
            colmask.data_ptr<int64_t>(), B, K, N, W, scratch.data_ptr<int>());
    }
    colmass_rel_kernel<<<B * K, 256, 0, NA_STREAM>>>(
        cols.data_ptr<int>(), scratch.data_ptr<int>(), rel.data_ptr<float>(),
        (int)rel.numel(), seeds.data_ptr<int>(), K, N, (int)threshold,
        out.data_ptr<float>(), omax.data_ptr<int>());
    return {out, omax};
}


torch::Tensor hashed_presence(torch::Tensor seeds, int64_t n_pre, int64_t n_post,
                              int64_t threshold) {
    seeds = seeds.contiguous();
    const int B = seeds.size(0), W = (int)((n_post + 31) / 32);
    auto out = torch::empty({B, n_pre, (int64_t)W},
                            torch::dtype(torch::kInt32).device(seeds.device()));
    const long long tot = (long long)B * n_pre * W;
    const int th = 256;
    presence_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        seeds.data_ptr<int>(), B, (int)n_pre, (int)n_post, W, (int)threshold,
        reinterpret_cast<unsigned int*>(out.data_ptr<int>()));
    return out;
}

