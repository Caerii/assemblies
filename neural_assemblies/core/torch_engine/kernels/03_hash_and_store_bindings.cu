torch::Tensor hashed_drive(torch::Tensor rows, torch::Tensor seeds,
                           int64_t n, int64_t threshold) {
    TORCH_CHECK(rows.dim() == 2 && rows.is_cuda()
                && rows.scalar_type() == torch::kInt32, "rows: [B,K] i32 cuda");
    rows = rows.contiguous();
    seeds = seeds.contiguous();
    const int B = rows.size(0), K = rows.size(1);
    auto out = torch::empty({B, (int64_t)n},
                            torch::dtype(torch::kFloat32).device(rows.device()));
    const long long tot = (long long)B * n;
    const int th = 256;
    hashed_drive_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        rows.data_ptr<int>(), seeds.data_ptr<int>(), B, K, (int)n,
        (int)threshold, out.data_ptr<float>());
    return out;
}


torch::Tensor hashed_indegree(torch::Tensor seeds, int64_t n_pre, int64_t n, int64_t threshold, double floor_);
std::vector<torch::Tensor> column_mass(torch::Tensor cols, torch::Tensor rowmask, torch::Tensor colmask, torch::Tensor tab, torch::Tensor seeds, int64_t threshold);
void dev_correct(torch::Tensor S, torch::Tensor rowmask, torch::Tensor colids,
                 torch::Tensor colmask, torch::Tensor tab,
                 torch::Tensor seeds, int64_t threshold, torch::Tensor out) {
    S = S.contiguous(); rowmask = rowmask.contiguous();
    colids = colids.contiguous(); colmask = colmask.contiguous();
    tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = S.size(0), K = S.size(1), C = colids.size(1);
    const int N = out.size(1);
    if (C == 0 || K == 0) return;
    const long long tot = (long long)B * K * C;
    const int th = 256;
    dev_correct_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        S.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
        colids.data_ptr<int>(), colmask.data_ptr<int64_t>(),
        tab.data_ptr<float>(), (int)tab.numel(), seeds.data_ptr<int>(),
        B, K, C, N, (int)(rowmask.numel() / ((long long)B * N)),
        (int)threshold, out.data_ptr<float>());
}


torch::Tensor hashed_indegree(torch::Tensor seeds, int64_t n_pre, int64_t n,
                              int64_t threshold, double floor_) {
    seeds = seeds.contiguous();
    const int B = seeds.size(0);
    auto out = torch::empty({B, (int64_t)n},
                            torch::dtype(torch::kFloat32).device(seeds.device()));
    const long long tot = (long long)B * n;
    const int th = 256;
    indegree_kernel<<<(tot + th - 1) / th, th, 0, NA_STREAM>>>(
        seeds.data_ptr<int>(), B, (int)n_pre, (int)n, (int)threshold, (float)floor_,
        out.data_ptr<float>());
    return out;
}


std::vector<torch::Tensor> column_mass(torch::Tensor cols,
                          torch::Tensor rowmask,
                          torch::Tensor colmask, torch::Tensor tab,
                          torch::Tensor seeds, int64_t threshold) {
    cols = cols.contiguous(); rowmask = rowmask.contiguous();
    colmask = colmask.contiguous(); tab = tab.contiguous();
    seeds = seeds.contiguous();
    const int B = cols.size(0), K = cols.size(1);
    const int N = (int)colmask.size(-1);
    const int W = (int)(rowmask.numel() / ((long long)B * N));
    auto opt = torch::dtype(torch::kFloat32).device(cols.device());
    auto out = torch::empty({B, (int64_t)K}, opt);
    auto omax = torch::empty({B, (int64_t)K}, opt);
    colmass_kernel<<<B * K, 256, 0, NA_STREAM>>>(
        cols.data_ptr<int>(), rowmask.data_ptr<int64_t>(),
        colmask.data_ptr<int64_t>(), tab.data_ptr<float>(),
        (int)tab.numel(), seeds.data_ptr<int>(), K, N, W, (int)threshold,
        out.data_ptr<float>(), omax.data_ptr<float>());
    return {out, omax};
}


void dev_correct_csr(torch::Tensor S, torch::Tensor keys, torch::Tensor cnts,
                     torch::Tensor offs, torch::Tensor tab,
                     torch::Tensor seeds, int64_t threshold,
                     torch::Tensor out) {
    S = S.contiguous(); keys = keys.contiguous(); cnts = cnts.contiguous();
    offs = offs.contiguous(); tab = tab.contiguous(); seeds = seeds.contiguous();
    const int B = S.size(0), K = S.size(1), N = out.size(1);
    if (K == 0 || keys.numel() == 0) return;
    dev_csr_kernel<<<B * K, 128, 0, NA_STREAM>>>(
        S.data_ptr<int>(), keys.data_ptr<int64_t>(), cnts.data_ptr<int>(),
        offs.data_ptr<int64_t>(), (int)offs.numel() - 1,
        tab.data_ptr<float>(), (int)tab.numel(),
        seeds.data_ptr<int>(), K, N, (int)threshold, out.data_ptr<float>());
}



