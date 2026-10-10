void organ_write(torch::Tensor P, torch::Tensor Wn, torch::Tensor C, torch::Tensor pres,
                 torch::Tensor err) {
    P = P.contiguous(); Wn = Wn.contiguous();
    TORCH_CHECK(C.scalar_type() == torch::kInt8 || C.scalar_type() == torch::kInt16 ||
                C.scalar_type() == torch::kUInt8,
                "counts are int8, int16 or packed 4-bit (uint8)");
    if (C.scalar_type() == torch::kUInt8) {
        organ_write_packed(P, Wn, C, pres, err);
        return;
    }
    const int B = C.size(0), Npre = C.size(1), N = C.size(2), W = pres.size(2);
    const int KP = P.size(1), KW = Wn.size(1);
    if (KP == 0 || KW == 0) return;
    if (C.scalar_type() == torch::kInt16) {
        organ_write_kernel<short><<<B * KW, 128, 0, NA_STREAM>>>(
            P.data_ptr<int>(), KP, Wn.data_ptr<int>(), KW, C.data_ptr<short>(),
            reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
            Npre, N, err.data_ptr<int>());
        return;
    }
    organ_write_kernel<signed char><<<B * KW, 128, 0, NA_STREAM>>>(
        P.data_ptr<int>(), KP, Wn.data_ptr<int>(), KW, C.data_ptr<signed char>(),
        reinterpret_cast<const unsigned int*>(pres.data_ptr<int>()), W,
        Npre, N, err.data_ptr<int>());
}

