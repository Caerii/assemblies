
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Every launch goes on torch's CURRENT stream, not the legacy default
// stream: kernels then order with torch's own ops under a stream context,
// and a sequence of them can be captured as a CUDA graph
// (DESIGN_memory_throughput.md). Outside a stream context the current
// stream IS the default stream, so nothing else changes.
#define NA_STREAM at::cuda::getCurrentCUDAStream()

#define NB    4096
#define NTH   1024
// Shared candidate slots. 2048 keys x 8B = 16 KB, plus hist 16 KB and part
// 4 KB = 36 KB, inside the 48 KB default. Raised from 1024 because k=sqrt(n)
// at n=16000 (k=126) overflowed at 1551 candidates -- the guard refused rather
// than truncating, which is correct, but it blocked the measurement.
#define CAPS  2048
#define CHB   (NB / NTH)

