"""
Custom CUDA kernels for GPU-optimized sparse assembly projection.

These kernels eliminate GPU-to-CPU sync points and fuse multiple CuPy
operations into single kernel launches.  Used by CudaImplicitEngine
to override NumpySparseEngine.project_into hot paths.

Requires: cupy
"""

import cupy as cp
import numpy as np

# ---------------------------------------------------------------------------
# K1: Scatter penalty — vectorized LRI application
# ---------------------------------------------------------------------------

_scatter_penalty_src = r'''
extern "C" __global__
void scatter_penalty(
    float* inputs,              // all_inputs array to modify in-place
    const unsigned int* indices, // neuron indices to penalise
    const float* penalties,     // penalty amount per entry
    const unsigned int n        // number of (index, penalty) pairs
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int idx = indices[tid];
    atomicAdd(&inputs[idx], -penalties[tid]);
}
'''

scatter_penalty_kernel = cp.RawKernel(_scatter_penalty_src, 'scatter_penalty')


def apply_lri_penalties(all_inputs, indices, penalties):
    """Apply LRI penalties via a single GPU kernel launch.

    Args:
        all_inputs: CuPy float32 array — modified in-place.
        indices: CuPy uint32 array — neuron indices to penalise.
        penalties: CuPy float32 array — penalty magnitudes (positive).
    """
    n = len(indices)
    if n == 0:
        return
    block = 256
    grid = (n + block - 1) // block
    scatter_penalty_kernel((grid,), (block,),
                           (all_inputs, indices, penalties, cp.uint32(n)))


# ---------------------------------------------------------------------------
# K2: Fused 1-D Hebbian update (stim -> area)
# ---------------------------------------------------------------------------

_hebbian_1d_src = r'''
extern "C" __global__
void hebbian_1d(
    float* weights,             // 1-D weight array (length >= w_len)
    const unsigned int* winners, // winner indices to update
    const unsigned int n_winners,
    const float beta,
    const float w_max,          // 0 = no clamp
    const unsigned int w_len    // valid length of weights
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_winners) return;
    unsigned int idx = winners[tid];
    if (idx >= w_len) return;
    float w = weights[idx] * (1.0f + beta);
    if (w_max > 0.0f && w > w_max) w = w_max;
    weights[idx] = w;
}
'''

hebbian_1d_kernel = cp.RawKernel(_hebbian_1d_src, 'hebbian_1d')


def apply_hebbian_1d(weights, winners, beta, w_max):
    """Fused Hebbian update for 1-D stim->area weights.

    Args:
        weights: CuPy float32 array — modified in-place.
        winners: CuPy uint32 array of winner indices.
        beta: plasticity rate.
        w_max: weight ceiling (0 or None = no clamp).
    """
    n = len(winners)
    if n == 0:
        return
    wm = float(w_max) if w_max is not None else 0.0
    block = 256
    grid = (n + block - 1) // block
    hebbian_1d_kernel((grid,), (block,),
                      (weights, winners,
                       cp.uint32(n), cp.float32(beta),
                       cp.float32(wm), cp.uint32(len(weights))))


# ---------------------------------------------------------------------------
# K3: Fused 2-D Hebbian update (area -> area)
# ---------------------------------------------------------------------------

_hebbian_2d_src = r'''
extern "C" __global__
void hebbian_2d(
    float* matrix,                  // 2-D weight matrix (row-major)
    const unsigned int stride,      // number of columns (physical width)
    const unsigned int* src_winners, // row indices
    const unsigned int n_src,
    const unsigned int* tgt_winners, // column indices
    const unsigned int n_tgt,
    const float beta,
    const float w_max,              // 0 = no clamp
    const unsigned int n_rows,      // matrix row count (bounds check)
    const unsigned int n_cols       // matrix col count (bounds check)
) {
    // Each thread handles one (src, tgt) pair
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_pairs = n_src * n_tgt;
    if (tid >= total_pairs) return;

    unsigned int si = tid / n_tgt;
    unsigned int ti = tid % n_tgt;
    unsigned int row = src_winners[si];
    unsigned int col = tgt_winners[ti];

    if (row >= n_rows || col >= n_cols) return;

    unsigned int offset = row * stride + col;
    float w = matrix[offset] * (1.0f + beta);
    if (w_max > 0.0f && w > w_max) w = w_max;
    matrix[offset] = w;
}
'''

hebbian_2d_kernel = cp.RawKernel(_hebbian_2d_src, 'hebbian_2d')


def apply_hebbian_2d(matrix, src_winners, tgt_winners, beta, w_max):
    """Fused Hebbian update for 2-D area->area weight matrix.

    Args:
        matrix: CuPy float32 2-D array — modified in-place.
        src_winners: CuPy uint32 array of source (row) indices.
        tgt_winners: CuPy uint32 array of target (column) indices.
        beta: plasticity rate.
        w_max: weight ceiling (0 or None = no clamp).
    """
    n_src = len(src_winners)
    n_tgt = len(tgt_winners)
    if n_src == 0 or n_tgt == 0:
        return
    total = n_src * n_tgt
    wm = float(w_max) if w_max is not None else 0.0
    n_rows, n_cols = matrix.shape
    stride = matrix.strides[0] // matrix.itemsize  # physical column count
    block = 256
    grid = (total + block - 1) // block
    hebbian_2d_kernel(
        (grid,), (block,),
        (matrix, cp.uint32(stride),
         src_winners, cp.uint32(n_src),
         tgt_winners, cp.uint32(n_tgt),
         cp.float32(beta), cp.float32(wm),
         cp.uint32(n_rows), cp.uint32(n_cols)))


# ---------------------------------------------------------------------------
# K4: Fused first-timer detection + input gathering + index remapping
# ---------------------------------------------------------------------------
# NOTE: This kernel runs with k threads (one per winner). k is small
# (typically 100), so it runs as a single block.  The sequential
# prefix-sum for remapping is fine at this scale.

_detect_first_timers_src = r'''
extern "C" __global__
void detect_first_timers(
    const unsigned int* winners,    // input: raw winner indices (k)
    const float* all_inputs,        // input: all_inputs array
    const unsigned int k,
    const unsigned int w,           // current ever-fired count
    unsigned int* out_remapped,     // output: remapped winner indices (k)
    float* out_first_inputs,        // output: first-timer input values (up to k)
    unsigned int* out_num_first,    // output: scalar count of first-timers
    const unsigned int all_inputs_len // bounds check for all_inputs
) {
    // Single block kernel — k threads cooperate via shared memory
    extern __shared__ unsigned int s_is_first[];  // k entries: 1 if first-timer, 0 otherwise

    unsigned int tid = threadIdx.x;
    if (tid >= k) return;

    unsigned int idx = winners[tid];
    unsigned int is_first = (idx >= w) ? 1u : 0u;
    s_is_first[tid] = is_first;
    __syncthreads();

    // Thread 0 does a sequential prefix-sum (k <= ~5000, fast enough)
    __shared__ unsigned int s_prefix[1];  // running count
    if (tid == 0) {
        s_prefix[0] = 0;
        unsigned int count = 0;
        for (unsigned int i = 0; i < k; i++) {
            if (s_is_first[i]) {
                unsigned int remap = w + count;
                out_remapped[i] = remap;
                // Gather input value for this first-timer
                unsigned int orig_idx = winners[i];
                if (orig_idx < all_inputs_len) {
                    out_first_inputs[count] = all_inputs[orig_idx];
                } else {
                    out_first_inputs[count] = 0.0f;
                }
                count++;
            } else {
                out_remapped[i] = winners[i];
            }
        }
        *out_num_first = count;
    }
}
'''

detect_first_timers_kernel = cp.RawKernel(
    _detect_first_timers_src, 'detect_first_timers')


def detect_and_remap_first_timers(winners_gpu, all_inputs, k, w):
    """Detect first-timers, remap indices, gather inputs — all on GPU.

    Args:
        winners_gpu: CuPy uint32 array of raw winner indices (length k).
        all_inputs: CuPy float32 array of all input strengths.
        k: assembly size.
        w: current ever-fired count.

    Returns:
        (remapped_winners_gpu, first_inputs_gpu, num_first):
        - remapped_winners_gpu: CuPy uint32[k] with first-timers remapped to w, w+1, ...
        - first_inputs_gpu: CuPy float32[num_first] with input values for first-timers
        - num_first: int (transferred from GPU)
    """
    out_remapped = cp.empty(k, dtype=cp.uint32)
    out_first_inputs = cp.empty(k, dtype=cp.float32)
    out_num_first = cp.zeros(1, dtype=cp.uint32)

    shared_mem = k * 4  # s_is_first: k uint32s
    detect_first_timers_kernel(
        (1,), (k,),
        (winners_gpu, all_inputs, cp.uint32(k), cp.uint32(w),
         out_remapped, out_first_inputs, out_num_first,
         cp.uint32(len(all_inputs))),
        shared_mem=shared_mem)

    num_first = int(out_num_first[0])  # single GPU->CPU sync
    first_inputs = out_first_inputs[:num_first]
    return out_remapped, first_inputs, num_first
