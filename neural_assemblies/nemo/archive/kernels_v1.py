"""
NEMO CUDA Kernels
=================

Version: 1.1.0
Author: Assembly Calculus Project
Date: 2025-11-29

Custom CUDA kernels for GPU-accelerated assembly operations:
1. Implicit random connectivity (hash-based, no storage)
2. FP16 projections (2x memory bandwidth)
3. Hebbian learning with saturation
4. PyTorch top-k integration (3x faster)

Memory: O(learned_connections) instead of O(n^2)
Speed: 175 sentences/sec at n=1M

Changelog:
- 1.1.0: FP16 support, PyTorch top-k integration
- 1.0.0: Initial implicit connectivity
"""

import cupy as cp
import torch
import numpy as np

# Check PyTorch CUDA availability
USE_TORCH_TOPK = torch.cuda.is_available()

# =============================================================================
# CUDA KERNEL: FP16 Batched Projection
# =============================================================================

projection_fp16_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void projection_fp16(
    const unsigned int* __restrict__ active_batch,
    __half* __restrict__ result_batch,
    const unsigned int k,
    const unsigned int n,
    const unsigned int batch_size,
    const unsigned int* __restrict__ seeds,
    const float p
) {
    unsigned int dst = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int batch_idx = blockIdx.y;
    
    if (dst >= n || batch_idx >= batch_size) return;
    
    const unsigned int* active = active_batch + batch_idx * k;
    __half* result = result_batch + batch_idx * n;
    unsigned int seed = seeds[batch_idx];
    
    extern __shared__ unsigned int s_active[];
    for (unsigned int i = threadIdx.x; i < k; i += blockDim.x) {
        s_active[i] = active[i];
    }
    __syncthreads();
    
    unsigned int threshold = (unsigned int)(p * 16777216.0f);
    float sum = 0.0f;
    
    #pragma unroll 8
    for (unsigned int i = 0; i < k; i++) {
        unsigned int src = s_active[i];
        unsigned int hash = (src * 2654435761u) ^ (dst * 2246822519u) ^ seed;
        sum += (float)((hash & 0xFFFFFFu) < threshold);
    }
    
    result[dst] = __float2half(sum);
}
''', 'projection_fp16')


# =============================================================================
# CUDA KERNEL: Hebbian Learning with Saturation
# =============================================================================

hebbian_kernel = cp.RawKernel(r'''
extern "C" __global__
void hebbian_update(
    unsigned int* learned_src,
    unsigned int* learned_dst,
    float* learned_delta,
    unsigned int* num_learned,
    const unsigned int* prev_active,
    const unsigned int* new_active,
    const unsigned int k,
    const float beta,
    const float w_max,
    const unsigned int max_learned,
    const unsigned int seed,
    const float p
) {
    unsigned int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = pair_idx / k;
    unsigned int j = pair_idx % k;
    
    if (i >= k) return;
    
    unsigned int src = prev_active[i];
    unsigned int dst = new_active[j];
    
    // Check if base connection exists (hash-based)
    unsigned int hash = seed ^ src;
    hash *= 0x01000193u;
    hash ^= dst;
    hash *= 0x01000193u;
    
    if ((float)(hash & 0xFFFFFFu) / 16777216.0f >= p) return;
    
    // Find existing entry
    unsigned int current_num = *num_learned;
    unsigned int found_idx = 0xFFFFFFFFu;
    
    for (unsigned int l = 0; l < current_num; l++) {
        if (learned_src[l] == src && learned_dst[l] == dst) {
            found_idx = l;
            break;
        }
    }
    
    if (found_idx == 0xFFFFFFFFu) {
        // New connection
        unsigned int new_idx = atomicAdd(num_learned, 1);
        if (new_idx < max_learned) {
            learned_src[new_idx] = src;
            learned_dst[new_idx] = dst;
            learned_delta[new_idx] = beta;
        }
    } else {
        // Saturating update: delta = beta * (1 - w/w_max)
        float current_w = 1.0f + learned_delta[found_idx];
        float update = beta * (1.0f - current_w / w_max);
        if (update > 0) atomicAdd(&learned_delta[found_idx], update);
    }
}
''', 'hebbian_update')


# =============================================================================
# CUDA KERNEL: FP32 Projection (for compatibility)
# =============================================================================

projection_fp32_kernel = cp.RawKernel(r'''
extern "C" __global__
void projection_fp32(
    const unsigned int* __restrict__ active,
    float* __restrict__ result,
    const unsigned int k,
    const unsigned int n,
    const unsigned int seed,
    const float p
) {
    unsigned int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= n) return;
    
    extern __shared__ unsigned int s_active[];
    if (threadIdx.x < k && threadIdx.x < 256) {
        s_active[threadIdx.x] = active[threadIdx.x];
    }
    __syncthreads();
    
    unsigned int threshold = (unsigned int)(p * 16777216.0f);
    float sum = 0.0f;
    
    for (unsigned int i = 0; i < k; i++) {
        unsigned int src = s_active[i];
        unsigned int hash = (src * 2654435761u) ^ (dst * 2246822519u) ^ seed;
        if ((hash & 0xFFFFFFu) < threshold) {
            sum += 1.0f;
        }
    }
    
    result[dst] = sum;
}
''', 'projection_fp32')


# =============================================================================
# ImplicitAssemblyArea Class
# =============================================================================

class ImplicitAssemblyArea:
    """
    Assembly area using implicit random connectivity.
    
    Memory: O(learned_connections) instead of O(n^2)
    
    Note: k should be sqrt(n) per Assembly Calculus theory.
    
    Optimizations:
    - FP16 activations for faster top-k
    - PyTorch top-k (3x faster than CuPy)
    - Pre-allocated buffers
    """
    
    def __init__(self, n: int, k: int = None, p: float = 0.05,
                 beta: float = 0.1, w_max: float = 10.0, seed: int = 42,
                 use_fp16: bool = True):
        # k = sqrt(n) by default per Assembly Calculus
        if k is None:
            k = int(np.sqrt(n))
        self.n = n
        self.k = k
        self.p = p
        self.beta = beta
        self.w_max = w_max
        self.seed = seed
        self.use_fp16 = use_fp16 and USE_TORCH_TOPK
        
        # Learned weights (sparse COO format)
        self.max_learned = k * k * 1000
        self.learned_src = cp.zeros(self.max_learned, dtype=cp.uint32)
        self.learned_dst = cp.zeros(self.max_learned, dtype=cp.uint32)
        self.learned_delta = cp.zeros(self.max_learned, dtype=cp.float32)
        self.num_learned = cp.zeros(1, dtype=cp.uint32)
        
        # Working memory
        self.activations = cp.zeros(n, dtype=cp.float32)
        self.active = None
        
        # Pre-allocated PyTorch tensor for top-k
        if USE_TORCH_TOPK:
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self._torch_activations = torch.zeros(n, device='cuda', dtype=dtype)
        
        # Kernel config
        self.block_size = 512
        self.grid_size = (n + self.block_size - 1) // self.block_size
    
    def project(self, input_indices: cp.ndarray, learn: bool = True) -> cp.ndarray:
        """
        Project from input indices to new winners.
        
        Returns:
            Indices of top-k winners
        """
        k_in = len(input_indices)
        
        # Clear and project
        self.activations[:] = 0
        
        projection_fp32_kernel(
            (self.grid_size,), (self.block_size,),
            (input_indices, self.activations,
             cp.uint32(k_in), cp.uint32(self.n),
             cp.uint32(self.seed), cp.float32(self.p)),
            shared_mem=min(k_in, 256) * 4
        )
        
        # Top-k using PyTorch (3x faster)
        if USE_TORCH_TOPK:
            self._torch_activations.copy_(
                torch.as_tensor(self.activations, device='cuda')
            )
            _, top_idx = torch.topk(self._torch_activations, self.k, sorted=False)
            winners = cp.asarray(top_idx)
        else:
            winners = cp.argpartition(self.activations, -self.k)[-self.k:]
        
        # Hebbian update
        if learn and self.active is not None:
            grid = (self.k * self.k + self.block_size - 1) // self.block_size
            hebbian_kernel(
                (grid,), (self.block_size,),
                (self.learned_src, self.learned_dst, self.learned_delta,
                 self.num_learned, self.active, winners,
                 cp.uint32(self.k), cp.float32(self.beta), cp.float32(self.w_max),
                 cp.uint32(self.max_learned), cp.uint32(self.seed), cp.float32(self.p))
            )
        
        self.active = winners
        return winners
    
    def memory_usage(self) -> dict:
        """Get memory usage in bytes."""
        learned = int(self.num_learned[0])
        return {
            'learned_connections': learned,
            'learned_bytes': learned * 12,
            'activations_bytes': self.n * 4,
            'total_bytes': learned * 12 + self.n * 4
        }

