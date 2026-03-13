"""
NEMO Core Kernels
=================

Version: 1.1.0
Date: 2025-12-01

CUDA kernels for assembly operations:
1. projection_kernel - Single area projection
2. projection_fp16_kernel - Batched FP16 projection
3. hebbian_kernel - Hebbian learning with saturation

Based on implicit random connectivity (hash-based, no storage).
"""

import cupy as cp

# =============================================================================
# KERNEL 1: Simple projection (single area)
# =============================================================================

projection_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void projection(
    const unsigned int* __restrict__ active,
    __half* __restrict__ result,
    const unsigned int k,
    const unsigned int n,
    const unsigned int seed,
    const float p
) {
    unsigned int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= n) return;
    
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
''', 'projection')


# =============================================================================
# KERNEL 2: Batched FP16 projection (multiple areas in parallel)
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
# KERNEL 3: Hebbian learning with saturation
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
    
    // Check if connection exists (hash-based)
    unsigned int hash = seed ^ src;
    hash *= 0x01000193u;
    hash ^= dst;
    hash *= 0x01000193u;
    
    if ((float)(hash & 0xFFFFFFu) / 16777216.0f >= p) return;
    
    // Look for existing learned connection
    unsigned int current_num = *num_learned;
    unsigned int found_idx = 0xFFFFFFFFu;
    
    for (unsigned int l = 0; l < current_num; l++) {
        if (learned_src[l] == src && learned_dst[l] == dst) {
            found_idx = l;
            break;
        }
    }
    
    if (found_idx == 0xFFFFFFFFu) {
        // Add new connection
        unsigned int new_idx = atomicAdd(num_learned, 1);
        if (new_idx < max_learned) {
            learned_src[new_idx] = src;
            learned_dst[new_idx] = dst;
            learned_delta[new_idx] = beta;
        }
    } else {
        // Update with saturation: delta = beta * (1 - w/w_max)
        float current_w = 1.0f + learned_delta[found_idx];
        float update = beta * (1.0f - current_w / w_max);
        if (update > 0) atomicAdd(&learned_delta[found_idx], update);
    }
}
''', 'hebbian_update')

