#!/usr/bin/env python3
"""
Custom CUDA Kernels using PyTorch's JIT compilation
====================================================

This uses torch.utils.cpp_extension to compile custom CUDA kernels at runtime.
No need for separate nvcc compilation!
"""

import torch
from torch.utils.cpp_extension import load_inline
import os
import time

# CUDA source code for dense assembly operations
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized weight accumulation kernel
__global__ void dense_accumulate_kernel(
    const float* __restrict__ W,
    const int64_t* __restrict__ active_indices,
    float* __restrict__ output,
    int n,
    int k
) {
    extern __shared__ int64_t shared_active[];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Cooperatively load active indices into shared memory
    for (int i = tid; i < k; i += blockDim.x) {
        shared_active[i] = active_indices[i];
    }
    __syncthreads();
    
    if (row >= n) return;
    
    float sum = 0.0f;
    
    // Accumulate weights from active neurons
    #pragma unroll 4
    for (int i = 0; i < k; i++) {
        int col = shared_active[i];
        sum += W[row * n + col];
    }
    
    output[row] = sum;
}

// Optimized Hebbian update kernel (outer product)
__global__ void hebbian_update_kernel(
    float* __restrict__ W,
    const int64_t* __restrict__ active_indices,
    float beta,
    int n,
    int k
) {
    extern __shared__ int64_t shared_active[];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_size = blockDim.x * blockDim.y;
    
    // Load active indices into shared memory
    for (int i = tid; i < k; i += block_size) {
        shared_active[i] = active_indices[i];
    }
    __syncthreads();
    
    int i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i_idx >= k || j_idx >= k) return;
    
    int i = shared_active[i_idx];
    int j = shared_active[j_idx];
    
    atomicAdd(&W[i * n + j], beta);
}

// Host functions
torch::Tensor dense_accumulate_cuda(torch::Tensor W, torch::Tensor active_indices) {
    int n = W.size(0);
    int k = active_indices.size(0);
    
    auto output = torch::zeros({n}, W.options());
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    int shared_mem = k * sizeof(int64_t);
    
    dense_accumulate_kernel<<<grid_size, block_size, shared_mem>>>(
        W.data_ptr<float>(),
        active_indices.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        n, k
    );
    
    return output;
}

void hebbian_update_cuda(torch::Tensor W, torch::Tensor active_indices, float beta) {
    int n = W.size(0);
    int k = active_indices.size(0);
    
    dim3 block(16, 16);
    dim3 grid((k + 15) / 16, (k + 15) / 16);
    int shared_mem = k * sizeof(int64_t);
    
    hebbian_update_kernel<<<grid, block, shared_mem>>>(
        W.data_ptr<float>(),
        active_indices.data_ptr<int64_t>(),
        beta,
        n, k
    );
}
"""

cpp_source = """
torch::Tensor dense_accumulate_cuda(torch::Tensor W, torch::Tensor active_indices);
void hebbian_update_cuda(torch::Tensor W, torch::Tensor active_indices, float beta);
"""


def load_custom_kernels():
    """Load custom CUDA kernels using JIT compilation"""
    print("Compiling custom CUDA kernels (this may take a minute on first run)...")
    
    try:
        module = load_inline(
            name='dense_assembly_cuda',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=['dense_accumulate_cuda', 'hebbian_update_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        print("Custom CUDA kernels compiled successfully!")
        return module
    except Exception as e:
        print(f"Failed to compile custom kernels: {e}")
        return None


def pattern_completion_custom_cuda(n: int, k: int, beta: float = 0.5,
                                   train_rounds: int = 50, recovery_rounds: int = 10,
                                   cuda_module=None):
    """Pattern completion using custom CUDA kernels"""
    if cuda_module is None:
        raise RuntimeError("Custom CUDA module not loaded")
    
    device = torch.device('cuda')
    
    # Warmup
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    # Initialize on GPU
    W = torch.zeros((n, n), dtype=torch.float32, device=device)
    
    torch.manual_seed(42)
    stim = torch.randperm(n, device=device)[:k]
    stim_input = torch.zeros(n, dtype=torch.float32, device=device)
    stim_input[stim] = 1.0
    winners = stim.clone()
    
    # Training
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    
    for _ in range(train_rounds):
        # Custom kernel for accumulation
        recurrent = cuda_module.dense_accumulate_cuda(W, winners)
        total = recurrent + stim_input
        
        # Top-k (use PyTorch's optimized version)
        _, new_winners = torch.topk(total, k)
        
        # Custom kernel for Hebbian update
        cuda_module.hebbian_update_cuda(W, new_winners, beta)
        
        winners = new_winners
    
    torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start
    
    original = set(winners.cpu().numpy())
    
    # Pattern completion
    torch.cuda.synchronize()
    comp_start = time.perf_counter()
    
    partial_list = list(original)[:k//2]
    partial = torch.tensor(partial_list, dtype=torch.long, device=device)
    
    for _ in range(recovery_rounds):
        recurrent = cuda_module.dense_accumulate_cuda(W, partial)
        _, partial = torch.topk(recurrent, k)
    
    torch.cuda.synchronize()
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial.cpu().numpy())
    recovery = len(recovered & original) / k
    
    total_time = time.perf_counter() - start
    
    # Clean up
    del W, stim, stim_input, winners, partial
    torch.cuda.empty_cache()
    
    return {
        'method': 'Custom CUDA Kernels',
        'n': n,
        'k': k,
        'train_time': train_time,
        'completion_time': comp_time,
        'recovery_rate': recovery,
        'total_time': total_time
    }


if __name__ == "__main__":
    print("Testing Custom CUDA Kernels for Dense Assembly Operations")
    print("=" * 60)
    
    # Load kernels
    cuda_module = load_custom_kernels()
    
    if cuda_module is not None:
        print("\nRunning pattern completion test...")
        
        for n in [1000, 5000, 10000, 20000, 30000]:
            k = int(n ** 0.5)
            try:
                result = pattern_completion_custom_cuda(n, k, cuda_module=cuda_module)
                status = "OK" if result['recovery_rate'] > 0.9 else "FAIL"
                print(f"n={n:>6}, k={k:>4}: train={result['train_time']:.3f}s, "
                      f"complete={result['completion_time']:.4f}s, "
                      f"recovery={result['recovery_rate']:.0%} [{status}]")
            except Exception as e:
                print(f"n={n:>6}, k={k:>4}: ERROR - {e}")
    else:
        print("Custom CUDA kernels not available")

