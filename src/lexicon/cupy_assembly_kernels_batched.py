"""
Batched CUDA Kernels for Assembly Calculus
==========================================

Key insight: Batching multiple areas together gives massive speedup
because GPU is underutilized with single-area operations.

At n=1M, k=50:
- Single area: 2,913 proj/sec
- Batch of 8:  11,492 proj/sec (4x faster!)
- Batch of 16: 16,692 proj/sec (5.7x faster!)
"""

import cupy as cp
import torch
import numpy as np
import time
from typing import List, Dict, Optional, Tuple

# =============================================================================
# BATCHED PROJECTION KERNEL
# =============================================================================

batched_projection_kernel = cp.RawKernel(r'''
extern "C" __global__
void batched_projection(
    const unsigned int* active_batch,  // [batch_size, k] flattened
    float* result_batch,               // [batch_size, n] flattened
    const unsigned int k,
    const unsigned int n,
    const unsigned int batch_size,
    const unsigned int* seeds,         // [batch_size] different seed per area
    const float p
) {
    // 2D grid: x = destination neuron, y = batch index
    unsigned int dst = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int batch_idx = blockIdx.y;
    
    if (dst >= n || batch_idx >= batch_size) return;
    
    // Get this batch's data
    const unsigned int* active = active_batch + batch_idx * k;
    float* result = result_batch + batch_idx * n;
    unsigned int seed = seeds[batch_idx];
    
    // Load active indices to shared memory
    extern __shared__ unsigned int s_active[];
    for (unsigned int i = threadIdx.x; i < k; i += blockDim.x) {
        s_active[i] = active[i];
    }
    __syncthreads();
    
    // Compute activation
    unsigned int threshold = (unsigned int)(p * 16777216.0f);
    float sum = 0.0f;
    
    #pragma unroll 4
    for (unsigned int i = 0; i < k; i++) {
        unsigned int src = s_active[i];
        unsigned int hash = (src * 2654435761u) ^ (dst * 2246822519u) ^ seed;
        sum += (float)((hash & 0xFFFFFFu) < threshold);
    }
    
    result[dst] = sum;
}
''', 'batched_projection')


class BatchedAssemblySystem:
    """
    Process multiple brain areas in parallel using batched GPU operations.
    
    Instead of processing each area sequentially, we batch them together
    for massive speedup (4-6x faster).
    """
    
    def __init__(self, n: int, k: int, num_areas: int = 8,
                 p: float = 0.05, beta: float = 0.1, w_max: float = 10.0,
                 base_seed: int = 42):
        self.n = n
        self.k = k
        self.num_areas = num_areas
        self.p = p
        self.beta = beta
        self.w_max = w_max
        
        # Different seed for each area
        self.seeds = cp.array([base_seed + i * 1000 for i in range(num_areas)], dtype=cp.uint32)
        
        # Batched buffers
        self.active_batch = cp.zeros((num_areas, k), dtype=cp.uint32)
        self.result_batch = cp.zeros((num_areas, n), dtype=cp.float32)
        
        # PyTorch tensor for batched top-k (FP16 for speed)
        self.result_torch = torch.zeros((num_areas, n), device='cuda', dtype=torch.float16)
        
        # Learned weights per area (sparse COO)
        self.max_learned_per_area = k * k * 500
        self.learned_src = [cp.zeros(self.max_learned_per_area, dtype=cp.uint32) for _ in range(num_areas)]
        self.learned_dst = [cp.zeros(self.max_learned_per_area, dtype=cp.uint32) for _ in range(num_areas)]
        self.learned_delta = [cp.zeros(self.max_learned_per_area, dtype=cp.float32) for _ in range(num_areas)]
        self.num_learned = [cp.zeros(1, dtype=cp.uint32) for _ in range(num_areas)]
        
        # Previous activations per area
        self.prev_active = [None for _ in range(num_areas)]
        
        # Kernel config
        self.block_size = 512
        self.grid_x = (n + self.block_size - 1) // self.block_size
    
    def project_batch(self, inputs: List[cp.ndarray], learn: bool = True) -> List[cp.ndarray]:
        """
        Project multiple inputs in parallel.
        
        Args:
            inputs: List of active indices for each area
            learn: Whether to apply Hebbian learning
            
        Returns:
            List of winner indices for each area
        """
        batch_size = min(len(inputs), self.num_areas)
        
        # Copy inputs to batch buffer
        for i, inp in enumerate(inputs[:batch_size]):
            k_in = min(len(inp), self.k)
            self.active_batch[i, :k_in] = inp[:k_in]
        
        # Clear results
        self.result_batch[:batch_size].fill(0)
        
        # Batched projection kernel
        batched_projection_kernel(
            (self.grid_x, batch_size), (self.block_size,),
            (self.active_batch, self.result_batch,
             cp.uint32(self.k), cp.uint32(self.n), cp.uint32(batch_size),
             self.seeds, cp.float32(self.p)),
            shared_mem=self.k * 4
        )
        
        # Copy to PyTorch tensor for batched top-k
        self.result_torch[:batch_size].copy_(
            torch.as_tensor(self.result_batch[:batch_size], device='cuda')
        )
        
        # Batched top-k (FP16, unsorted)
        _, top_indices = torch.topk(self.result_torch[:batch_size], self.k, dim=1, sorted=False)
        
        # Convert back to CuPy and extract results
        top_indices_cp = cp.asarray(top_indices)
        results = [top_indices_cp[i].copy() for i in range(batch_size)]
        
        # Hebbian update (still sequential for now - could batch this too)
        if learn:
            for i, (inp, winners) in enumerate(zip(inputs[:batch_size], results)):
                if self.prev_active[i] is not None:
                    self._hebbian_update(i, self.prev_active[i], winners)
                self.prev_active[i] = winners.copy()
        
        return results
    
    def _hebbian_update(self, area_idx: int, prev: cp.ndarray, new: cp.ndarray):
        """Update learned weights for one area."""
        from cupy_assembly_kernels import hebbian_update_kernel
        
        update_grid = (self.k * self.k + self.block_size - 1) // self.block_size
        hebbian_update_kernel(
            (update_grid,), (self.block_size,),
            (self.learned_src[area_idx], self.learned_dst[area_idx], 
             self.learned_delta[area_idx], self.num_learned[area_idx],
             prev, new, cp.uint32(self.k), cp.float32(self.beta),
             cp.float32(self.w_max), cp.uint32(self.max_learned_per_area),
             self.seeds[area_idx], cp.float32(self.p))
        )
    
    def memory_usage(self) -> dict:
        """Get memory usage."""
        total_learned = sum(int(nl[0]) for nl in self.num_learned)
        return {
            'areas': self.num_areas,
            'total_learned': total_learned,
            'batch_buffers_mb': (self.num_areas * self.n * 4 * 2) / 1e6,
            'learned_mb': total_learned * 12 / 1e6
        }


def benchmark():
    """Benchmark batched vs sequential."""
    print("=" * 70)
    print("BATCHED vs SEQUENTIAL BENCHMARK")
    print("=" * 70)
    
    from cupy_assembly_kernels import ImplicitAssemblyArea
    
    n = 1000000
    k = 50
    num_areas = 8
    
    # Sequential (8 separate areas)
    areas_seq = [ImplicitAssemblyArea(n, k, seed=42+i) for i in range(num_areas)]
    inputs = [cp.random.randint(0, n, k, dtype=cp.uint32) for _ in range(num_areas)]
    
    # Warmup
    for area, inp in zip(areas_seq, inputs):
        area.project(inp, learn=False)
    
    # Benchmark sequential
    n_iter = 100
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        for area, inp in zip(areas_seq, inputs):
            area.project(inp, learn=False)
    cp.cuda.Stream.null.synchronize()
    seq_time = (time.perf_counter() - start) / n_iter * 1000
    
    print(f"Sequential ({num_areas} areas):")
    print(f"  Total: {seq_time:.2f}ms")
    print(f"  Per area: {seq_time/num_areas:.2f}ms")
    print(f"  Throughput: {num_areas * 1000 / seq_time:.0f} proj/sec")
    print()
    
    # Batched
    batched = BatchedAssemblySystem(n, k, num_areas=num_areas)
    
    # Warmup
    for _ in range(10):
        batched.project_batch(inputs, learn=False)
    
    # Benchmark batched
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        batched.project_batch(inputs, learn=False)
    cp.cuda.Stream.null.synchronize()
    batch_time = (time.perf_counter() - start) / n_iter * 1000
    
    print(f"Batched ({num_areas} areas):")
    print(f"  Total: {batch_time:.2f}ms")
    print(f"  Per area: {batch_time/num_areas:.2f}ms")
    print(f"  Throughput: {num_areas * 1000 / batch_time:.0f} proj/sec")
    print()
    
    speedup = seq_time / batch_time
    print(f"Speedup: {speedup:.2f}x")
    print()
    
    # Verify correctness (should produce similar results)
    print("Verifying correctness...")
    results_seq = [areas_seq[i].project(inputs[i], learn=False) for i in range(num_areas)]
    results_batch = batched.project_batch(inputs, learn=False)
    
    for i in range(num_areas):
        overlap = len(set(results_seq[i].get()) & set(results_batch[i].get())) / k
        print(f"  Area {i}: {overlap*100:.1f}% overlap")


if __name__ == "__main__":
    benchmark()

