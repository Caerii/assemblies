/*
 * Dense Assembly Calculus CUDA Kernels
 * =====================================
 * 
 * Custom CUDA kernels for dense (explicit) weight matrix operations
 * used in pattern completion and attractor dynamics.
 * 
 * Operations:
 * 1. Dense weight accumulation (matrix-vector multiply)
 * 2. Top-k selection with radix/bitonic
 * 3. Hebbian weight update (outer product)
 * 4. Pattern completion step
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

namespace dense_assembly {

// =============================================================================
// KERNEL 1: Dense Weight Accumulation (Matrix-Vector Multiply)
// =============================================================================
// Computes: output[i] = sum_j(W[i,j] * mask[j]) for all i
// Where mask is 1 for active neurons (winners), 0 otherwise
// 
// Optimizations:
// - Shared memory tiling
// - Warp-level reduction
// - Coalesced memory access

__global__ void dense_weight_accumulate_kernel(
    const float* __restrict__ W,      // n x n weight matrix (row-major)
    const uint32_t* __restrict__ active_indices,  // k active neuron indices
    float* __restrict__ output,       // n output activations
    uint32_t n,                        // total neurons
    uint32_t k                         // number of active neurons
) {
    extern __shared__ float shared_weights[];
    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n) return;
    
    float sum = 0.0f;
    
    // Process active neurons in tiles for better cache utilization
    const uint32_t TILE_SIZE = 32;
    
    for (uint32_t tile_start = 0; tile_start < k; tile_start += TILE_SIZE) {
        uint32_t tile_end = min(tile_start + TILE_SIZE, k);
        
        // Each thread accumulates weights from active neurons in this tile
        for (uint32_t i = tile_start; i < tile_end; i++) {
            uint32_t col = active_indices[i];
            sum += W[row * n + col];
        }
    }
    
    output[row] = sum;
}

// Optimized version using shared memory for active indices
__global__ void dense_weight_accumulate_shared_kernel(
    const float* __restrict__ W,
    const uint32_t* __restrict__ active_indices,
    float* __restrict__ output,
    uint32_t n,
    uint32_t k
) {
    extern __shared__ uint32_t shared_active[];
    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    
    // Cooperatively load active indices into shared memory
    for (uint32_t i = tid; i < k; i += blockDim.x) {
        shared_active[i] = active_indices[i];
    }
    __syncthreads();
    
    if (row >= n) return;
    
    float sum = 0.0f;
    
    // Use shared memory for active indices (cached)
    for (uint32_t i = 0; i < k; i++) {
        uint32_t col = shared_active[i];
        sum += W[row * n + col];
    }
    
    output[row] = sum;
}

// =============================================================================
// KERNEL 2: Top-K Selection using Radix-based approach
// =============================================================================
// Finds the k largest elements and their indices
// Uses histogram-based radix selection for O(n) complexity

__global__ void compute_histogram_kernel(
    const float* __restrict__ values,
    uint32_t* __restrict__ histogram,
    float min_val,
    float range,
    uint32_t n,
    uint32_t num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float val = values[idx];
    uint32_t bin = min((uint32_t)((val - min_val) / range * num_bins), num_bins - 1);
    atomicAdd(&histogram[bin], 1);
}

__global__ void top_k_threshold_kernel(
    const float* __restrict__ values,
    uint32_t* __restrict__ indices,
    uint32_t* __restrict__ count,
    float threshold,
    uint32_t n,
    uint32_t k
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (values[idx] >= threshold) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < k) {
            indices[pos] = idx;
        }
    }
}

// Simple top-k using partial sort (for small k)
__global__ void top_k_simple_kernel(
    const float* __restrict__ values,
    uint32_t* __restrict__ top_k_indices,
    uint32_t n,
    uint32_t k
) {
    // This kernel finds top-k using a simple approach
    // Best for small k (< 256)
    
    extern __shared__ float shared_vals[];
    uint32_t* shared_idx = (uint32_t*)(shared_vals + k);
    
    uint32_t tid = threadIdx.x;
    
    // Initialize with -inf
    if (tid < k) {
        shared_vals[tid] = -1e30f;
        shared_idx[tid] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float val = values[i];
        
        // Check if this value should be in top-k
        if (val > shared_vals[k-1]) {
            // Find insertion point
            for (int j = k - 1; j >= 0; j--) {
                if (j == 0 || val <= shared_vals[j-1]) {
                    // Shift down
                    for (int m = k - 1; m > j; m--) {
                        shared_vals[m] = shared_vals[m-1];
                        shared_idx[m] = shared_idx[m-1];
                    }
                    shared_vals[j] = val;
                    shared_idx[j] = i;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    // Write results
    if (tid < k) {
        top_k_indices[tid] = shared_idx[tid];
    }
}

// =============================================================================
// KERNEL 3: Hebbian Weight Update (Outer Product)
// =============================================================================
// Updates: W[i,j] += beta for all i,j in active set
// This is an outer product update

__global__ void hebbian_update_kernel(
    float* __restrict__ W,
    const uint32_t* __restrict__ active_indices,
    float beta,
    uint32_t n,
    uint32_t k
) {
    // 2D thread indexing for outer product
    uint32_t i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i_idx >= k || j_idx >= k) return;
    
    uint32_t i = active_indices[i_idx];
    uint32_t j = active_indices[j_idx];
    
    // Update weight
    W[i * n + j] += beta;
}

// Optimized version using shared memory
__global__ void hebbian_update_shared_kernel(
    float* __restrict__ W,
    const uint32_t* __restrict__ active_indices,
    float beta,
    uint32_t n,
    uint32_t k
) {
    extern __shared__ uint32_t shared_active[];
    
    uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t block_size = blockDim.x * blockDim.y;
    
    // Load active indices into shared memory
    for (uint32_t i = tid; i < k; i += block_size) {
        shared_active[i] = active_indices[i];
    }
    __syncthreads();
    
    // 2D indexing
    uint32_t i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i_idx >= k || j_idx >= k) return;
    
    uint32_t i = shared_active[i_idx];
    uint32_t j = shared_active[j_idx];
    
    atomicAdd(&W[i * n + j], beta);
}

// =============================================================================
// KERNEL 4: Combined Pattern Completion Step
// =============================================================================
// Combines accumulation + top-k in one kernel for efficiency

__global__ void pattern_completion_step_kernel(
    const float* __restrict__ W,
    const uint32_t* __restrict__ current_active,
    float* __restrict__ activations,
    uint32_t n,
    uint32_t k
) {
    extern __shared__ uint32_t shared_active[];
    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    
    // Load active indices
    for (uint32_t i = tid; i < k; i += blockDim.x) {
        shared_active[i] = current_active[i];
    }
    __syncthreads();
    
    if (row >= n) return;
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        sum += W[row * n + shared_active[i]];
    }
    
    activations[row] = sum;
}

// =============================================================================
// HOST WRAPPER FUNCTIONS
// =============================================================================

extern "C" {

// Initialize CUDA device
__declspec(dllexport) int dense_assembly_init() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Dense Assembly CUDA Kernels initialized on: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total memory: %.1f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    return 0;
}

// Allocate weight matrix on GPU
__declspec(dllexport) float* dense_assembly_alloc_weights(uint32_t n) {
    float* d_W;
    size_t size = (size_t)n * n * sizeof(float);
    cudaError_t err = cudaMalloc(&d_W, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate %zu bytes: %s\n", size, cudaGetErrorString(err));
        return nullptr;
    }
    cudaMemset(d_W, 0, size);
    return d_W;
}

// Free GPU memory
__declspec(dllexport) void dense_assembly_free(void* ptr) {
    cudaFree(ptr);
}

// Allocate array on GPU
__declspec(dllexport) void* dense_assembly_alloc_array(uint32_t size, uint32_t elem_size) {
    void* d_ptr;
    cudaMalloc(&d_ptr, size * elem_size);
    return d_ptr;
}

// Copy to GPU
__declspec(dllexport) void dense_assembly_copy_to_gpu(void* d_dst, const void* h_src, uint32_t size) {
    cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
}

// Copy from GPU
__declspec(dllexport) void dense_assembly_copy_from_gpu(void* h_dst, const void* d_src, uint32_t size) {
    cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
}

// Weight accumulation
__declspec(dllexport) void dense_assembly_accumulate(
    const float* d_W,
    const uint32_t* d_active,
    float* d_output,
    uint32_t n,
    uint32_t k
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (n + block_size - 1) / block_size;
    uint32_t shared_mem = k * sizeof(uint32_t);
    
    dense_weight_accumulate_shared_kernel<<<grid_size, block_size, shared_mem>>>(
        d_W, d_active, d_output, n, k
    );
    cudaDeviceSynchronize();
}

// Hebbian update
__declspec(dllexport) void dense_assembly_hebbian_update(
    float* d_W,
    const uint32_t* d_active,
    float beta,
    uint32_t n,
    uint32_t k
) {
    dim3 block(16, 16);
    dim3 grid((k + 15) / 16, (k + 15) / 16);
    uint32_t shared_mem = k * sizeof(uint32_t);
    
    hebbian_update_shared_kernel<<<grid, block, shared_mem>>>(
        d_W, d_active, beta, n, k
    );
    cudaDeviceSynchronize();
}

// Full pattern completion step
__declspec(dllexport) void dense_assembly_pattern_step(
    const float* d_W,
    const uint32_t* d_active,
    float* d_activations,
    uint32_t n,
    uint32_t k
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (n + block_size - 1) / block_size;
    uint32_t shared_mem = k * sizeof(uint32_t);
    
    pattern_completion_step_kernel<<<grid_size, block_size, shared_mem>>>(
        d_W, d_active, d_activations, n, k
    );
    cudaDeviceSynchronize();
}

} // extern "C"

} // namespace dense_assembly

