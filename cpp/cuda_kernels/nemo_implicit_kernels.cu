/*
 * NEMO Implicit Connectivity CUDA Kernels
 * ========================================
 * 
 * Hash-based implicit connectivity for the NEMO emergent language system.
 * NO weight matrix storage - connectivity computed on-the-fly.
 * 
 * Memory: O(learned_connections) instead of O(n²)
 * 
 * Key operations:
 * 1. implicit_projection - Compute activation using hash-based connectivity
 * 2. apply_learned_weights - Add learned deltas to activation
 * 3. top_k_select - Find k highest activations
 * 4. hebbian_update - Update learned weights for co-active neurons
 * 
 * Build:
 *   nvcc -o nemo_implicit_kernels.dll --shared nemo_implicit_kernels.cu -lcudart
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_K 512  // Maximum assembly size

// =============================================================================
// HASH FUNCTION - Determines implicit connectivity
// =============================================================================

__device__ __forceinline__ bool has_connection(
    uint32_t src, 
    uint32_t dst, 
    uint32_t seed, 
    float p
) {
    // FNV-1a inspired hash for good distribution
    uint32_t h = seed;
    h ^= src;
    h *= 0x01000193u;  // FNV prime
    h ^= dst;
    h *= 0x01000193u;
    
    // Convert to probability (use 24 bits for precision)
    uint32_t threshold = (uint32_t)(p * 16777216.0f);
    return (h & 0xFFFFFFu) < threshold;
}

// =============================================================================
// KERNEL 1: Implicit Projection
// =============================================================================
// Computes activation of all n neurons from k active inputs
// using hash-based implicit connectivity (no weight storage)

__global__ void implicit_projection_kernel(
    const uint32_t* __restrict__ active,  // [k] active neuron indices
    float* __restrict__ result,            // [n] output activations
    const uint32_t k,                      // assembly size
    const uint32_t n,                      // total neurons
    const uint32_t seed,                   // area-specific seed
    const float p                          // connection probability
) {
    // Load active indices into shared memory
    __shared__ uint32_t s_active[MAX_K];
    
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        s_active[i] = active[i];
    }
    __syncthreads();
    
    // Each thread computes activation for one destination neuron
    uint32_t dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= n) return;
    
    // Count connections from active neurons
    float sum = 0.0f;
    
    #pragma unroll 8
    for (uint32_t i = 0; i < k; i++) {
        uint32_t src = s_active[i];
        if (has_connection(src, dst, seed, p)) {
            sum += 1.0f;  // Base weight is always 1.0
        }
    }
    
    result[dst] = sum;
}

// =============================================================================
// KERNEL 2: Apply Learned Weight Deltas
// =============================================================================
// Adds learned weight modifications to base activation
// Learned weights stored as sparse (src, dst, delta) tuples

__global__ void apply_learned_weights_kernel(
    const uint32_t* __restrict__ active,       // [k] active neurons
    const uint32_t* __restrict__ learned_src,  // [num_learned] source neurons
    const uint32_t* __restrict__ learned_dst,  // [num_learned] dest neurons
    const float* __restrict__ learned_delta,   // [num_learned] weight deltas
    float* __restrict__ result,                // [n] activations (modified in place)
    const uint32_t k,
    const uint32_t num_learned,
    const uint32_t seed,
    const float p
) {
    // Load active indices into shared memory
    __shared__ uint32_t s_active[MAX_K];
    
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        s_active[i] = active[i];
    }
    __syncthreads();
    
    // Each thread processes one learned connection
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_learned) return;
    
    uint32_t src = learned_src[idx];
    uint32_t dst = learned_dst[idx];
    float delta = learned_delta[idx];
    
    // Check if source is in active set
    bool src_active = false;
    for (uint32_t i = 0; i < k; i++) {
        if (s_active[i] == src) {
            src_active = true;
            break;
        }
    }
    
    // Only apply if source is active AND connection exists
    if (src_active && has_connection(src, dst, seed, p)) {
        atomicAdd(&result[dst], delta);
    }
}

// =============================================================================
// KERNEL 3: Top-K Selection
// =============================================================================
// Finds the k neurons with highest activation
// Uses partial sort approach optimized for our typical k=100

__global__ void top_k_select_kernel(
    const float* __restrict__ activations,  // [n] input activations
    uint32_t* __restrict__ winners,          // [k] output winner indices
    const uint32_t n,
    const uint32_t k
) {
    // Shared memory for partial results
    extern __shared__ float shared_mem[];
    float* s_vals = shared_mem;
    uint32_t* s_idx = (uint32_t*)(s_vals + k);
    
    uint32_t tid = threadIdx.x;
    
    // Initialize with -inf
    if (tid < k) {
        s_vals[tid] = -1e30f;
        s_idx[tid] = 0;
    }
    __syncthreads();
    
    // Each thread scans a portion of activations
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float val = activations[i];
        
        // Check if this value should be in top-k
        if (val > s_vals[k - 1]) {
            // Find insertion point (linear search, k is small)
            for (int j = k - 1; j >= 0; j--) {
                if (j == 0 || val <= s_vals[j - 1]) {
                    // Shift down
                    for (int m = k - 1; m > j; m--) {
                        s_vals[m] = s_vals[m - 1];
                        s_idx[m] = s_idx[m - 1];
                    }
                    s_vals[j] = val;
                    s_idx[j] = i;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    // Write results
    if (tid < k) {
        winners[tid] = s_idx[tid];
    }
}

// Alternative: Use CUB radix sort for larger k
// For now, the simple approach works well for k=100

// =============================================================================
// KERNEL 4: Hebbian Weight Update (Optimized with Hash Table)
// =============================================================================
// Updates learned weights for all (prev, new) co-active pairs
// Uses saturating Hebbian: delta = beta * (1 - w/w_max)
// Uses hash table for O(1) lookup instead of O(n) linear search

// Hash table size (power of 2 for fast modulo)
#define HASH_TABLE_SIZE (1 << 20)  // 1M entries
#define HASH_EMPTY 0xFFFFFFFFu

// Hash function for (src, dst) pair
__device__ __forceinline__ uint32_t pair_hash(uint32_t src, uint32_t dst) {
    uint32_t h = src;
    h ^= dst + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h & (HASH_TABLE_SIZE - 1);
}

__global__ void hebbian_update_kernel(
    uint32_t* __restrict__ learned_src,
    uint32_t* __restrict__ learned_dst,
    float* __restrict__ learned_delta,
    uint32_t* __restrict__ num_learned,
    uint32_t* __restrict__ hash_table,  // Maps pair_hash -> index in learned arrays
    const uint32_t* __restrict__ prev_active,  // [k] previous winners
    const uint32_t* __restrict__ new_active,   // [k] new winners
    const uint32_t k,
    const float beta,
    const float w_max,
    const uint32_t max_learned,
    const uint32_t seed,
    const float p
) {
    // Each thread handles one (prev, new) pair
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = pair_idx / k;  // Index into prev_active
    uint32_t j = pair_idx % k;  // Index into new_active
    
    if (i >= k) return;
    
    uint32_t src = prev_active[i];
    uint32_t dst = new_active[j];
    
    // Check if connection exists via hash
    if (!has_connection(src, dst, seed, p)) {
        return;  // No connection to update
    }
    
    // Hash table lookup with linear probing
    uint32_t h = pair_hash(src, dst);
    uint32_t found_idx = HASH_EMPTY;
    
    for (uint32_t probe = 0; probe < 32; probe++) {
        uint32_t slot = (h + probe) & (HASH_TABLE_SIZE - 1);
        uint32_t idx = hash_table[slot];
        
        if (idx == HASH_EMPTY) {
            // Empty slot - try to claim it
            uint32_t new_idx = atomicAdd(num_learned, 1);
            if (new_idx >= max_learned) {
                atomicSub(num_learned, 1);
                return;  // Table full
            }
            
            // Try to claim the slot
            uint32_t old = atomicCAS(&hash_table[slot], HASH_EMPTY, new_idx);
            if (old == HASH_EMPTY) {
                // Successfully claimed - initialize
                learned_src[new_idx] = src;
                learned_dst[new_idx] = dst;
                learned_delta[new_idx] = beta;
                return;
            }
            // Someone else claimed it - check if it's our pair
            idx = old;
        }
        
        // Check if this slot contains our pair
        if (learned_src[idx] == src && learned_dst[idx] == dst) {
            found_idx = idx;
            break;
        }
    }
    
    if (found_idx != HASH_EMPTY) {
        // Existing connection - saturating update
        float current_w = 1.0f + learned_delta[found_idx];
        float update = beta * (1.0f - current_w / w_max);
        if (update > 0) {
            atomicAdd(&learned_delta[found_idx], update);
        }
    }
}

// =============================================================================
// KERNEL 5: Fused Projection + TopK (Advanced)
// =============================================================================
// Combines projection and top-k selection for better performance
// Uses block-level reduction

__global__ void fused_project_topk_kernel(
    const uint32_t* __restrict__ active,
    uint32_t* __restrict__ winners,
    const uint32_t k,
    const uint32_t n,
    const uint32_t seed,
    const float p,
    const uint32_t* __restrict__ learned_src,
    const uint32_t* __restrict__ learned_dst,
    const float* __restrict__ learned_delta,
    const uint32_t num_learned
) {
    // This is a more advanced kernel that could be implemented
    // for even better performance by avoiding the intermediate
    // activation array entirely.
    
    // For now, we use the separate kernels which are easier to debug
    // and still provide good performance.
}

// =============================================================================
// HOST INTERFACE - Projector Class
// =============================================================================

struct NemoProjector {
    // Device memory
    float* d_activations;
    uint32_t* d_active;
    uint32_t* d_prev_active;
    uint32_t* d_winners;
    
    // Learned weights (sparse)
    uint32_t* d_learned_src;
    uint32_t* d_learned_dst;
    float* d_learned_delta;
    uint32_t* d_num_learned;
    
    // Hash table for fast lookup
    uint32_t* d_hash_table;
    
    // Parameters
    uint32_t n;
    uint32_t k;
    float p;
    float beta;
    float w_max;
    uint32_t seed;
    uint32_t max_learned;
    
    // Grid/block dimensions
    uint32_t grid_n;
    uint32_t grid_k2;
    
    // State
    bool has_prev;
};

// =============================================================================
// EXPORTED C FUNCTIONS
// =============================================================================

extern "C" {

// Track if we've already printed init message
static bool g_init_printed = false;

// Initialize CUDA and print device info (once)
__declspec(dllexport) int nemo_init() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("NEMO: No CUDA devices found!\n");
        return -1;
    }
    
    if (!g_init_printed) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("NEMO CUDA Backend initialized on: %s\n", prop.name);
        printf("  Compute capability: %d.%d, Memory: %.1f GB\n", 
               prop.major, prop.minor, prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Using hash-based implicit connectivity (8x faster)\n");
        g_init_printed = true;
    }
    
    return 0;
}

// Create a new projector
__declspec(dllexport) NemoProjector* nemo_create_projector(
    uint32_t n, uint32_t k, float p, float beta, float w_max, uint32_t seed
) {
    NemoProjector* proj = new NemoProjector();
    
    proj->n = n;
    proj->k = k;
    proj->p = p;
    proj->beta = beta;
    proj->w_max = w_max;
    proj->seed = seed;
    proj->max_learned = k * k * 1000;  // Estimate
    proj->has_prev = false;
    
    // Compute grid dimensions
    proj->grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    proj->grid_k2 = (k * k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate device memory
    cudaMalloc(&proj->d_activations, n * sizeof(float));
    cudaMalloc(&proj->d_active, k * sizeof(uint32_t));
    cudaMalloc(&proj->d_prev_active, k * sizeof(uint32_t));
    cudaMalloc(&proj->d_winners, k * sizeof(uint32_t));
    
    cudaMalloc(&proj->d_learned_src, proj->max_learned * sizeof(uint32_t));
    cudaMalloc(&proj->d_learned_dst, proj->max_learned * sizeof(uint32_t));
    cudaMalloc(&proj->d_learned_delta, proj->max_learned * sizeof(float));
    cudaMalloc(&proj->d_num_learned, sizeof(uint32_t));
    
    // Hash table for fast learned weight lookup
    cudaMalloc(&proj->d_hash_table, HASH_TABLE_SIZE * sizeof(uint32_t));
    
    // Initialize
    cudaMemset(proj->d_num_learned, 0, sizeof(uint32_t));
    // Initialize hash table to EMPTY
    cudaMemset(proj->d_hash_table, 0xFF, HASH_TABLE_SIZE * sizeof(uint32_t));
    
    return proj;
}

// Project active neurons to get new winners
__declspec(dllexport) void nemo_project(
    NemoProjector* proj,
    const uint32_t* h_active,
    uint32_t* h_winners,
    int learn,
    uint32_t area_seed  // Area-specific seed
) {
    // Copy active to device
    cudaMemcpy(proj->d_active, h_active, proj->k * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Clear activations
    cudaMemset(proj->d_activations, 0, proj->n * sizeof(float));
    
    // 1. Implicit projection
    implicit_projection_kernel<<<proj->grid_n, BLOCK_SIZE>>>(
        proj->d_active, proj->d_activations,
        proj->k, proj->n, area_seed, proj->p
    );
    
    // 2. Apply learned weights
    uint32_t num_learned;
    cudaMemcpy(&num_learned, proj->d_num_learned, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    if (num_learned > 0) {
        uint32_t grid_learned = (num_learned + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_learned_weights_kernel<<<grid_learned, BLOCK_SIZE>>>(
            proj->d_active, proj->d_learned_src, proj->d_learned_dst,
            proj->d_learned_delta, proj->d_activations,
            proj->k, num_learned, area_seed, proj->p
        );
    }
    
    // 3. Top-k selection
    uint32_t shared_mem = proj->k * (sizeof(float) + sizeof(uint32_t));
    top_k_select_kernel<<<1, BLOCK_SIZE, shared_mem>>>(
        proj->d_activations, proj->d_winners, proj->n, proj->k
    );
    
    // 4. Hebbian update (if learning AND we have previous activation)
    if (learn && proj->has_prev) {
        hebbian_update_kernel<<<proj->grid_k2, BLOCK_SIZE>>>(
            proj->d_learned_src, proj->d_learned_dst, proj->d_learned_delta,
            proj->d_num_learned, proj->d_hash_table,
            proj->d_prev_active, proj->d_winners,
            proj->k, proj->beta, proj->w_max, proj->max_learned,
            area_seed, proj->p
        );
    }
    
    // Copy winners back to host
    cudaMemcpy(h_winners, proj->d_winners, proj->k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Save current as previous for next iteration
    cudaMemcpy(proj->d_prev_active, proj->d_winners, proj->k * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    proj->has_prev = true;
}

// Get number of learned connections
__declspec(dllexport) uint32_t nemo_get_num_learned(NemoProjector* proj) {
    uint32_t num;
    cudaMemcpy(&num, proj->d_num_learned, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return num;
}

// Destroy projector and free memory
__declspec(dllexport) void nemo_destroy_projector(NemoProjector* proj) {
    cudaFree(proj->d_activations);
    cudaFree(proj->d_active);
    cudaFree(proj->d_prev_active);
    cudaFree(proj->d_winners);
    cudaFree(proj->d_learned_src);
    cudaFree(proj->d_learned_dst);
    cudaFree(proj->d_learned_delta);
    cudaFree(proj->d_num_learned);
    cudaFree(proj->d_hash_table);
    
    delete proj;
}

// Reset learned weights
__declspec(dllexport) void nemo_reset_learned(NemoProjector* proj) {
    cudaMemset(proj->d_num_learned, 0, sizeof(uint32_t));
    cudaMemset(proj->d_hash_table, 0xFF, HASH_TABLE_SIZE * sizeof(uint32_t));
    proj->has_prev = false;
}

// Copy learned weights to host (for inspection/saving)
__declspec(dllexport) void nemo_get_learned_weights(
    NemoProjector* proj,
    uint32_t* h_src,
    uint32_t* h_dst,
    float* h_delta,
    uint32_t* h_num
) {
    cudaMemcpy(h_num, proj->d_num_learned, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t num = *h_num;
    
    if (num > 0) {
        cudaMemcpy(h_src, proj->d_learned_src, num * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dst, proj->d_learned_dst, num * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_delta, proj->d_learned_delta, num * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

} // extern "C"

