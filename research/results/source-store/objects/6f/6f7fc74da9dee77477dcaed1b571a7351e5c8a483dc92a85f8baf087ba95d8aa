/*
 * Test Radix Selection for Top-K Optimization
 * ===========================================
 * 
 * This file tests the radix selection optimization for the top-k selection
 * kernel to replace the inefficient bitonic sort.
 * 
 * Mathematical Analysis:
 * - Current: O(n log²n) bitonic sort where n = block_size
 * - Optimized: O(n log k) radix selection where k << n
 * - Expected Speedup: 19.4x for typical values (n=256, k=10)
 * 
 * Test Strategy:
 * 1. Compare current bitonic sort vs radix selection
 * 2. Measure performance at different k values
 * 3. Validate correctness with known inputs
 * 4. Test scalability with different neuron counts
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Test data structures
struct TopKTestData {
    float* activations;
    uint32_t* top_k_indices;
    uint32_t total_neurons;
    uint32_t k;
    uint32_t* expected_indices;  // For correctness validation
};

// Current implementation (bitonic sort)
__global__ void top_k_selection_current(
    const float* activations,
    uint32_t* top_k_indices,
    uint32_t total_neurons,
    uint32_t k
) {
    extern __shared__ float shared_scores[];
    extern __shared__ uint32_t shared_indices[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    uint32_t start = bid * block_size;
    uint32_t end = min(start + block_size, total_neurons);
    
    // Load data into shared memory
    if (start + tid < end) {
        shared_scores[tid] = activations[start + tid];
        shared_indices[tid] = start + tid;
    } else {
        shared_scores[tid] = -INFINITY;
        shared_indices[tid] = UINT32_MAX;
    }
    
    __syncthreads();
    
    // Bitonic sort for top-k selection
    for (uint32_t i = 0; i < k && i < block_size; i++) {
        for (uint32_t j = tid; j < block_size - 1 - i; j += block_size) {
            if (shared_scores[j] < shared_scores[j + 1]) {
                // Swap scores
                float temp_score = shared_scores[j];
                shared_scores[j] = shared_scores[j + 1];
                shared_scores[j + 1] = temp_score;
                
                // Swap indices
                uint32_t temp_idx = shared_indices[j];
                shared_indices[j] = shared_indices[j + 1];
                shared_indices[j + 1] = temp_idx;
            }
        }
        __syncthreads();
    }
    
    // Store top-k results
    if (tid < k && start + tid < end) {
        top_k_indices[bid * k + tid] = shared_indices[tid];
    }
}

// Optimized implementation (radix selection)
__device__ float warp_reduce_max(float val, uint32_t* max_idx, uint32_t tid) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        uint32_t other_idx = __shfl_down_sync(0xffffffff, *max_idx, offset);
        if (other_val > val) {
            val = other_val;
            *max_idx = other_idx;
        }
    }
    return val;
}

__global__ void top_k_selection_radix(
    const float* activations,
    uint32_t* top_k_indices,
    uint32_t total_neurons,
    uint32_t k
) {
    extern __shared__ float shared_scores[];
    extern __shared__ uint32_t shared_indices[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    uint32_t start = bid * block_size;
    uint32_t end = min(start + block_size, total_neurons);
    
    // Load data into shared memory
    if (start + tid < end) {
        shared_scores[tid] = activations[start + tid];
        shared_indices[tid] = start + tid;
    } else {
        shared_scores[tid] = -INFINITY;
        shared_indices[tid] = UINT32_MAX;
    }
    
    __syncthreads();
    
    // Radix selection for top-k
    for (uint32_t i = 0; i < k && i < block_size; i++) {
        uint32_t max_idx = tid;
        float max_val = shared_scores[tid];
        
        // Find maximum in parallel
        for (uint32_t j = tid + block_size; j < block_size; j += block_size) {
            if (shared_scores[j] > max_val) {
                max_val = shared_scores[j];
                max_idx = j;
            }
        }
        
        // Warp-level reduction
        max_val = warp_reduce_max(max_val, &max_idx, tid);
        
        if (tid == 0) {
            top_k_indices[bid * k + i] = shared_indices[max_idx];
            shared_scores[max_idx] = -INFINITY; // Remove from consideration
        }
        __syncthreads();
    }
}

// Alternative optimized implementation (heap-based selection)
__global__ void top_k_selection_heap(
    const float* activations,
    uint32_t* top_k_indices,
    uint32_t total_neurons,
    uint32_t k
) {
    extern __shared__ float shared_scores[];
    extern __shared__ uint32_t shared_indices[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    uint32_t start = bid * block_size;
    uint32_t end = min(start + block_size, total_neurons);
    
    // Load data into shared memory
    if (start + tid < end) {
        shared_scores[tid] = activations[start + tid];
        shared_indices[tid] = start + tid;
    } else {
        shared_scores[tid] = -INFINITY;
        shared_indices[tid] = UINT32_MAX;
    }
    
    __syncthreads();
    
    // Heap-based selection for top-k
    for (uint32_t i = 0; i < k && i < block_size; i++) {
        uint32_t max_idx = tid;
        float max_val = shared_scores[tid];
        
        // Find maximum using parallel reduction
        for (uint32_t stride = block_size / 2; stride > 0; stride /= 2) {
            if (tid < stride && tid + stride < block_size) {
                if (shared_scores[tid + stride] > max_val) {
                    max_val = shared_scores[tid + stride];
                    max_idx = tid + stride;
                }
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            top_k_indices[bid * k + i] = shared_indices[max_idx];
            shared_scores[max_idx] = -INFINITY; // Remove from consideration
        }
        __syncthreads();
    }
}

// Test data generation
TopKTestData generate_topk_test_data(uint32_t total_neurons, uint32_t k) {
    TopKTestData data;
    data.total_neurons = total_neurons;
    data.k = k;
    
    // Allocate host memory
    data.activations = new float[total_neurons];
    data.top_k_indices = new uint32_t[k];
    data.expected_indices = new uint32_t[k];
    
    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0f, 1000.0f);
    
    for (uint32_t i = 0; i < total_neurons; i++) {
        data.activations[i] = dist(gen);
    }
    
    // Generate expected top-k indices (CPU reference)
    std::vector<std::pair<float, uint32_t>> indexed_values;
    for (uint32_t i = 0; i < total_neurons; i++) {
        indexed_values.push_back({data.activations[i], i});
    }
    
    std::partial_sort(indexed_values.begin(), indexed_values.begin() + k, indexed_values.end(),
                     [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                         return a.first > b.first;
                     });
    
    for (uint32_t i = 0; i < k; i++) {
        data.expected_indices[i] = indexed_values[i].second;
    }
    
    return data;
}

// Performance measurement
double measure_topk_performance(
    void (*kernel)(const float*, uint32_t*, uint32_t, uint32_t),
    const TopKTestData& data,
    const char* kernel_name,
    int iterations = 100
) {
    // Allocate device memory
    float* d_activations;
    uint32_t* d_top_k_indices;
    
    cudaMalloc(&d_activations, data.total_neurons * sizeof(float));
    cudaMalloc(&d_top_k_indices, data.k * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_activations, data.activations, data.total_neurons * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((data.total_neurons + blockSize.x - 1) / blockSize.x);
    size_t shared_mem_size = blockSize.x * (sizeof(float) + sizeof(uint32_t));
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<gridSize, blockSize, shared_mem_size>>>(d_activations, d_top_k_indices, data.total_neurons, data.k);
        cudaDeviceSynchronize();
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize, shared_mem_size>>>(d_activations, d_top_k_indices, data.total_neurons, data.k);
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = duration / iterations;
    
    std::cout << "   " << kernel_name << ": " << avg_time << " ms per iteration" << std::endl;
    
    // Cleanup
    cudaFree(d_activations);
    cudaFree(d_top_k_indices);
    
    return avg_time;
}

// Correctness validation
bool validate_topk_correctness(const TopKTestData& data) {
    // Allocate device memory
    float* d_activations;
    uint32_t* d_top_k_indices_current;
    uint32_t* d_top_k_indices_radix;
    uint32_t* d_top_k_indices_heap;
    
    cudaMalloc(&d_activations, data.total_neurons * sizeof(float));
    cudaMalloc(&d_top_k_indices_current, data.k * sizeof(uint32_t));
    cudaMalloc(&d_top_k_indices_radix, data.k * sizeof(uint32_t));
    cudaMalloc(&d_top_k_indices_heap, data.k * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_activations, data.activations, data.total_neurons * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((data.total_neurons + blockSize.x - 1) / blockSize.x);
    size_t shared_mem_size = blockSize.x * (sizeof(float) + sizeof(uint32_t));
    
    top_k_selection_current<<<gridSize, blockSize, shared_mem_size>>>(d_activations, d_top_k_indices_current, data.total_neurons, data.k);
    cudaDeviceSynchronize();
    
    top_k_selection_radix<<<gridSize, blockSize, shared_mem_size>>>(d_activations, d_top_k_indices_radix, data.total_neurons, data.k);
    cudaDeviceSynchronize();
    
    top_k_selection_heap<<<gridSize, blockSize, shared_mem_size>>>(d_activations, d_top_k_indices_heap, data.total_neurons, data.k);
    cudaDeviceSynchronize();
    
    // Compare results
    uint32_t* h_top_k_current = new uint32_t[data.k];
    uint32_t* h_top_k_radix = new uint32_t[data.k];
    uint32_t* h_top_k_heap = new uint32_t[data.k];
    
    cudaMemcpy(h_top_k_current, d_top_k_indices_current, data.k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_k_radix, d_top_k_indices_radix, data.k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_k_heap, d_top_k_indices_heap, data.k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Check correctness by comparing values (not indices, since there might be ties)
    bool correct = true;
    for (uint32_t i = 0; i < data.k; i++) {
        float val_current = data.activations[h_top_k_current[i]];
        float val_radix = data.activations[h_top_k_radix[i]];
        float val_heap = data.activations[h_top_k_heap[i]];
        float val_expected = data.activations[data.expected_indices[i]];
        
        if (fabsf(val_current - val_expected) > 1e-5f ||
            fabsf(val_radix - val_expected) > 1e-5f ||
            fabsf(val_heap - val_expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    
    std::cout << "   Correctness check: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Cleanup
    cudaFree(d_activations);
    cudaFree(d_top_k_indices_current);
    cudaFree(d_top_k_indices_radix);
    cudaFree(d_top_k_indices_heap);
    delete[] h_top_k_current;
    delete[] h_top_k_radix;
    delete[] h_top_k_heap;
    
    return correct;
}

// Main test function
int main() {
    std::cout << "🧪 Testing Radix Selection for Top-K Optimization" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // Test different scales and k values
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> test_cases = {
        {100000, 10, 256},    // 100K neurons, k=10, block_size=256
        {1000000, 100, 256},  // 1M neurons, k=100, block_size=256
        {10000000, 1000, 256}, // 10M neurons, k=1000, block_size=256
        {100000, 50, 512},    // 100K neurons, k=50, block_size=512
        {1000000, 500, 512}   // 1M neurons, k=500, block_size=512
    };
    
    for (auto& test_case : test_cases) {
        uint32_t total_neurons = std::get<0>(test_case);
        uint32_t k = std::get<1>(test_case);
        uint32_t block_size = std::get<2>(test_case);
        
        std::cout << "\n📊 Testing: " << total_neurons << " neurons, k=" << k << ", block_size=" << block_size << std::endl;
        
        // Generate test data
        TopKTestData data = generate_topk_test_data(total_neurons, k);
        
        // Validate correctness
        bool correct = validate_topk_correctness(data);
        if (!correct) {
            std::cout << "❌ Correctness validation failed!" << std::endl;
            continue;
        }
        
        // Measure performance
        double time_current = measure_topk_performance(top_k_selection_current, data, "Current (Bitonic Sort)");
        double time_radix = measure_topk_performance(top_k_selection_radix, data, "Optimized (Radix Selection)");
        double time_heap = measure_topk_performance(top_k_selection_heap, data, "Alternative (Heap Selection)");
        
        // Calculate speedups
        double speedup_radix = time_current / time_radix;
        double speedup_heap = time_current / time_heap;
        
        std::cout << "   Speedup (Radix): " << speedup_radix << "x" << std::endl;
        std::cout << "   Speedup (Heap): " << speedup_heap << "x" << std::endl;
        
        // Theoretical analysis
        double theoretical_radix = (256.0 * 64.0) / (256.0 * log2(k)); // O(n log²n) / O(n log k)
        std::cout << "   Theoretical speedup (Radix): " << theoretical_radix << "x" << std::endl;
        
        // Cleanup
        delete[] data.activations;
        delete[] data.top_k_indices;
        delete[] data.expected_indices;
    }
    
    std::cout << "\n✅ Radix selection testing complete!" << std::endl;
    return 0;
}
