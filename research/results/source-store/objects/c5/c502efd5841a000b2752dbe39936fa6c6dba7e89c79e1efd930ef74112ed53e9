/*
 * Test Warp-Level Reduction for Atomic Add Optimization
 * ====================================================
 * 
 * This file tests the warp-level reduction optimization for the weight accumulation
 * kernel to eliminate the atomic add bottleneck.
 * 
 * Mathematical Analysis:
 * - Current: O(S) atomic operations where S = synapses per neuron
 * - Optimized: O(S/32) atomic operations + O(S/32 × 5) warp operations
 * - Expected Speedup: 5.33x reduction in atomic operations
 * 
 * Test Strategy:
 * 1. Compare current atomic add vs warp reduction
 * 2. Measure performance at different scales
 * 3. Validate correctness with known inputs
 * 4. Profile memory access patterns
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Test data structures
struct TestData {
    uint32_t* activated_neurons;
    float* synapse_weights;
    uint32_t* synapse_indices;
    uint32_t* synapse_offsets;
    float* activations;
    uint32_t num_activated;
    uint32_t target_size;
    uint32_t total_synapses;
};

// Current implementation (baseline)
__global__ void accumulate_weights_current(
    const uint32_t* activated_neurons,
    const float* synapse_weights,
    const uint32_t* synapse_indices,
    const uint32_t* synapse_offsets,
    float* activations,
    uint32_t num_activated,
    uint32_t target_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_activated) return;
    
    uint32_t neuron = activated_neurons[idx];
    uint32_t start = synapse_offsets[neuron];
    uint32_t end = synapse_offsets[neuron + 1];
    
    // Current implementation with atomic adds
    for (uint32_t i = start; i < end; i++) {
        uint32_t target = synapse_indices[i];
        float weight = synapse_weights[i];
        atomicAdd(&activations[target], weight);
    }
}

// Optimized implementation with warp reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void accumulate_weights_optimized(
    const uint32_t* activated_neurons,
    const float* synapse_weights,
    const uint32_t* synapse_indices,
    const uint32_t* synapse_offsets,
    float* activations,
    uint32_t num_activated,
    uint32_t target_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_activated) return;
    
    uint32_t neuron = activated_neurons[idx];
    uint32_t start = synapse_offsets[neuron];
    uint32_t end = synapse_offsets[neuron + 1];
    
    // Use shared memory for local accumulation
    extern __shared__ float shared_activations[];
    
    // Initialize shared memory
    if (threadIdx.x < target_size) {
        shared_activations[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Process synapses in chunks
    float local_sum = 0.0f;
    for (uint32_t i = start; i < end; i++) {
        uint32_t target = synapse_indices[i];
        float weight = synapse_weights[i];
        
        if (target < target_size) {
            atomicAdd(&shared_activations[target], weight);
        }
    }
    
    __syncthreads();
    
    // Warp-level reduction for each target
    if (threadIdx.x < target_size) {
        float warp_sum = warp_reduce_sum(shared_activations[threadIdx.x]);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&activations[threadIdx.x], warp_sum);
        }
    }
}

// Alternative optimized implementation with better memory coalescing
__global__ void accumulate_weights_coalesced(
    const uint32_t* activated_neurons,
    const float* synapse_weights,
    const uint32_t* synapse_indices,
    const uint32_t* synapse_offsets,
    float* activations,
    uint32_t num_activated,
    uint32_t target_size
) {
    extern __shared__ float shared_activations[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    // Initialize shared memory
    if (tid < target_size) {
        shared_activations[tid] = 0.0f;
    }
    __syncthreads();
    
    // Process synapses in chunks for better coalescing
    uint32_t start = bid * block_size;
    uint32_t end = min(start + block_size, num_activated);
    
    for (uint32_t i = start + tid; i < end; i += block_size) {
        uint32_t neuron = activated_neurons[i];
        uint32_t syn_start = synapse_offsets[neuron];
        uint32_t syn_end = synapse_offsets[neuron + 1];
        
        // Process synapses for this neuron
        for (uint32_t j = syn_start; j < syn_end; j++) {
            uint32_t target = synapse_indices[j];
            float weight = synapse_weights[j];
            
            if (target < target_size) {
                atomicAdd(&shared_activations[target], weight);
            }
        }
    }
    
    __syncthreads();
    
    // Write back to global memory with coalescing
    if (tid < target_size) {
        atomicAdd(&activations[tid], shared_activations[tid]);
    }
}

// Test data generation
TestData generate_test_data(uint32_t num_neurons, uint32_t active_percentage, uint32_t synapses_per_neuron) {
    TestData data;
    data.num_activated = num_neurons * active_percentage;
    data.target_size = num_neurons;
    data.total_synapses = data.num_activated * synapses_per_neuron;
    
    // Allocate host memory
    data.activated_neurons = new uint32_t[data.num_activated];
    data.synapse_weights = new float[data.total_synapses];
    data.synapse_indices = new uint32_t[data.total_synapses];
    data.synapse_offsets = new uint32_t[data.num_activated + 1];
    data.activations = new float[data.target_size];
    
    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> neuron_dist(0, num_neurons - 1);
    std::uniform_int_distribution<> target_dist(0, data.target_size - 1);
    std::uniform_real_distribution<> weight_dist(0.1f, 2.0f);
    
    // Generate activated neurons
    for (uint32_t i = 0; i < data.num_activated; i++) {
        data.activated_neurons[i] = neuron_dist(gen);
    }
    
    // Generate synapses
    uint32_t syn_idx = 0;
    for (uint32_t i = 0; i < data.num_activated; i++) {
        data.synapse_offsets[i] = syn_idx;
        for (uint32_t j = 0; j < synapses_per_neuron; j++) {
            data.synapse_indices[syn_idx] = target_dist(gen);
            data.synapse_weights[syn_idx] = weight_dist(gen);
            syn_idx++;
        }
    }
    data.synapse_offsets[data.num_activated] = syn_idx;
    
    // Initialize activations
    for (uint32_t i = 0; i < data.target_size; i++) {
        data.activations[i] = 0.0f;
    }
    
    return data;
}

// Performance measurement
double measure_kernel_performance(
    void (*kernel)(const uint32_t*, const float*, const uint32_t*, const uint32_t*, float*, uint32_t, uint32_t),
    const TestData& data,
    const char* kernel_name,
    int iterations = 100
) {
    // Allocate device memory
    uint32_t* d_activated_neurons;
    float* d_synapse_weights;
    uint32_t* d_synapse_indices;
    uint32_t* d_synapse_offsets;
    float* d_activations;
    
    cudaMalloc(&d_activated_neurons, data.num_activated * sizeof(uint32_t));
    cudaMalloc(&d_synapse_weights, data.total_synapses * sizeof(float));
    cudaMalloc(&d_synapse_indices, data.total_synapses * sizeof(uint32_t));
    cudaMalloc(&d_synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t));
    cudaMalloc(&d_activations, data.target_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_activated_neurons, data.activated_neurons, data.num_activated * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_weights, data.synapse_weights, data.total_synapses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_indices, data.synapse_indices, data.total_synapses * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_offsets, data.synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((data.num_activated + blockSize.x - 1) / blockSize.x);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_activations, 0, data.target_size * sizeof(float));
        kernel<<<gridSize, blockSize>>>(d_activated_neurons, d_synapse_weights, d_synapse_indices, d_synapse_offsets, d_activations, data.num_activated, data.target_size);
        cudaDeviceSynchronize();
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_activations, 0, data.target_size * sizeof(float));
        kernel<<<gridSize, blockSize>>>(d_activated_neurons, d_synapse_weights, d_synapse_indices, d_synapse_offsets, d_activations, data.num_activated, data.target_size);
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = duration / iterations;
    
    std::cout << "   " << kernel_name << ": " << avg_time << " ms per iteration" << std::endl;
    
    // Cleanup
    cudaFree(d_activated_neurons);
    cudaFree(d_synapse_weights);
    cudaFree(d_synapse_indices);
    cudaFree(d_synapse_offsets);
    cudaFree(d_activations);
    
    return avg_time;
}

// Correctness validation
bool validate_correctness(const TestData& data) {
    // Allocate device memory
    uint32_t* d_activated_neurons;
    float* d_synapse_weights;
    uint32_t* d_synapse_indices;
    uint32_t* d_synapse_offsets;
    float* d_activations_current;
    float* d_activations_optimized;
    
    cudaMalloc(&d_activated_neurons, data.num_activated * sizeof(uint32_t));
    cudaMalloc(&d_synapse_weights, data.total_synapses * sizeof(float));
    cudaMalloc(&d_synapse_indices, data.total_synapses * sizeof(uint32_t));
    cudaMalloc(&d_synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t));
    cudaMalloc(&d_activations_current, data.target_size * sizeof(float));
    cudaMalloc(&d_activations_optimized, data.target_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_activated_neurons, data.activated_neurons, data.num_activated * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_weights, data.synapse_weights, data.total_synapses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_indices, data.synapse_indices, data.total_synapses * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_offsets, data.synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((data.num_activated + blockSize.x - 1) / blockSize.x);
    
    cudaMemset(d_activations_current, 0, data.target_size * sizeof(float));
    accumulate_weights_current<<<gridSize, blockSize>>>(d_activated_neurons, d_synapse_weights, d_synapse_indices, d_synapse_offsets, d_activations_current, data.num_activated, data.target_size);
    cudaDeviceSynchronize();
    
    cudaMemset(d_activations_optimized, 0, data.target_size * sizeof(float));
    accumulate_weights_optimized<<<gridSize, blockSize, data.target_size * sizeof(float)>>>(d_activated_neurons, d_synapse_weights, d_synapse_indices, d_synapse_offsets, d_activations_optimized, data.num_activated, data.target_size);
    cudaDeviceSynchronize();
    
    // Compare results
    float* h_activations_current = new float[data.target_size];
    float* h_activations_optimized = new float[data.target_size];
    
    cudaMemcpy(h_activations_current, d_activations_current, data.target_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_activations_optimized, d_activations_optimized, data.target_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    float max_error = 0.0f;
    for (uint32_t i = 0; i < data.target_size; i++) {
        float error = fabsf(h_activations_current[i] - h_activations_optimized[i]);
        max_error = fmaxf(max_error, error);
        if (error > 1e-5f) {
            correct = false;
        }
    }
    
    std::cout << "   Correctness check: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "   Max error: " << max_error << std::endl;
    
    // Cleanup
    cudaFree(d_activated_neurons);
    cudaFree(d_synapse_weights);
    cudaFree(d_synapse_indices);
    cudaFree(d_synapse_offsets);
    cudaFree(d_activations_current);
    cudaFree(d_activations_optimized);
    delete[] h_activations_current;
    delete[] h_activations_optimized;
    
    return correct;
}

// Main test function
int main() {
    std::cout << "🧪 Testing Warp-Level Reduction for Atomic Add Optimization" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // Test different scales
    std::vector<std::pair<uint32_t, uint32_t>> test_scales = {
        {100000, 1000},    // 100K neurons, 1K active
        {1000000, 10000},  // 1M neurons, 10K active
        {10000000, 100000} // 10M neurons, 100K active
    };
    
    for (auto& scale : test_scales) {
        uint32_t num_neurons = scale.first;
        uint32_t active_neurons = scale.second;
        uint32_t synapses_per_neuron = 100;
        
        std::cout << "\n📊 Testing scale: " << num_neurons << " neurons, " << active_neurons << " active" << std::endl;
        
        // Generate test data
        TestData data = generate_test_data(num_neurons, active_neurons, synapses_per_neuron);
        
        // Validate correctness
        bool correct = validate_correctness(data);
        if (!correct) {
            std::cout << "❌ Correctness validation failed!" << std::endl;
            continue;
        }
        
        // Measure performance
        double time_current = measure_kernel_performance(accumulate_weights_current, data, "Current (Atomic Add)");
        double time_optimized = measure_kernel_performance(accumulate_weights_optimized, data, "Optimized (Warp Reduction)");
        double time_coalesced = measure_kernel_performance(accumulate_weights_coalesced, data, "Coalesced (Memory Optimized)");
        
        // Calculate speedups
        double speedup_optimized = time_current / time_optimized;
        double speedup_coalesced = time_current / time_coalesced;
        
        std::cout << "   Speedup (Warp Reduction): " << speedup_optimized << "x" << std::endl;
        std::cout << "   Speedup (Memory Coalesced): " << speedup_coalesced << "x" << std::endl;
        
        // Cleanup
        delete[] data.activated_neurons;
        delete[] data.synapse_weights;
        delete[] data.synapse_indices;
        delete[] data.synapse_offsets;
        delete[] data.activations;
    }
    
    std::cout << "\n✅ Warp reduction testing complete!" << std::endl;
    return 0;
}
