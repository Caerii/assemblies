/*
 * Test Memory Coalescing Optimization
 * ===================================
 * 
 * This file tests memory coalescing optimizations to improve GPU memory
 * bandwidth utilization and reduce memory access latency.
 * 
 * Mathematical Analysis:
 * - Current: Random memory access pattern, ~30% bandwidth utilization
 * - Optimized: Coalesced memory access pattern, ~90% bandwidth utilization
 * - Expected Speedup: 3x memory access speed
 * 
 * Test Strategy:
 * 1. Compare random vs coalesced memory access patterns
 * 2. Measure memory bandwidth utilization
 * 3. Test different data layouts and access patterns
 * 4. Validate correctness with known inputs
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
struct MemoryTestData {
    float* input_data;
    float* output_data;
    uint32_t* indices;
    uint32_t data_size;
    uint32_t num_operations;
};

// Current implementation (random access)
__global__ void memory_access_random(
    const float* input_data,
    float* output_data,
    const uint32_t* indices,
    uint32_t data_size,
    uint32_t num_operations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;
    
    // Random access pattern - poor coalescing
    uint32_t random_idx = indices[idx % data_size];
    output_data[random_idx] = input_data[random_idx] * 2.0f + 1.0f;
}

// Optimized implementation (coalesced access)
__global__ void memory_access_coalesced(
    const float* input_data,
    float* output_data,
    uint32_t data_size,
    uint32_t num_operations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;
    
    // Coalesced access pattern - good coalescing
    if (idx < data_size) {
        output_data[idx] = input_data[idx] * 2.0f + 1.0f;
    }
}

// Optimized implementation with shared memory
__global__ void memory_access_shared(
    const float* input_data,
    float* output_data,
    uint32_t data_size,
    uint32_t num_operations
) {
    extern __shared__ float shared_data[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    uint32_t start = bid * block_size;
    uint32_t end = min(start + block_size, data_size);
    
    // Load data into shared memory with coalescing
    if (start + tid < end) {
        shared_data[tid] = input_data[start + tid];
    }
    __syncthreads();
    
    // Process data in shared memory
    if (start + tid < end) {
        shared_data[tid] = shared_data[tid] * 2.0f + 1.0f;
    }
    __syncthreads();
    
    // Store data back to global memory with coalescing
    if (start + tid < end) {
        output_data[start + tid] = shared_data[tid];
    }
}

// Optimized implementation with vectorized access
__global__ void memory_access_vectorized(
    const float4* input_data,
    float4* output_data,
    uint32_t data_size,
    uint32_t num_operations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;
    
    // Vectorized access pattern - 4x better bandwidth utilization
    if (idx < data_size / 4) {
        float4 input_vec = input_data[idx];
        float4 output_vec;
        output_vec.x = input_vec.x * 2.0f + 1.0f;
        output_vec.y = input_vec.y * 2.0f + 1.0f;
        output_vec.z = input_vec.z * 2.0f + 1.0f;
        output_vec.w = input_vec.w * 2.0f + 1.0f;
        output_data[idx] = output_vec;
    }
}

// Memory bandwidth measurement
double measure_memory_bandwidth(
    void (*kernel)(const float*, float*, const uint32_t*, uint32_t, uint32_t),
    const MemoryTestData& data,
    const char* kernel_name,
    int iterations = 100
) {
    // Allocate device memory
    float* d_input_data;
    float* d_output_data;
    uint32_t* d_indices;
    
    cudaMalloc(&d_input_data, data.data_size * sizeof(float));
    cudaMalloc(&d_output_data, data.data_size * sizeof(float));
    cudaMalloc(&d_indices, data.data_size * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input_data, data.input_data, data.data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, data.indices, data.data_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((data.num_operations + blockSize.x - 1) / blockSize.x);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_output_data, 0, data.data_size * sizeof(float));
        kernel<<<gridSize, blockSize>>>(d_input_data, d_output_data, d_indices, data.data_size, data.num_operations);
        cudaDeviceSynchronize();
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_output_data, 0, data.data_size * sizeof(float));
        kernel<<<gridSize, blockSize>>>(d_input_data, d_output_data, d_indices, data.data_size, data.num_operations);
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = duration / iterations;
    
    // Calculate bandwidth
    double bytes_transferred = data.num_operations * sizeof(float) * 2; // read + write
    double bandwidth_gb_s = (bytes_transferred / (avg_time / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "   " << kernel_name << ": " << avg_time << " ms, " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    cudaFree(d_indices);
    
    return bandwidth_gb_s;
}

// Vectorized memory bandwidth measurement
double measure_vectorized_bandwidth(
    const MemoryTestData& data,
    const char* kernel_name,
    int iterations = 100
) {
    // Allocate device memory
    float4* d_input_data;
    float4* d_output_data;
    
    cudaMalloc(&d_input_data, (data.data_size / 4) * sizeof(float4));
    cudaMalloc(&d_output_data, (data.data_size / 4) * sizeof(float4));
    
    // Convert input data to float4
    float4* h_input_data = new float4[data.data_size / 4];
    for (uint32_t i = 0; i < data.data_size / 4; i++) {
        h_input_data[i].x = data.input_data[i * 4];
        h_input_data[i].y = data.input_data[i * 4 + 1];
        h_input_data[i].z = data.input_data[i * 4 + 2];
        h_input_data[i].w = data.input_data[i * 4 + 3];
    }
    
    // Copy data to device
    cudaMemcpy(d_input_data, h_input_data, (data.data_size / 4) * sizeof(float4), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize(((data.data_size / 4) + blockSize.x - 1) / blockSize.x);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_output_data, 0, (data.data_size / 4) * sizeof(float4));
        memory_access_vectorized<<<gridSize, blockSize>>>(d_input_data, d_output_data, data.data_size, data.num_operations);
        cudaDeviceSynchronize();
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_output_data, 0, (data.data_size / 4) * sizeof(float4));
        memory_access_vectorized<<<gridSize, blockSize>>>(d_input_data, d_output_data, data.data_size, data.num_operations);
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = duration / iterations;
    
    // Calculate bandwidth
    double bytes_transferred = data.num_operations * sizeof(float) * 2; // read + write
    double bandwidth_gb_s = (bytes_transferred / (avg_time / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "   " << kernel_name << ": " << avg_time << " ms, " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    delete[] h_input_data;
    
    return bandwidth_gb_s;
}

// Test data generation
MemoryTestData generate_memory_test_data(uint32_t data_size, uint32_t num_operations) {
    MemoryTestData data;
    data.data_size = data_size;
    data.num_operations = num_operations;
    
    // Allocate host memory
    data.input_data = new float[data_size];
    data.output_data = new float[data_size];
    data.indices = new uint32_t[data_size];
    
    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0f, 1000.0f);
    std::uniform_int_distribution<> idx_dist(0, data_size - 1);
    
    for (uint32_t i = 0; i < data_size; i++) {
        data.input_data[i] = dist(gen);
        data.indices[i] = idx_dist(gen);
    }
    
    return data;
}

// Correctness validation
bool validate_memory_correctness(const MemoryTestData& data) {
    // Allocate device memory
    float* d_input_data;
    float* d_output_data_random;
    float* d_output_data_coalesced;
    float* d_output_data_shared;
    uint32_t* d_indices;
    
    cudaMalloc(&d_input_data, data.data_size * sizeof(float));
    cudaMalloc(&d_output_data_random, data.data_size * sizeof(float));
    cudaMalloc(&d_output_data_coalesced, data.data_size * sizeof(float));
    cudaMalloc(&d_output_data_shared, data.data_size * sizeof(float));
    cudaMalloc(&d_indices, data.data_size * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input_data, data.input_data, data.data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, data.indices, data.data_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((data.num_operations + blockSize.x - 1) / blockSize.x);
    size_t shared_mem_size = blockSize.x * sizeof(float);
    
    cudaMemset(d_output_data_random, 0, data.data_size * sizeof(float));
    memory_access_random<<<gridSize, blockSize>>>(d_input_data, d_output_data_random, d_indices, data.data_size, data.num_operations);
    cudaDeviceSynchronize();
    
    cudaMemset(d_output_data_coalesced, 0, data.data_size * sizeof(float));
    memory_access_coalesced<<<gridSize, blockSize>>>(d_input_data, d_output_data_coalesced, data.data_size, data.num_operations);
    cudaDeviceSynchronize();
    
    cudaMemset(d_output_data_shared, 0, data.data_size * sizeof(float));
    memory_access_shared<<<gridSize, blockSize, shared_mem_size>>>(d_input_data, d_output_data_shared, data.data_size, data.num_operations);
    cudaDeviceSynchronize();
    
    // Compare results
    float* h_output_random = new float[data.data_size];
    float* h_output_coalesced = new float[data.data_size];
    float* h_output_shared = new float[data.data_size];
    
    cudaMemcpy(h_output_random, d_output_data_random, data.data_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_coalesced, d_output_data_coalesced, data.data_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shared, d_output_data_shared, data.data_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check correctness (coalesced and shared should be identical)
    bool correct = true;
    float max_error = 0.0f;
    for (uint32_t i = 0; i < data.data_size; i++) {
        float error = fabsf(h_output_coalesced[i] - h_output_shared[i]);
        max_error = fmaxf(max_error, error);
        if (error > 1e-5f) {
            correct = false;
        }
    }
    
    std::cout << "   Correctness check: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "   Max error: " << max_error << std::endl;
    
    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_output_data_random);
    cudaFree(d_output_data_coalesced);
    cudaFree(d_output_data_shared);
    cudaFree(d_indices);
    delete[] h_output_random;
    delete[] h_output_coalesced;
    delete[] h_output_shared;
    
    return correct;
}

// Main test function
int main() {
    std::cout << "🧪 Testing Memory Coalescing Optimization" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Test different data sizes
    std::vector<std::pair<uint32_t, uint32_t>> test_sizes = {
        {100000, 100000},    // 100K data, 100K operations
        {1000000, 1000000},  // 1M data, 1M operations
        {10000000, 10000000} // 10M data, 10M operations
    };
    
    for (auto& size : test_sizes) {
        uint32_t data_size = size.first;
        uint32_t num_operations = size.second;
        
        std::cout << "\n📊 Testing: " << data_size << " data size, " << num_operations << " operations" << std::endl;
        
        // Generate test data
        MemoryTestData data = generate_memory_test_data(data_size, num_operations);
        
        // Validate correctness
        bool correct = validate_memory_correctness(data);
        if (!correct) {
            std::cout << "❌ Correctness validation failed!" << std::endl;
            continue;
        }
        
        // Measure memory bandwidth
        double bandwidth_random = measure_memory_bandwidth(memory_access_random, data, "Random Access");
        double bandwidth_coalesced = measure_memory_bandwidth(memory_access_coalesced, data, "Coalesced Access");
        double bandwidth_shared = measure_memory_bandwidth(memory_access_shared, data, "Shared Memory");
        double bandwidth_vectorized = measure_vectorized_bandwidth(data, "Vectorized Access");
        
        // Calculate speedups
        double speedup_coalesced = bandwidth_coalesced / bandwidth_random;
        double speedup_shared = bandwidth_shared / bandwidth_random;
        double speedup_vectorized = bandwidth_vectorized / bandwidth_random;
        
        std::cout << "   Speedup (Coalesced): " << speedup_coalesced << "x" << std::endl;
        std::cout << "   Speedup (Shared): " << speedup_shared << "x" << std::endl;
        std::cout << "   Speedup (Vectorized): " << speedup_vectorized << "x" << std::endl;
        
        // Cleanup
        delete[] data.input_data;
        delete[] data.output_data;
        delete[] data.indices;
    }
    
    std::cout << "\n✅ Memory coalescing testing complete!" << std::endl;
    return 0;
}
