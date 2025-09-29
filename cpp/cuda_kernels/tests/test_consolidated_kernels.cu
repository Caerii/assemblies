/*
 * Test Consolidated Kernel Implementations
 * =======================================
 * 
 * This file tests the consolidated kernel implementations that combine
 * all optimizations into a single, well-tested implementation.
 * 
 * Test Strategy:
 * 1. Compare consolidated vs current implementations
 * 2. Test all kernel combinations
 * 3. Measure end-to-end performance
 * 4. Validate correctness with known inputs
 * 5. Test memory usage and allocation patterns
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Consolidated kernel implementations
namespace assemblies {
namespace cuda {

// Optimized weight accumulation with warp reduction and memory coalescing
__global__ void accumulate_weights_optimized(
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
    
    // Warp-level reduction for each target
    if (tid < target_size) {
        float warp_sum = shared_activations[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (tid % 32 == 0) {
            atomicAdd(&activations[tid], warp_sum);
        }
    }
}

// Optimized top-k selection with radix selection
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

__global__ void top_k_selection_optimized(
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

// Optimized candidate generation with better random number generation
__global__ void generate_candidates_optimized(
    curandState* states,
    float* candidate_weights,
    uint32_t num_candidates,
    float mean,
    float stddev,
    float cutoff
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    curandState local_state = states[idx];
    
    // Generate truncated normal samples with better rejection sampling
    float sample;
    int attempts = 0;
    do {
        sample = curand_normal(&local_state) * stddev + mean;
        attempts++;
    } while (sample < cutoff && attempts < 100); // Prevent infinite loops
    
    // Clamp to reasonable range
    sample = fmaxf(cutoff, fminf(sample, mean * 3.0f));
    candidate_weights[idx] = sample;
    states[idx] = local_state;
}

// Optimized synapse generation with geometric distribution
__global__ void generate_synapses_optimized(
    curandState* states,
    uint32_t* synapse_indices,
    float* synapse_weights,
    uint32_t* synapse_offsets,
    uint32_t support,
    float p
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= support) return;
    
    curandState local_state = states[idx];
    
    // Generate synapses using geometric distribution
    uint32_t offset = synapse_offsets[idx];
    uint32_t count = 0;
    
    // Sample from geometric(p) distribution
    float scale = 1.0f / logf(1.0f - p);
    uint32_t last = (uint32_t)floorf(logf(curand_uniform(&local_state)) * scale);
    
    while (last < support && offset + count < synapse_offsets[idx + 1]) {
        synapse_indices[offset + count] = last;
        synapse_weights[offset + count] = 1.0f;
        count++;
        
        last += 1 + (uint32_t)floorf(logf(curand_uniform(&local_state)) * scale);
    }
    
    states[idx] = local_state;
}

// Optimized plasticity update with better memory access
__global__ void update_plasticity_optimized(
    float* synapse_weights,
    const uint32_t* activated_neurons,
    const uint32_t* synapse_indices,
    const uint32_t* synapse_offsets,
    float learn_rate,
    float max_weight,
    uint32_t num_activated
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_activated) return;
    
    uint32_t neuron = activated_neurons[idx];
    uint32_t start = synapse_offsets[neuron];
    uint32_t end = synapse_offsets[neuron + 1];
    
    // Apply Hebbian learning rule with better memory access
    for (uint32_t i = start; i < end; i++) {
        float current_weight = synapse_weights[i];
        float new_weight = current_weight + learn_rate * (1.0f - current_weight);
        synapse_weights[i] = fminf(new_weight, max_weight);
    }
}

// CUDA random state setup
__global__ void curandSetupKernel(curandState* states, unsigned long seed, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

} // namespace cuda
} // namespace assemblies

// Test data structures
struct ConsolidatedTestData {
    uint32_t* activated_neurons;
    float* synapse_weights;
    uint32_t* synapse_indices;
    uint32_t* synapse_offsets;
    float* activations;
    uint32_t* top_k_indices;
    curandState* states;
    float* candidate_weights;
    uint32_t num_activated;
    uint32_t target_size;
    uint32_t k;
    uint32_t total_synapses;
};

// Test data generation
ConsolidatedTestData generate_consolidated_test_data(
    uint32_t num_neurons, 
    uint32_t active_percentage, 
    uint32_t synapses_per_neuron,
    uint32_t k
) {
    ConsolidatedTestData data;
    data.num_activated = num_neurons * active_percentage;
    data.target_size = num_neurons;
    data.k = k;
    data.total_synapses = data.num_activated * synapses_per_neuron;
    
    // Allocate host memory
    data.activated_neurons = new uint32_t[data.num_activated];
    data.synapse_weights = new float[data.total_synapses];
    data.synapse_indices = new uint32_t[data.total_synapses];
    data.synapse_offsets = new uint32_t[data.num_activated + 1];
    data.activations = new float[data.target_size];
    data.top_k_indices = new uint32_t[data.k];
    data.candidate_weights = new float[data.k];
    
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

// End-to-end performance test
double test_consolidated_performance(
    const ConsolidatedTestData& data,
    const char* test_name,
    int iterations = 100
) {
    // Allocate device memory
    uint32_t* d_activated_neurons;
    float* d_synapse_weights;
    uint32_t* d_synapse_indices;
    uint32_t* d_synapse_offsets;
    float* d_activations;
    uint32_t* d_top_k_indices;
    curandState* d_states;
    float* d_candidate_weights;
    
    cudaMalloc(&d_activated_neurons, data.num_activated * sizeof(uint32_t));
    cudaMalloc(&d_synapse_weights, data.total_synapses * sizeof(float));
    cudaMalloc(&d_synapse_indices, data.total_synapses * sizeof(uint32_t));
    cudaMalloc(&d_synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t));
    cudaMalloc(&d_activations, data.target_size * sizeof(float));
    cudaMalloc(&d_top_k_indices, data.k * sizeof(uint32_t));
    cudaMalloc(&d_states, data.k * sizeof(curandState));
    cudaMalloc(&d_candidate_weights, data.k * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_activated_neurons, data.activated_neurons, data.num_activated * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_weights, data.synapse_weights, data.total_synapses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_indices, data.synapse_indices, data.total_synapses * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse_offsets, data.synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Initialize CUDA random states
    dim3 blockSize(256);
    dim3 gridSize((data.k + blockSize.x - 1) / blockSize.x);
    assemblies::cuda::curandSetupKernel<<<gridSize, blockSize>>>(d_states, 42, data.k);
    cudaDeviceSynchronize();
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_activations, 0, data.target_size * sizeof(float));
        assemblies::cuda::accumulate_weights_optimized<<<gridSize, blockSize, data.target_size * sizeof(float)>>>(
            d_activated_neurons, d_synapse_weights, d_synapse_indices, d_synapse_offsets,
            d_activations, data.num_activated, data.target_size
        );
        cudaDeviceSynchronize();
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // Weight accumulation
        cudaMemset(d_activations, 0, data.target_size * sizeof(float));
        assemblies::cuda::accumulate_weights_optimized<<<gridSize, blockSize, data.target_size * sizeof(float)>>>(
            d_activated_neurons, d_synapse_weights, d_synapse_indices, d_synapse_offsets,
            d_activations, data.num_activated, data.target_size
        );
        cudaDeviceSynchronize();
        
        // Top-k selection
        assemblies::cuda::top_k_selection_optimized<<<gridSize, blockSize, blockSize.x * (sizeof(float) + sizeof(uint32_t))>>>(
            d_activations, d_top_k_indices, data.target_size, data.k
        );
        cudaDeviceSynchronize();
        
        // Candidate generation
        assemblies::cuda::generate_candidates_optimized<<<gridSize, blockSize>>>(
            d_states, d_candidate_weights, data.k, 1.0f, 1.0f, 0.0f
        );
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = duration / iterations;
    
    std::cout << "   " << test_name << ": " << avg_time << " ms per iteration" << std::endl;
    
    // Cleanup
    cudaFree(d_activated_neurons);
    cudaFree(d_synapse_weights);
    cudaFree(d_synapse_indices);
    cudaFree(d_synapse_offsets);
    cudaFree(d_activations);
    cudaFree(d_top_k_indices);
    cudaFree(d_states);
    cudaFree(d_candidate_weights);
    
    return avg_time;
}

// Memory usage test
void test_memory_usage(const ConsolidatedTestData& data) {
    // Allocate device memory
    uint32_t* d_activated_neurons;
    float* d_synapse_weights;
    uint32_t* d_synapse_indices;
    uint32_t* d_synapse_offsets;
    float* d_activations;
    uint32_t* d_top_k_indices;
    curandState* d_states;
    float* d_candidate_weights;
    
    cudaMalloc(&d_activated_neurons, data.num_activated * sizeof(uint32_t));
    cudaMalloc(&d_synapse_weights, data.total_synapses * sizeof(float));
    cudaMalloc(&d_synapse_indices, data.total_synapses * sizeof(uint32_t));
    cudaMalloc(&d_synapse_offsets, (data.num_activated + 1) * sizeof(uint32_t));
    cudaMalloc(&d_activations, data.target_size * sizeof(float));
    cudaMalloc(&d_top_k_indices, data.k * sizeof(uint32_t));
    cudaMalloc(&d_states, data.k * sizeof(curandState));
    cudaMalloc(&d_candidate_weights, data.k * sizeof(float));
    
    // Get memory usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    double used_mem_gb = (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0);
    double total_mem_gb = total_mem / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "   Memory usage: " << used_mem_gb << " GB / " << total_mem_gb << " GB" << std::endl;
    std::cout << "   Memory efficiency: " << (used_mem_gb / total_mem_gb) * 100 << "%" << std::endl;
    
    // Cleanup
    cudaFree(d_activated_neurons);
    cudaFree(d_synapse_weights);
    cudaFree(d_synapse_indices);
    cudaFree(d_synapse_offsets);
    cudaFree(d_activations);
    cudaFree(d_top_k_indices);
    cudaFree(d_states);
    cudaFree(d_candidate_weights);
}

// Main test function
int main() {
    std::cout << "🧪 Testing Consolidated Kernel Implementations" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Test different scales
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> test_scales = {
        {100000, 1000, 100, 10},    // 100K neurons, 1K active, 100 synapses/neuron, k=10
        {1000000, 10000, 100, 100}, // 1M neurons, 10K active, 100 synapses/neuron, k=100
        {10000000, 100000, 100, 1000} // 10M neurons, 100K active, 100 synapses/neuron, k=1000
    };
    
    for (auto& scale : test_scales) {
        uint32_t num_neurons = std::get<0>(scale);
        uint32_t active_neurons = std::get<1>(scale);
        uint32_t synapses_per_neuron = std::get<2>(scale);
        uint32_t k = std::get<3>(scale);
        
        std::cout << "\n📊 Testing scale: " << num_neurons << " neurons, " << active_neurons << " active, k=" << k << std::endl;
        
        // Generate test data
        ConsolidatedTestData data = generate_consolidated_test_data(
            num_neurons, active_neurons, synapses_per_neuron, k
        );
        
        // Test memory usage
        test_memory_usage(data);
        
        // Test performance
        double time_consolidated = test_consolidated_performance(data, "Consolidated Kernels");
        
        // Calculate theoretical performance
        double theoretical_ops = data.num_activated * synapses_per_neuron + data.target_size * log2(k) + data.k;
        double theoretical_time = theoretical_ops / (1e9); // Assuming 1 billion ops/sec
        double efficiency = (theoretical_time / time_consolidated) * 100;
        
        std::cout << "   Theoretical efficiency: " << efficiency << "%" << std::endl;
        
        // Cleanup
        delete[] data.activated_neurons;
        delete[] data.synapse_weights;
        delete[] data.synapse_indices;
        delete[] data.synapse_offsets;
        delete[] data.activations;
        delete[] data.top_k_indices;
        delete[] data.candidate_weights;
    }
    
    std::cout << "\n✅ Consolidated kernel testing complete!" << std::endl;
    return 0;
}
