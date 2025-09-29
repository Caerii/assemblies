#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace assemblies {
namespace cuda {

// Weight accumulation kernel - THE HOTTEST KERNEL
__global__ void accumulate_weights_kernel(
    const uint32_t* activated_neurons,     // Input: active neuron IDs
    const float* synapse_weights,          // Input: synapse weights
    const uint32_t* synapse_indices,       // Input: synapse target indices
    const uint32_t* synapse_offsets,       // Input: synapse offsets (CSR format)
    float* activations,                    // Output: accumulated weights
    uint32_t num_activated,                // Input: number of active neurons
    uint32_t target_size                   // Input: target area size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_activated) return;
    
    uint32_t neuron = activated_neurons[idx];
    uint32_t start = synapse_offsets[neuron];
    uint32_t end = synapse_offsets[neuron + 1];
    
    // Each thread processes synapses for one active neuron
    // This is the most parallelizable part of the algorithm
    for (uint32_t i = start; i < end; i++) {
        uint32_t target = synapse_indices[i];
        float weight = synapse_weights[i];
        
        // Atomic add for thread safety
        atomicAdd(&activations[target], weight);
    }
}

// Parallel Top-K selection using radix selection
__global__ void top_k_selection_kernel(
    const float* activations,              // Input: activation scores
    uint32_t* top_k_indices,              // Output: top-k neuron indices
    uint32_t total_neurons,               // Input: total neurons
    uint32_t k                            // Input: k value
) {
    // Shared memory for local sorting
    extern __shared__ float shared_scores[];
    extern __shared__ uint32_t shared_indices[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    // Each block processes a chunk of neurons
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
    
    // Parallel bitonic sort for top-k selection
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

// Parallel candidate generation using truncated normal distribution
__global__ void generate_candidates_kernel(
    curandState* states,                   // Input: CUDA RNG states
    float* candidate_weights,             // Output: candidate weights
    uint32_t num_candidates,              // Input: number of candidates
    float mean, float stddev, float cutoff // Input: distribution parameters
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    curandState local_state = states[idx];
    
    // Generate truncated normal samples in parallel
    float sample;
    do {
        sample = curand_normal(&local_state) * stddev + mean;
    } while (sample < cutoff);
    
    candidate_weights[idx] = fminf(mean * 2.0f, roundf(sample));
    states[idx] = local_state;
}

// Parallel synapse generation using geometric distribution
__global__ void generate_synapses_kernel(
    curandState* states,                   // Input: CUDA RNG states
    uint32_t* synapse_indices,            // Output: synapse target indices
    float* synapse_weights,               // Output: synapse weights
    uint32_t* synapse_offsets,            // Output: synapse offsets (CSR)
    uint32_t support,                     // Input: support size
    float p                               // Input: connection probability
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

// Parallel plasticity update using Hebbian learning
__global__ void update_plasticity_kernel(
    float* synapse_weights,                // Input/Output: synapse weights
    const uint32_t* activated_neurons,    // Input: activated neuron indices
    const uint32_t* synapse_indices,      // Input: synapse target indices
    const uint32_t* synapse_offsets,      // Input: synapse offsets
    float learn_rate,                     // Input: learning rate
    float max_weight,                     // Input: maximum weight
    uint32_t num_activated                // Input: number of activated neurons
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_activated) return;
    
    uint32_t neuron = activated_neurons[idx];
    uint32_t start = synapse_offsets[neuron];
    uint32_t end = synapse_offsets[neuron + 1];
    
    // Apply Hebbian learning rule: w += learn_rate * (1 - w)
    for (uint32_t i = start; i < end; i++) {
        float current_weight = synapse_weights[i];
        float new_weight = current_weight + learn_rate * (1.0f - current_weight);
        synapse_weights[i] = fminf(new_weight, max_weight);
    }
}

// Utility kernel for memory initialization
__global__ void initialize_memory_kernel(
    float* data,
    uint32_t size,
    float value
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// Utility kernel for data copying
__global__ void copy_data_kernel(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// CUDA random state setup kernel
__global__ void curandSetupKernel(curandState* states, unsigned long seed, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

} // namespace cuda
} // namespace assemblies
