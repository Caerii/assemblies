#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <random>

extern "C" {

// Simple CUDA kernel for candidate generation
__global__ void generate_candidates_kernel(
    curandState* states,
    float* candidate_weights,
    uint32_t num_candidates
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    curandState local_state = states[idx];
    float sample = curand_uniform(&local_state);
    candidate_weights[idx] = -logf(1.0f - sample); // Exponential distribution
    states[idx] = local_state;
}

// Simple CUDA kernel for top-k selection
__global__ void top_k_selection_kernel(
    const float* activations,
    uint32_t* top_k_indices,
    uint32_t total_neurons,
    uint32_t k
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    // Simple top-k selection
    float max_val = -1e30f;
    uint32_t max_idx = 0;
    
    for (uint32_t i = 0; i < total_neurons; i++) {
        if (activations[i] > max_val) {
            max_val = activations[i];
            max_idx = i;
        }
    }
    
    top_k_indices[idx] = max_idx;
}

// CUDA random state setup kernel
__global__ void curandSetupKernel(curandState* states, unsigned long seed, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA brain wrapper class
class SimpleCudaBrainWrapper {
private:
    uint32_t n_neurons_;
    uint32_t k_active_;
    uint32_t n_areas_;
    uint32_t seed_;
    
    // CUDA device memory
    curandState* d_states_;
    float* d_candidates_;
    uint32_t* d_top_k_indices_;
    
    // Host memory
    std::vector<float> h_candidates_;
    std::vector<uint32_t> h_top_k_indices_;
    
public:
    SimpleCudaBrainWrapper(uint32_t n_neurons, uint32_t k_active, uint32_t n_areas, uint32_t seed = 42)
        : n_neurons_(n_neurons), k_active_(k_active), n_areas_(n_areas), seed_(seed) {
        
        // Allocate CUDA memory
        cudaMalloc(&d_states_, n_neurons * sizeof(curandState));
        cudaMalloc(&d_candidates_, n_neurons * sizeof(float));
        cudaMalloc(&d_top_k_indices_, k_active * sizeof(uint32_t));
        
        // Allocate host memory
        h_candidates_.resize(n_neurons);
        h_top_k_indices_.resize(k_active);
        
        // Initialize CUDA random states
        curandSetupKernel<<<(n_neurons + 255) / 256, 256>>>(d_states_, seed, n_neurons);
        cudaDeviceSynchronize();
        
        std::cout << "🚀 Simple CUDA Brain Wrapper initialized" << std::endl;
        std::cout << "   Neurons: " << n_neurons_ << std::endl;
        std::cout << "   Active: " << k_active_ << std::endl;
        std::cout << "   Areas: " << n_areas_ << std::endl;
    }
    
    ~SimpleCudaBrainWrapper() {
        cudaFree(d_states_);
        cudaFree(d_candidates_);
        cudaFree(d_top_k_indices_);
    }
    
    void simulate_step() {
        // Generate candidates using CUDA
        generate_candidates_kernel<<<(n_neurons_ + 255) / 256, 256>>>(
            d_states_, d_candidates_, n_neurons_
        );
        cudaDeviceSynchronize();
        
        // Copy candidates to host
        cudaMemcpy(h_candidates_.data(), d_candidates_, n_neurons_ * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Select top-k using CUDA
        top_k_selection_kernel<<<(n_neurons_ + 255) / 256, 256>>>(
            d_candidates_, d_top_k_indices_, n_neurons_, k_active_
        );
        cudaDeviceSynchronize();
        
        // Copy top-k indices to host
        cudaMemcpy(h_top_k_indices_.data(), d_top_k_indices_, k_active_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    
    float* get_candidates() { return h_candidates_.data(); }
    uint32_t* get_top_k_indices() { return h_top_k_indices_.data(); }
};

// C interface functions
extern "C" {
    SimpleCudaBrainWrapper* create_simple_cuda_brain(uint32_t n_neurons, uint32_t k_active, uint32_t n_areas, uint32_t seed) {
        return new SimpleCudaBrainWrapper(n_neurons, k_active, n_areas, seed);
    }
    
    void destroy_simple_cuda_brain(SimpleCudaBrainWrapper* brain) {
        delete brain;
    }
    
    void simulate_step(SimpleCudaBrainWrapper* brain) {
        brain->simulate_step();
    }
    
    float* get_candidates(SimpleCudaBrainWrapper* brain) {
        return brain->get_candidates();
    }
    
    uint32_t* get_top_k_indices(SimpleCudaBrainWrapper* brain) {
        return brain->get_top_k_indices();
    }
}

} // extern "C"
