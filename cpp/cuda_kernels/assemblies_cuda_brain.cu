#include "cuda_brain_fixed.h"
#include <iostream>
#include <algorithm>
#include <numeric>

// CUDA kernel implementations
__global__ void accumulate_weights_kernel(
    const uint32_t* activated_neurons,
    const CudaSynapse* synapses,
    const uint32_t* synapse_indices,
    const uint32_t* synapse_offsets,
    float* activations,
    uint32_t num_activated,
    uint32_t total_neurons
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_neurons) return;
    
    float activation = 0.0f;
    
    // Simplified accumulation (placeholder)
    for (uint32_t i = 0; i < num_activated; ++i) {
        if (activated_neurons[i] == idx) {
            activation += 1.0f;
        }
    }
    
    activations[idx] = activation;
}

__global__ void curandSetupKernel(curandState* state, uint32_t seed, uint32_t num_states) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void generate_candidates_kernel(
    curandState* state,
    float* candidates,
    uint32_t num_candidates,
    float mean,
    float stddev,
    float cutoff
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    float value = curand_normal(&state[idx]) * stddev + mean;
    candidates[idx] = (value > cutoff) ? value : 0.0f;
}

__global__ void top_k_selection_kernel(
    const float* activations,
    uint32_t* top_k_indices,
    uint32_t total_neurons,
    uint32_t k
) {
    // Simplified top-k selection (placeholder)
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    // Simple selection - just pick first k neurons
    top_k_indices[idx] = idx;
}

namespace nemo {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

CudaBrain::CudaBrain(float p, float beta, float max_weight, uint32_t seed)
    : rng_(seed), p_(p), beta_(beta), learn_rate_(1.0f + beta_), max_weight_(max_weight) {
    
    // Initialize device pointers to nullptr
    d_areas_ = nullptr;
    d_fibers_ = nullptr;
    
    std::cout << "🧠 CUDA Brain initialized for RTX 4090" << std::endl;
    std::cout << "   Parameters: p=" << p_ << ", beta=" << beta_ << ", max_weight=" << max_weight_ << std::endl;
}

CudaBrain::~CudaBrain() {
    FreeDeviceMemory();
}

void CudaBrain::AddArea(const std::string& name, uint32_t n, uint32_t k, bool recurrent, bool is_explicit) {
    uint32_t area_idx = h_areas_.size();
    area_name_to_index_[name] = area_idx;
    area_index_to_name_.push_back(name);

    CudaArea new_area;
    new_area.index = area_idx;
    new_area.n = n;
    new_area.k = k;
    new_area.support = is_explicit ? n : 0;
    new_area.is_fixed = false;
    new_area.d_activated = nullptr;
    new_area.activated_size = 0;

    h_areas_.push_back(new_area);
    
    std::cout << "✓ Added area: " << name << " (n=" << n << ", k=" << k << ")" << std::endl;
}

void CudaBrain::AddStimulus(const std::string& name, uint32_t k) {
    uint32_t stim_idx = h_areas_.size();
    area_name_to_index_[name] = stim_idx;
    area_index_to_name_.push_back(name);

    CudaArea new_stim_area;
    new_stim_area.index = stim_idx;
    new_stim_area.n = k;
    new_stim_area.k = k;
    new_stim_area.support = k;
    new_stim_area.is_fixed = true;
    new_stim_area.d_activated = nullptr;
    new_stim_area.activated_size = k;

    h_areas_.push_back(new_stim_area);
    
    std::cout << "✓ Added stimulus: " << name << " (k=" << k << ")" << std::endl;
}

void CudaBrain::AddFiber(const std::string& from_name, const std::string& to_name, bool bidirectional) {
    uint32_t from_idx = area_name_to_index_[from_name];
    uint32_t to_idx = area_name_to_index_[to_name];

    CudaFiber new_fiber;
    new_fiber.from_area = from_idx;
    new_fiber.to_area = to_idx;
    new_fiber.is_active = true;
    new_fiber.d_outgoing_synapses_data = nullptr;
    new_fiber.d_outgoing_synapses_offsets = nullptr;
    new_fiber.total_synapses = 0;

    h_fibers_.push_back(new_fiber);
    
    std::cout << "✓ Added fiber: " << from_name << " -> " << to_name << std::endl;

    if (bidirectional) {
        CudaFiber new_fiber_rev;
        new_fiber_rev.from_area = to_idx;
        new_fiber_rev.to_area = from_idx;
        new_fiber_rev.is_active = true;
        new_fiber_rev.d_outgoing_synapses_data = nullptr;
        new_fiber_rev.d_outgoing_synapses_offsets = nullptr;
        new_fiber_rev.total_synapses = 0;
        h_fibers_.push_back(new_fiber_rev);
        
        std::cout << "✓ Added bidirectional fiber: " << to_name << " -> " << from_name << std::endl;
    }
}

void CudaBrain::AllocateDeviceMemory() {
    std::cout << "🔧 Allocating GPU memory..." << std::endl;
    
    // Allocate d_areas_
    CUDA_CHECK(cudaMalloc(&d_areas_, h_areas_.size() * sizeof(CudaArea)));
    
    // Allocate d_fibers_
    CUDA_CHECK(cudaMalloc(&d_fibers_, h_fibers_.size() * sizeof(CudaFiber)));

    // Allocate activated neuron arrays for each area
    for (size_t i = 0; i < h_areas_.size(); ++i) {
        if (h_areas_[i].n > 0) {
            CUDA_CHECK(cudaMalloc(&h_areas_[i].d_activated, h_areas_[i].n * sizeof(uint32_t)));
            std::cout << "  ✓ Allocated " << h_areas_[i].n << " neurons for area " << area_index_to_name_[i] << std::endl;
        }
    }

    // Allocate synapse data for each fiber (simplified)
    for (size_t i = 0; i < h_fibers_.size(); ++i) {
        uint32_t from_n = h_areas_[h_fibers_[i].from_area].n;
        uint32_t to_k = h_areas_[h_fibers_[i].to_area].k;
        
        h_fibers_[i].total_synapses = from_n * to_k;
        if (h_fibers_[i].total_synapses > 0) {
            CUDA_CHECK(cudaMalloc(&h_fibers_[i].d_outgoing_synapses_data, h_fibers_[i].total_synapses * sizeof(CudaSynapse)));
            CUDA_CHECK(cudaMalloc(&h_fibers_[i].d_outgoing_synapses_offsets, (from_n + 1) * sizeof(uint32_t)));
        }
    }
    
    std::cout << "✅ GPU memory allocation complete!" << std::endl;
}

void CudaBrain::CopyHostToDevice() {
    std::cout << "📤 Copying data to GPU..." << std::endl;
    
    // Copy h_areas_ to d_areas_
    CUDA_CHECK(cudaMemcpy(d_areas_, h_areas_.data(), h_areas_.size() * sizeof(CudaArea), cudaMemcpyHostToDevice));
    
    // Copy h_fibers_ to d_fibers_
    CUDA_CHECK(cudaMemcpy(d_fibers_, h_fibers_.data(), h_fibers_.size() * sizeof(CudaFiber), cudaMemcpyHostToDevice));
    
    std::cout << "✅ Data copied to GPU!" << std::endl;
}

void CudaBrain::FreeDeviceMemory() {
    if (d_areas_) CUDA_CHECK(cudaFree(d_areas_));
    if (d_fibers_) CUDA_CHECK(cudaFree(d_fibers_));

    for (size_t i = 0; i < h_areas_.size(); ++i) {
        if (h_areas_[i].d_activated) CUDA_CHECK(cudaFree(h_areas_[i].d_activated));
    }
    
    for (size_t i = 0; i < h_fibers_.size(); ++i) {
        if (h_fibers_[i].d_outgoing_synapses_data) CUDA_CHECK(cudaFree(h_fibers_[i].d_outgoing_synapses_data));
        if (h_fibers_[i].d_outgoing_synapses_offsets) CUDA_CHECK(cudaFree(h_fibers_[i].d_outgoing_synapses_offsets));
    }
}

void CudaBrain::SimulateOneStep(bool update_plasticity) {
    // This is the core simulation loop on the GPU
    std::cout << "🧠 CUDA simulation step " << step_ << std::endl;
    
    // For each area, compute activations
    for (size_t i = 0; i < h_areas_.size(); ++i) {
        CudaArea& to_area = h_areas_[i];
        if (to_area.is_fixed) continue;

        // Allocate device memory for activations
        float* d_activations;
        CUDA_CHECK(cudaMalloc(&d_activations, to_area.n * sizeof(float)));
        
        // Initialize activations to 0
        CUDA_CHECK(cudaMemset(d_activations, 0, to_area.n * sizeof(float)));

        // Launch accumulation kernel
        dim3 block_size(256);
        dim3 grid_size((to_area.n + block_size.x - 1) / block_size.x);
        
        accumulate_weights_kernel<<<grid_size, block_size>>>(
            to_area.d_activated,  // activated neurons
            nullptr,              // synapse weights (placeholder)
            nullptr,              // synapse indices (placeholder)
            nullptr,              // synapse offsets (placeholder)
            d_activations,        // output activations
            to_area.activated_size,
            to_area.n
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());

        // Generate new candidates
        if (to_area.support < to_area.n) {
            curandState* d_states;
            CUDA_CHECK(cudaMalloc(&d_states, to_area.k * sizeof(curandState)));
            
            // Initialize random states
            dim3 setup_block(256);
            dim3 setup_grid((to_area.k + setup_block.x - 1) / block_size.x);
            curandSetupKernel<<<setup_grid, setup_block>>>(d_states, rng_(), to_area.k);
            
            // Generate candidates
            float* d_candidates;
            CUDA_CHECK(cudaMalloc(&d_candidates, to_area.k * sizeof(float)));
            
            generate_candidates_kernel<<<setup_grid, setup_block>>>(
                d_states,
                d_candidates,
                to_area.k,
                0.5f,  // mean
                0.1f,  // stddev
                0.0f   // cutoff
            );
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Clean up
            CUDA_CHECK(cudaFree(d_states));
            CUDA_CHECK(cudaFree(d_candidates));
        }

        // Select top K (simplified)
        uint32_t* d_top_k;
        CUDA_CHECK(cudaMalloc(&d_top_k, to_area.k * sizeof(uint32_t)));
        
        top_k_selection_kernel<<<1, to_area.k>>>(
            d_activations,
            d_top_k,
            to_area.n,
            to_area.k
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update activated neurons
        CUDA_CHECK(cudaMemcpy(to_area.d_activated, d_top_k, to_area.k * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        to_area.activated_size = to_area.k;
        to_area.support = std::max(to_area.support, to_area.k);

        // Clean up
        CUDA_CHECK(cudaFree(d_activations));
        CUDA_CHECK(cudaFree(d_top_k));
    }
    
    step_++;
    std::cout << "✅ CUDA simulation step " << (step_ - 1) << " complete!" << std::endl;
}

void CudaBrain::Project(const std::map<std::string, std::vector<std::string>>& graph, uint32_t num_steps, bool update_plasticity) {
    std::cout << "🚀 Starting CUDA projection for " << num_steps << " steps" << std::endl;
    
    // Ensure device memory is allocated
    if (d_areas_ == nullptr) {
        AllocateDeviceMemory();
        CopyHostToDevice();
    }

    for (uint32_t i = 0; i < num_steps; ++i) {
        SimulateOneStep(update_plasticity);
    }
    
    std::cout << "✅ CUDA projection complete!" << std::endl;
}

std::vector<uint32_t> CudaBrain::GetActivatedNeurons(const std::string& area_name) {
    if (area_name_to_index_.find(area_name) == area_name_to_index_.end()) {
        return {};
    }
    
    uint32_t area_idx = area_name_to_index_[area_name];
    CudaArea& area = h_areas_[area_idx];

    std::vector<uint32_t> activated_neurons(area.activated_size);
    if (area.d_activated && area.activated_size > 0) {
        CUDA_CHECK(cudaMemcpy(activated_neurons.data(), area.d_activated, 
                              area.activated_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    return activated_neurons;
}

} // namespace cuda
} // namespace nemo
