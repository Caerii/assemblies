#ifndef CUDA_BRAIN_H_
#define CUDA_BRAIN_H_

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <random>
#include <cstdint>

namespace nemo {
namespace cuda {

// GPU-accelerated data structures
struct CudaSynapse {
    uint32_t neuron;
    float weight;
};

struct CudaArea {
    uint32_t index;
    uint32_t n;
    uint32_t k;
    uint32_t support;
    bool is_fixed;
    
    // GPU memory pointers
    uint32_t* d_activated;           // Device memory for activated neurons
    float* d_activations;            // Device memory for activation scores
    uint32_t* d_top_k_indices;       // Device memory for top-k selection
    
    // Sparse matrix data for synapses
    float* d_synapse_weights;        // Device memory for synapse weights
    uint32_t* d_synapse_indices;     // Device memory for synapse target indices
    uint32_t* d_synapse_offsets;     // Device memory for synapse offsets (CSR format)
    uint32_t* d_synapse_counts;      // Device memory for synapse counts per neuron
};

struct CudaFiber {
    uint32_t from_area;
    uint32_t to_area;
    bool is_active;
    
    // Sparse matrix representation on GPU
    float* d_weights;                // Device memory for weights
    uint32_t* d_col_indices;         // Device memory for column indices
    uint32_t* d_row_offsets;         // Device memory for row offsets
    uint32_t* d_col_offsets;         // Device memory for column offsets
    int nnz;                         // Number of non-zeros
};

class CudaBrain {
public:
    CudaBrain(float p, float beta, float max_weight, uint32_t seed);
    ~CudaBrain();
    
    // Core simulation methods
    void SimulateOneStep(bool update_plasticity = true);
    void Project(const std::map<std::string, std::vector<std::string>>& graph, 
                uint32_t num_steps, bool update_plasticity = true);
    
    // Area management
    CudaArea& AddArea(const std::string& name, uint32_t n, uint32_t k, 
                      bool recurrent = true, bool is_explicit = false);
    void AddStimulus(const std::string& name, uint32_t k);
    void AddFiber(const std::string& from, const std::string& to, 
                  bool bidirectional = false);
    
    // Data access
    CudaArea& GetArea(const std::string& name);
    const CudaArea& GetArea(const std::string& name) const;
    
    // GPU memory management
    void SyncToHost();
    void SyncToDevice();
    
    // Performance monitoring
    void LogPerformanceStats();
    void SetLogLevel(int level) { log_level_ = level; }
    
private:
    // GPU memory management
    void AllocateAreaMemory(CudaArea& area);
    void FreeAreaMemory(CudaArea& area);
    void AllocateFiberMemory(CudaFiber& fiber);
    void FreeFiberMemory(CudaFiber& fiber);
    
    // CUDA kernels
    void LaunchWeightAccumulationKernel(const CudaArea& from_area, 
                                       const CudaFiber& fiber, 
                                       CudaArea& to_area);
    void LaunchTopKSelectionKernel(CudaArea& area);
    void LaunchCandidateGenerationKernel(CudaArea& area, uint32_t total_k);
    void LaunchSynapseGenerationKernel(CudaArea& area, float p);
    
    // CUDA utilities
    void InitializeCuda();
    void CleanupCuda();
    void CheckCudaError(cudaError_t error, const char* file, int line);
    
    // Member variables
    std::mt19937 rng_;
    int log_level_;
    
    const float p_;
    const float beta_;
    const float learn_rate_;
    const float max_weight_;
    
    std::vector<CudaArea> areas_;
    std::vector<CudaFiber> fibers_;
    std::map<std::string, uint32_t> area_by_name_;
    std::vector<std::string> area_name_;
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;
    curandGenerator_t curand_gen_;
    
    // GPU memory pools
    void* d_memory_pool_;
    size_t memory_pool_size_;
    size_t memory_pool_offset_;
    
    uint32_t step_;
};

// CUDA kernel declarations
extern "C" {
    // Weight accumulation kernel
    __global__ void accumulate_weights_kernel(
        const uint32_t* activated_neurons,
        const float* synapse_weights,
        const uint32_t* synapse_indices,
        const uint32_t* synapse_offsets,
        float* activations,
        uint32_t num_activated,
        uint32_t target_size
    );
    
    // Top-K selection kernel
    __global__ void top_k_selection_kernel(
        const float* activations,
        uint32_t* top_k_indices,
        uint32_t total_neurons,
        uint32_t k
    );
    
    // Candidate generation kernel
    __global__ void generate_candidates_kernel(
        curandState* states,
        float* candidate_weights,
        uint32_t num_candidates,
        float mean,
        float stddev,
        float cutoff
    );
    
    // Synapse generation kernel
    __global__ void generate_synapses_kernel(
        curandState* states,
        uint32_t* synapse_indices,
        float* synapse_weights,
        uint32_t* synapse_offsets,
        uint32_t support,
        float p
    );
    
    // Plasticity update kernel
    __global__ void update_plasticity_kernel(
        float* synapse_weights,
        const uint32_t* activated_neurons,
        const uint32_t* synapse_indices,
        const uint32_t* synapse_offsets,
        float learn_rate,
        float max_weight,
        uint32_t num_activated
    );
}

} // namespace cuda
} // namespace nemo

#endif // CUDA_BRAIN_H_
