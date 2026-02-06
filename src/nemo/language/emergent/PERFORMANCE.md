# NEMO Emergent Language System - Performance Analysis

## Executive Summary

The NEMO emergent language system achieves **~30ms per sentence** (~100 words/second), which is **40x faster than human speech**. This is sufficient for real-time interaction.

Further optimization requires a **C++ implementation** to eliminate Python overhead, which accounts for ~60% of execution time.

## Existing C++ CUDA Implementation

The repository contains a mature C++ CUDA implementation in `cpp/`:

```
cpp/
├── core_cpp/           # C++ brain implementation with pybind11
│   ├── brain.h         # Brain class header
│   ├── brain.cc        # Implementation (sparse CSR synapses)
│   └── pybind11_wrapper.cpp
├── cuda_kernels/       # CUDA kernels (multiple variants)
│   ├── assembly_projection_kernel.cu  # Implicit hash-based connectivity
│   ├── dense_assembly_kernels.cu      # Dense n×n weight matrices
│   ├── assemblies_cuda_brain_optimized.cu  # Optimized sparse CSR
│   └── ...
├── dlls/               # Pre-built DLLs (RTX 4090 optimized)
│   ├── dense_assembly_kernels.dll
│   ├── assemblies_cuda_kernels.dll
│   └── ... (CUDA runtime DLLs)
└── build_scripts/      # Build system for Windows
```

### Key Insight: Two Connectivity Models

1. **Implicit Hash-Based** (our NEMO approach):
   - No weight storage, connectivity computed via hash
   - Memory: O(learned_connections) 
   - Used in: `assembly_projection_kernel.cu`

2. **Explicit Sparse CSR** (original assemblies approach):
   - Synapses stored explicitly in CSR format
   - Memory: O(n × k × p) synapses
   - Used in: `brain.cc`, `assemblies_cuda_brain_optimized.cu`

The existing C++ implementation uses sparse CSR, which is different from our hash-based approach but could be adapted.

---

## Profiling Results

### Kernel-Level Performance (CUDA)

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Projection kernel | 0.066 | Hash-based implicit connectivity |
| TopK selection (torch) | 0.171 | Finding top-k neurons |
| TopK selection (CuPy argsort) | 0.236 | Alternative, similar speed |
| Hebbian kernel | 0.215 | Weight updates |

**Key insight**: Raw CUDA kernels are fast. A 5-word sentence with ~45 projections should theoretically take ~13ms of kernel time.

### System-Level Performance

| Configuration | Time/Sentence | Throughput |
|---------------|---------------|------------|
| 50 sentences × 1 epoch | 21.5 ms | 46 sent/s |
| 349 sentences × 1 epoch | 30.1 ms | 33 sent/s |
| 349 sentences × 3 epochs | 30.1 ms | 33 sent/s |

**Observed vs Theoretical**: 30ms observed vs 13ms theoretical = **~17ms Python overhead per sentence**

### Breakdown by Word Type

| Word Type | Time (ms) | Projections |
|-----------|-----------|-------------|
| Motor (verbs) | 46.2 | ~12 (grounding + category + tau×3 + roles) |
| Visual (nouns) | 18.1 | ~8 |
| Function words | 21.6 | ~6 |

---

## Why Python Optimizations Failed

### Attempted Optimizations

1. **Batched Hebbian Learning**
   - Idea: Queue updates, apply in single kernel
   - Result: **Slower** (0.66x)
   - Reason: Array copying overhead exceeded kernel launch savings

2. **Assembly Caching**
   - Idea: Cache stable assemblies, skip redundant projections
   - Result: **Broke NEMO dynamics**
   - Reason: Projections are context-dependent; caching removes learned weight effects

3. **Reduced Projections**
   - Idea: Skip tau loop, reduce to essential projections only
   - Result: **Broke NEMO dynamics**
   - Reason: Tau iterations needed for assembly stabilization

4. **CuPy argsort vs torch.topk**
   - Result: **Marginal improvement** (~10%)
   - Reason: Both are already fast; not the bottleneck

### Root Cause Analysis

The bottleneck is **Python interpreter overhead**:
- Function call overhead (~1μs per call, thousands of calls)
- Dictionary lookups for area management
- Object attribute access
- Loop iteration overhead
- CuPy/PyTorch interop

---

## C++ Implementation Plan

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                                │
│  (High-level interface for training, generation, parsing)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     pybind11 Bindings                           │
│  (Thin wrapper exposing C++ classes to Python)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      C++ Core Library                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ NemoBrain   │  │ NemoLearner │  │ NemoParser  │             │
│  │ - areas     │  │ - training  │  │ - parsing   │             │
│  │ - weights   │  │ - stats     │  │ - QA        │             │
│  │ - project() │  │ - present() │  │ - parse()   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CUDA Kernels                              │
│  - projection_kernel (existing, optimized)                      │
│  - hebbian_kernel (existing, optimized)                         │
│  - fused_project_learn_kernel (new, combines ops)               │
└─────────────────────────────────────────────────────────────────┘
```

### Key C++ Classes

```cpp
// Core brain class
class NemoBrain {
public:
    NemoBrain(int n = 10000, int k = 100, float p = 0.1);
    
    // Core operations
    std::vector<uint32_t> project(Area area, const std::vector<uint32_t>& input, bool learn = true);
    void clear_all();
    
    // Assembly management
    void store_assembly(Area area, const std::string& word, const std::vector<uint32_t>& assembly);
    std::optional<std::vector<uint32_t>> get_assembly(Area area, const std::string& word);
    
private:
    // Pre-allocated GPU buffers
    thrust::device_vector<uint32_t> d_active;
    thrust::device_vector<float> d_result;
    thrust::device_vector<uint32_t> d_winners;
    
    // Learned weights (per area)
    std::array<LearnedWeights, NUM_AREAS> weights;
    
    // Current/previous activations
    std::array<std::optional<std::vector<uint32_t>>, NUM_AREAS> current;
    std::array<std::optional<std::vector<uint32_t>>, NUM_AREAS> prev;
};

// Learner class
class NemoLearner {
public:
    NemoLearner(int n = 10000, int k = 100);
    
    void present_sentence(
        const std::vector<std::string>& words,
        const std::vector<GroundingContext>& contexts,
        const std::vector<std::string>& roles,
        const std::string& mood = "declarative"
    );
    
    std::string get_category(const std::string& word);
    std::string get_thematic_role(const std::string& word);
    
private:
    NemoBrain brain;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_grounding;
    // ... other statistics
};
```

### Fused Kernel (New)

```cuda
// Combines projection + topk + hebbian in single kernel launch
__global__ void fused_project_learn(
    const uint32_t* active,      // Input assembly
    uint32_t* winners,           // Output winners
    uint32_t* prev_active,       // Previous activation (for Hebbian)
    uint32_t* l_src,             // Learned source neurons
    uint32_t* l_dst,             // Learned destination neurons
    float* l_delta,              // Learned weight deltas
    uint32_t* l_num,             // Number of learned connections
    uint32_t k, uint32_t n,
    uint32_t seed, float p,
    float beta, float w_max,
    bool learn
) {
    // Phase 1: Projection (all threads)
    // Phase 2: Block-level topk using shared memory
    // Phase 3: Hebbian update (if learn=true)
}
```

### Expected Performance

| Implementation | Time/Sentence | Speedup |
|----------------|---------------|---------|
| Current Python | 30 ms | 1x |
| C++ with CUDA | ~5 ms | 6x |
| C++ with fused kernel | ~2 ms | 15x |

### Build System

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(nemo_cpp LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(nemo_core SHARED
    src/brain.cpp
    src/learner.cpp
    src/parser.cpp
    src/kernels.cu
)

target_compile_features(nemo_core PUBLIC cxx_std_17)
set_target_properties(nemo_core PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

pybind11_add_module(nemo_cpp src/bindings.cpp)
target_link_libraries(nemo_cpp PRIVATE nemo_core)
```

---

## Implementation Roadmap

### Phase 1: Core C++ Library (Week 1-2)
- [ ] Port NemoBrain class
- [ ] Port projection and Hebbian kernels
- [ ] Implement area management
- [ ] Unit tests

### Phase 2: Learner and Parser (Week 2-3)
- [ ] Port NemoLearner class
- [ ] Port SentenceParser
- [ ] Port QuestionAnswerer
- [ ] Integration tests

### Phase 3: Python Bindings (Week 3)
- [ ] pybind11 bindings
- [ ] Numpy/CuPy interop
- [ ] Python API compatibility

### Phase 4: Optimization (Week 4)
- [ ] Fused kernels
- [ ] Memory pooling
- [ ] Benchmark and profile

---

## Current System Capabilities

Despite Python overhead, the current system is **fully functional**:

### Learning (100% accuracy)
- Emergent word categorization (NOUN, VERB, etc.)
- Thematic role assignment (AGENT, PATIENT, ACTION)
- Word order learning (SVO)
- VP assembly formation (229 patterns)

### Comprehension (100% accuracy)
- Sentence parsing
- Subject/Verb/Object extraction
- Question answering

### Real-Time Feasibility
- 100 words/second processing
- 40x faster than human speech
- Suitable for interactive applications

---

## Files Structure

```
src/nemo/language/emergent/
├── __init__.py          # Package exports
├── areas.py             # Brain area definitions
├── brain.py             # Core brain implementation
├── learner.py           # Language learner
├── generator.py         # Sentence generation
├── params.py            # Configuration parameters
├── training_data.py     # Training data generation
├── profiler.py          # Performance profiling
├── parser/
│   ├── __init__.py
│   ├── core.py          # Sentence parser
│   └── comprehension.py # Question answering
└── tests/
    ├── run_all.py       # Test runner
    ├── benchmark.py     # Performance benchmark
    ├── test_training.py
    ├── test_parser.py
    └── test_comprehension.py
```

---

## Conclusion

The NEMO emergent language system demonstrates that **meaningful language understanding can emerge from neurobiologically plausible mechanisms** with a tiny model (~370K neurons).

Current Python implementation is sufficient for research and prototyping. For production real-time applications, a C++ implementation would provide 6-15x speedup.

The key insight from optimization attempts: **NEMO's dynamics cannot be simplified without losing computational power**. Every projection contributes to learning. The path to speed is better implementation, not algorithmic shortcuts.

