# CUDA Kernel Optimization Test Suite

This directory contains isolated test files for validating each algorithmic improvement to the CUDA kernels.

## Test Files

### 1. `test_warp_reduction.cu`
Tests the warp-level reduction optimization for atomic add operations.

**What it tests:**
- Current atomic add implementation vs warp reduction
- Performance at different scales (100K, 1M, 10M neurons)
- Correctness validation with known inputs
- Memory access pattern analysis

**Expected improvements:**
- 5.33x reduction in atomic operations
- Better memory bandwidth utilization
- Improved scalability

### 2. `test_radix_selection.cu`
Tests the radix selection optimization for top-k selection.

**What it tests:**
- Current bitonic sort vs radix selection
- Performance at different k values (10, 100, 1000)
- Correctness validation with known inputs
- Scalability with different neuron counts

**Expected improvements:**
- 19.4x speedup for typical values (n=256, k=10)
- O(n log k) vs O(n log²n) complexity
- Better memory access patterns

### 3. `test_memory_coalescing.cu`
Tests memory coalescing optimizations for better GPU memory utilization.

**What it tests:**
- Random vs coalesced memory access patterns
- Memory bandwidth utilization
- Different data layouts and access patterns
- Vectorized access patterns

**Expected improvements:**
- 3x memory access speed
- 90% vs 30% bandwidth utilization
- Better cache efficiency

### 4. `test_consolidated_kernels.cu`
Tests the consolidated kernel implementations that combine all optimizations.

**What it tests:**
- End-to-end performance comparison
- Memory usage analysis
- All kernel combinations
- Theoretical vs actual performance

**Expected improvements:**
- Combined benefits of all optimizations
- Better memory efficiency
- Improved overall performance

### 5. `test_benchmark_suite.cu`
Comprehensive benchmark suite that tests all improvements together.

**What it tests:**
- All individual optimizations
- End-to-end performance
- Scalability across different scales
- Memory usage analysis
- Generates comprehensive report

## Building and Running Tests

### Prerequisites
- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 6.0 or later
- C++ compiler (gcc, clang, or MSVC)

### Building Individual Tests
```bash
# Warp reduction test
nvcc -o test_warp_reduction test_warp_reduction.cu -lcudart -lcurand

# Radix selection test
nvcc -o test_radix_selection test_radix_selection.cu -lcudart

# Memory coalescing test
nvcc -o test_memory_coalescing test_memory_coalescing.cu -lcudart

# Consolidated kernels test
nvcc -o test_consolidated_kernels test_consolidated_kernels.cu -lcudart -lcurand

# Benchmark suite
nvcc -o test_benchmark_suite test_benchmark_suite.cu -lcudart -lcurand
```

### Running Tests
```bash
# Run individual tests
./test_warp_reduction
./test_radix_selection
./test_memory_coalescing
./test_consolidated_kernels

# Run comprehensive benchmark
./test_benchmark_suite
```

## Expected Results

### Performance Improvements
- **Warp Reduction**: 5.33x speedup for atomic operations
- **Radix Selection**: 19.4x speedup for top-k selection
- **Memory Coalescing**: 3x memory access speed
- **Overall**: Combined 20-50x speedup depending on workload

### Memory Efficiency
- Better bandwidth utilization (90% vs 30%)
- Reduced memory fragmentation
- Improved cache efficiency
- Lower memory usage per operation

### Scalability
- Better performance at larger scales
- More consistent performance across different neuron counts
- Improved memory usage patterns
- Better GPU utilization

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce test scale or increase GPU memory
2. **Compilation errors**: Ensure CUDA toolkit is properly installed
3. **Performance issues**: Check GPU compute capability and driver version

### Debug Mode
Compile with debug flags for detailed output:
```bash
nvcc -g -G -o test_warp_reduction test_warp_reduction.cu -lcudart -lcurand
```

## Contributing

When adding new tests:
1. Follow the existing naming convention
2. Include comprehensive documentation
3. Add correctness validation
4. Include performance measurements
5. Update this README

## License

This test suite is part of the assemblies project and follows the same license terms.
