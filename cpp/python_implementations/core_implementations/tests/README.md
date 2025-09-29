# Test Suite

Comprehensive testing for brain simulation implementations.

## Files

### Core Tests
- `test_cuda_kernels.py` - CUDA kernel functionality
- `test_comprehensive_cuda_kernels.py` - Complete CUDA testing
- `test_large_scale.py` - Scale performance testing
- `test_conservative_scales.py` - Conservative scale limits
- `test_large_active_percentages.py` - High active percentage testing
- `test_extreme_scales.py` - Billion-scale testing
- `compare_implementations.py` - Implementation comparison

### Client & Interface Tests
- `test_client.py` - Lightweight thin client interface
- `test_modular_simulator.py` - Modular simulator architecture
- `test_cleanup_fixes.py` - Cleanup and memory management

### Core Operations Tests
- `test_core_operations.py` - Core brain operations (project, merge, association)
- `test_large_scale_new.py` - Updated large scale testing

### Optimized Implementation Tests
- `test_all_optimized_implementations.py` - All optimized CUDA implementations
- `test_extreme_limits.py` - Extreme scale limits (2B-10B neurons)

## Usage

```bash
python test_cuda_kernels.py
python test_client.py
python test_core_operations.py
python test_all_optimized_implementations.py
```
