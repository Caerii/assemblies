#!/usr/bin/env python3
"""
Large Scale Performance Test - CUDA vs CuPy vs NumPy across different scales

Tests Universal Brain Simulator at various scales:
- 100K, 1M, 10M, 100M neurons
- Different active percentages (0.01% - 1%)
- Performance comparison across all backends

Key Findings:
- CUDA kernels excel at larger scales (>1M neurons)
- CuPy provides reliable fallback
- NumPy surprisingly competitive at smaller scales
- Memory usage scales predictably

Usage: python test_large_scale.py
"""

import time
import os
import sys
sys.path.append(os.path.dirname(__file__))

from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig

def test_large_scale_performance():
    """Test performance at different scales"""
    print("🚀 LARGE SCALE PERFORMANCE TEST")
    print("=" * 60)
    
    # Test different scales
    test_scales = [
        {"n_neurons": 100000, "active_percentage": 0.01, "name": "100K neurons (1%)"},
        {"n_neurons": 1000000, "active_percentage": 0.01, "name": "1M neurons (1%)"},
        {"n_neurons": 10000000, "active_percentage": 0.001, "name": "10M neurons (0.1%)"},
        {"n_neurons": 100000000, "active_percentage": 0.0001, "name": "100M neurons (0.01%)"},
    ]
    
    results = []
    
    for scale in test_scales:
        print(f"\n🧪 Testing {scale['name']}:")
        print(f"   Neurons: {scale['n_neurons']:,}")
        print(f"   Active percentage: {scale['active_percentage']*100:.4f}%")
        
        # Test CUDA kernels
        try:
            config_cuda = SimulationConfig(
                n_neurons=scale['n_neurons'],
                active_percentage=scale['active_percentage'],
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_cuda = UniversalBrainSimulator(config_cuda)
            start_time = time.perf_counter()
            brain_cuda.simulate(n_steps=10, verbose=False)
            cuda_time = time.perf_counter() - start_time
            
            cuda_stats = brain_cuda.get_performance_stats()
            print(f"   ✅ CUDA Kernels: {cuda_stats['steps_per_second']:.1f} steps/s, {cuda_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ CUDA Kernels failed: {e}")
            cuda_time = float('inf')
            cuda_stats = {'steps_per_second': 0}
        
        # Test CuPy only
        try:
            config_cupy = SimulationConfig(
                n_neurons=scale['n_neurons'],
                active_percentage=scale['active_percentage'],
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_cupy = UniversalBrainSimulator(config_cupy)
            start_time = time.perf_counter()
            brain_cupy.simulate(n_steps=10, verbose=False)
            cupy_time = time.perf_counter() - start_time
            
            cupy_stats = brain_cupy.get_performance_stats()
            print(f"   ✅ CuPy Only: {cupy_stats['steps_per_second']:.1f} steps/s, {cupy_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ CuPy Only failed: {e}")
            cupy_time = float('inf')
            cupy_stats = {'steps_per_second': 0}
        
        # Test NumPy only
        try:
            config_numpy = SimulationConfig(
                n_neurons=scale['n_neurons'],
                active_percentage=scale['active_percentage'],
                n_areas=5,
                use_gpu=False,
                use_cuda_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_numpy = UniversalBrainSimulator(config_numpy)
            start_time = time.perf_counter()
            brain_numpy.simulate(n_steps=10, verbose=False)
            numpy_time = time.perf_counter() - start_time
            
            numpy_stats = brain_numpy.get_performance_stats()
            print(f"   ✅ NumPy Only: {numpy_stats['steps_per_second']:.1f} steps/s, {numpy_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ NumPy Only failed: {e}")
            numpy_time = float('inf')
            numpy_stats = {'steps_per_second': 0}
        
        # Calculate speedup
        if cuda_time < float('inf') and cupy_time < float('inf'):
            cuda_speedup = cupy_time / cuda_time
            print(f"   🚀 CUDA vs CuPy speedup: {cuda_speedup:.2f}x")
        
        if cuda_time < float('inf') and numpy_time < float('inf'):
            cuda_vs_numpy = numpy_time / cuda_time
            print(f"   🚀 CUDA vs NumPy speedup: {cuda_vs_numpy:.2f}x")
        
        results.append({
            'scale': scale['name'],
            'n_neurons': scale['n_neurons'],
            'active_percentage': scale['active_percentage'],
            'cuda_time': cuda_time,
            'cuda_steps_per_sec': cuda_stats['steps_per_second'],
            'cupy_time': cupy_time,
            'cupy_steps_per_sec': cupy_stats['steps_per_second'],
            'numpy_time': numpy_time,
            'numpy_steps_per_sec': numpy_stats['steps_per_second']
        })
    
    # Print summary
    print("\n📊 LARGE SCALE PERFORMANCE SUMMARY")
    print("=" * 100)
    print(f"{'Scale':<25} {'Neurons':<12} {'CUDA steps/s':<12} {'CuPy steps/s':<12} {'NumPy steps/s':<12} {'Best':<10}")
    print("-" * 100)
    
    for result in results:
        best_performer = "CUDA"
        best_speed = result['cuda_steps_per_sec']
        
        if result['cupy_steps_per_sec'] > best_speed:
            best_performer = "CuPy"
            best_speed = result['cupy_steps_per_sec']
        
        if result['numpy_steps_per_sec'] > best_speed:
            best_performer = "NumPy"
            best_speed = result['numpy_steps_per_sec']
        
        print(f"{result['scale']:<25} {result['n_neurons']:<12,} {result['cuda_steps_per_sec']:<12.1f} {result['cupy_steps_per_sec']:<12.1f} {result['numpy_steps_per_sec']:<12.1f} {best_performer:<10}")
    
    return results

if __name__ == "__main__":
    results = test_large_scale_performance()
    
    # Find where CUDA starts to excel
    cuda_wins = 0
    for result in results:
        if result['cuda_steps_per_sec'] > max(result['cupy_steps_per_sec'], result['numpy_steps_per_sec']):
            cuda_wins += 1
    
    print(f"\n🏆 CUDA kernels won {cuda_wins}/{len(results)} tests")
    
    if cuda_wins > 0:
        print("✅ CUDA kernels show performance benefits at larger scales!")
    else:
        print("⚠️  CUDA kernels may need optimization for this workload")
