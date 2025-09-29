#!/usr/bin/env python3
"""
Conservative Scale Testing - Find working limits for all implementations

Tests conservative scales to identify working limits:
- 100K-1M neurons (conservative range)
- 1-50% active percentages
- CUDA vs CuPy vs NumPy performance limits

Key Findings:
- Identifies safe operating ranges
- Reveals memory and performance bottlenecks
- Helps determine optimal configuration

Usage: python test_conservative_scales.py
"""

import time
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig

def test_conservative_scales():
    """Test with conservative scales to find the working limits"""
    print("🚀 CONSERVATIVE SCALE TEST")
    print("=" * 60)
    
    # Test conservative scales
    test_scales = [
        {"n_neurons": 100000, "active_percentage": 0.01, "name": "100K neurons (1%)"},
        {"n_neurons": 100000, "active_percentage": 0.05, "name": "100K neurons (5%)"},
        {"n_neurons": 100000, "active_percentage": 0.1, "name": "100K neurons (10%)"},
        {"n_neurons": 100000, "active_percentage": 0.2, "name": "100K neurons (20%)"},
        {"n_neurons": 100000, "active_percentage": 0.5, "name": "100K neurons (50%)"},
        {"n_neurons": 500000, "active_percentage": 0.01, "name": "500K neurons (1%)"},
        {"n_neurons": 500000, "active_percentage": 0.05, "name": "500K neurons (5%)"},
        {"n_neurons": 500000, "active_percentage": 0.1, "name": "500K neurons (10%)"},
        {"n_neurons": 1000000, "active_percentage": 0.01, "name": "1M neurons (1%)"},
        {"n_neurons": 1000000, "active_percentage": 0.02, "name": "1M neurons (2%)"},
        {"n_neurons": 1000000, "active_percentage": 0.05, "name": "1M neurons (5%)"},
    ]
    
    results = []
    
    for scale in test_scales:
        active_per_area = int(scale['n_neurons'] * scale['active_percentage'])
        
        print(f"\n🧪 Testing {scale['name']}:")
        print(f"   Neurons: {scale['n_neurons']:,}")
        print(f"   Active percentage: {scale['active_percentage']*100:.1f}%")
        print(f"   Active per area: {active_per_area:,}")
        
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
            brain_cuda.simulate(n_steps=3, verbose=False)  # Very few steps
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
            brain_cupy.simulate(n_steps=3, verbose=False)
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
            brain_numpy.simulate(n_steps=3, verbose=False)
            numpy_time = time.perf_counter() - start_time
            
            numpy_stats = brain_numpy.get_performance_stats()
            print(f"   ✅ NumPy Only: {numpy_stats['steps_per_second']:.1f} steps/s, {numpy_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ NumPy Only failed: {e}")
            numpy_time = float('inf')
            numpy_stats = {'steps_per_second': 0}
        
        # Calculate speedups
        if cuda_time < float('inf') and cupy_time < float('inf'):
            cuda_speedup = cupy_time / cuda_time
            print(f"   🚀 CUDA vs CuPy speedup: {cuda_speedup:.2f}x")
        
        if cuda_time < float('inf') and numpy_time < float('inf'):
            cuda_vs_numpy = numpy_time / cuda_time
            print(f"   🚀 CUDA vs NumPy speedup: {cuda_vs_numpy:.2f}x")
        
        results.append({
            'name': scale['name'],
            'n_neurons': scale['n_neurons'],
            'active_percentage': scale['active_percentage'],
            'active_per_area': active_per_area,
            'cuda_time': cuda_time,
            'cuda_steps_per_sec': cuda_stats['steps_per_second'],
            'cupy_time': cupy_time,
            'cupy_steps_per_sec': cupy_stats['steps_per_second'],
            'numpy_time': numpy_time,
            'numpy_steps_per_sec': numpy_stats['steps_per_second']
        })
    
    # Print summary
    print(f"\n📊 CONSERVATIVE SCALE PERFORMANCE SUMMARY")
    print("=" * 120)
    print(f"{'Test':<25} {'Neurons':<10} {'Active%':<8} {'Per Area':<10} {'CUDA':<8} {'CuPy':<8} {'NumPy':<8} {'Best':<8}")
    print("-" * 120)
    
    for result in results:
        best_performer = "CUDA"
        best_speed = result['cuda_steps_per_sec']
        
        if result['cupy_steps_per_sec'] > best_speed:
            best_performer = "CuPy"
            best_speed = result['cupy_steps_per_sec']
        
        if result['numpy_steps_per_sec'] > best_speed:
            best_performer = "NumPy"
            best_speed = result['numpy_steps_per_sec']
        
        cuda_status = "✅" if result['cuda_steps_per_sec'] > 0 else "❌"
        cupy_status = "✅" if result['cupy_steps_per_sec'] > 0 else "❌"
        numpy_status = "✅" if result['numpy_steps_per_sec'] > 0 else "❌"
        
        print(f"{result['name']:<25} {result['n_neurons']:<10,} {result['active_percentage']*100:<8.1f} {result['active_per_area']:<10,} {cuda_status:<8} {cupy_status:<8} {numpy_status:<8} {best_performer:<8}")
    
    # Find the limits
    cuda_working = [r for r in results if r['cuda_steps_per_sec'] > 0]
    cupy_working = [r for r in results if r['cupy_steps_per_sec'] > 0]
    numpy_working = [r for r in results if r['numpy_steps_per_sec'] > 0]
    
    print(f"\n🔍 WORKING LIMITS:")
    print(f"   CUDA kernels: {len(cuda_working)}/{len(results)} tests passed")
    print(f"   CuPy: {len(cupy_working)}/{len(results)} tests passed")
    print(f"   NumPy: {len(numpy_working)}/{len(results)} tests passed")
    
    if cuda_working:
        max_cuda = max(cuda_working, key=lambda x: x['active_per_area'])
        print(f"   CUDA max active per area: {max_cuda['active_per_area']:,}")
    
    if cupy_working:
        max_cupy = max(cupy_working, key=lambda x: x['active_per_area'])
        print(f"   CuPy max active per area: {max_cupy['active_per_area']:,}")
    
    if numpy_working:
        max_numpy = max(numpy_working, key=lambda x: x['active_per_area'])
        print(f"   NumPy max active per area: {max_numpy['active_per_area']:,}")
    
    return results

if __name__ == "__main__":
    results = test_conservative_scales()
    
    # Find where CUDA starts to excel
    cuda_wins = 0
    cuda_vs_cupy_wins = 0
    
    for result in results:
        if result['cuda_steps_per_sec'] > 0 and result['cuda_steps_per_sec'] > max(result['cupy_steps_per_sec'], result['numpy_steps_per_sec']):
            cuda_wins += 1
        
        if result['cuda_steps_per_sec'] > result['cupy_steps_per_sec']:
            cuda_vs_cupy_wins += 1
    
    print(f"\n🏆 CUDA kernels won {cuda_wins}/{len([r for r in results if r['cuda_steps_per_sec'] > 0])} working tests")
    print(f"🏆 CUDA kernels beat CuPy in {cuda_vs_cupy_wins}/{len([r for r in results if r['cuda_steps_per_sec'] > 0 and r['cupy_steps_per_sec'] > 0])} tests")
