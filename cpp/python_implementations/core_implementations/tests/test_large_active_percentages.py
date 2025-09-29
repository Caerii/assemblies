#!/usr/bin/env python3
"""
Large Active Percentages Test - Test performance with higher active percentages

Tests with larger active percentages to stress test implementations:
- 1M neurons with 1-50% active
- Memory and performance stress testing
- CUDA vs CuPy vs NumPy at high loads

Key Findings:
- Reveals memory limits at high active percentages
- Shows performance degradation patterns
- Identifies optimal active percentage ranges

Usage: python test_large_active_percentages.py
"""

import time
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig

def test_large_active_percentages():
    """Test performance with larger active percentages"""
    print("🚀 LARGE ACTIVE PERCENTAGE PERFORMANCE TEST")
    print("=" * 70)
    
    # Test different active percentages with fixed neuron count
    base_neurons = 1000000  # 1M neurons
    test_percentages = [
        {"active_percentage": 0.01, "name": "1% active (10K per area)"},
        {"active_percentage": 0.05, "name": "5% active (50K per area)"},
        {"active_percentage": 0.1, "name": "10% active (100K per area)"},
        {"active_percentage": 0.2, "name": "20% active (200K per area)"},
        {"active_percentage": 0.5, "name": "50% active (500K per area)"},
    ]
    
    results = []
    
    for test_case in test_percentages:
        active_pct = test_case["active_percentage"]
        active_per_area = int(base_neurons * active_pct)
        
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {base_neurons:,}")
        print(f"   Active percentage: {active_pct*100:.1f}%")
        print(f"   Active per area: {active_per_area:,}")
        
        # Test CUDA kernels
        try:
            config_cuda = SimulationConfig(
                n_neurons=base_neurons,
                active_percentage=active_pct,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_cuda = UniversalBrainSimulator(config_cuda)
            start_time = time.perf_counter()
            brain_cuda.simulate(n_steps=5, verbose=False)  # Reduced steps for larger workloads
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
                n_neurons=base_neurons,
                active_percentage=active_pct,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_cupy = UniversalBrainSimulator(config_cupy)
            start_time = time.perf_counter()
            brain_cupy.simulate(n_steps=5, verbose=False)
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
                n_neurons=base_neurons,
                active_percentage=active_pct,
                n_areas=5,
                use_gpu=False,
                use_cuda_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_numpy = UniversalBrainSimulator(config_numpy)
            start_time = time.perf_counter()
            brain_numpy.simulate(n_steps=5, verbose=False)
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
        
        if cupy_time < float('inf') and numpy_time < float('inf'):
            cupy_vs_numpy = numpy_time / cupy_time
            print(f"   🚀 CuPy vs NumPy speedup: {cupy_vs_numpy:.2f}x")
        
        results.append({
            'active_percentage': active_pct,
            'active_per_area': active_per_area,
            'name': test_case['name'],
            'cuda_time': cuda_time,
            'cuda_steps_per_sec': cuda_stats['steps_per_second'],
            'cupy_time': cupy_time,
            'cupy_steps_per_sec': cupy_stats['steps_per_second'],
            'numpy_time': numpy_time,
            'numpy_steps_per_sec': numpy_stats['steps_per_second']
        })
    
    # Print summary
    print(f"\n📊 LARGE ACTIVE PERCENTAGE PERFORMANCE SUMMARY")
    print("=" * 120)
    print(f"{'Active %':<12} {'Per Area':<10} {'CUDA steps/s':<12} {'CuPy steps/s':<12} {'NumPy steps/s':<12} {'CUDA vs CuPy':<12} {'CUDA vs NumPy':<12} {'Best':<10}")
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
        
        cuda_vs_cupy = result['cupy_time'] / result['cuda_time'] if result['cuda_time'] < float('inf') and result['cupy_time'] < float('inf') else 0
        cuda_vs_numpy = result['numpy_time'] / result['cuda_time'] if result['cuda_time'] < float('inf') and result['numpy_time'] < float('inf') else 0
        
        print(f"{result['active_percentage']*100:<12.1f} {result['active_per_area']:<10,} {result['cuda_steps_per_sec']:<12.1f} {result['cupy_steps_per_sec']:<12.1f} {result['numpy_steps_per_sec']:<12.1f} {cuda_vs_cupy:<12.2f} {cuda_vs_numpy:<12.2f} {best_performer:<10}")
    
    return results

def test_extreme_scales():
    """Test with extreme scales to really stress test the implementations"""
    print(f"\n🔥 EXTREME SCALE STRESS TEST")
    print("=" * 70)
    
    # Test extreme scales
    extreme_tests = [
        {"n_neurons": 10000000, "active_percentage": 0.1, "name": "10M neurons, 10% active (1M per area)"},
        {"n_neurons": 50000000, "active_percentage": 0.05, "name": "50M neurons, 5% active (2.5M per area)"},
        {"n_neurons": 100000000, "active_percentage": 0.02, "name": "100M neurons, 2% active (2M per area)"},
    ]
    
    for test_case in extreme_tests:
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {test_case['n_neurons']:,}")
        print(f"   Active percentage: {test_case['active_percentage']*100:.1f}%")
        print(f"   Active per area: {int(test_case['n_neurons'] * test_case['active_percentage']):,}")
        
        # Only test CUDA and CuPy for extreme scales (NumPy will likely be too slow)
        try:
            config_cuda = SimulationConfig(
                n_neurons=test_case['n_neurons'],
                active_percentage=test_case['active_percentage'],
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            brain_cuda = UniversalBrainSimulator(config_cuda)
            start_time = time.perf_counter()
            brain_cuda.simulate(n_steps=3, verbose=False)  # Very few steps for extreme scales
            cuda_time = time.perf_counter() - start_time
            
            cuda_stats = brain_cuda.get_performance_stats()
            print(f"   ✅ CUDA Kernels: {cuda_stats['steps_per_second']:.1f} steps/s, {cuda_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ CUDA Kernels failed: {e}")
        
        try:
            config_cupy = SimulationConfig(
                n_neurons=test_case['n_neurons'],
                active_percentage=test_case['active_percentage'],
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

if __name__ == "__main__":
    # Test large active percentages
    results = test_large_active_percentages()
    
    # Find where CUDA starts to excel
    cuda_wins = 0
    cuda_vs_cupy_wins = 0
    
    for result in results:
        if result['cuda_steps_per_sec'] > max(result['cupy_steps_per_sec'], result['numpy_steps_per_sec']):
            cuda_wins += 1
        
        if result['cuda_steps_per_sec'] > result['cupy_steps_per_sec']:
            cuda_vs_cupy_wins += 1
    
    print(f"\n🏆 CUDA kernels won {cuda_wins}/{len(results)} overall tests")
    print(f"🏆 CUDA kernels beat CuPy in {cuda_vs_cupy_wins}/{len(results)} tests")
    
    if cuda_wins > 0:
        print("✅ CUDA kernels show performance benefits at larger active percentages!")
    else:
        print("⚠️  CUDA kernels may need optimization for this workload")
    
    # Test extreme scales
    test_extreme_scales()
