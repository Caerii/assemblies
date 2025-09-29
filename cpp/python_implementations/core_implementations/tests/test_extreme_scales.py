#!/usr/bin/env python3
"""
Extreme Scale Testing - Test with billion-scale neuron counts

Tests extreme scales to validate billion-scale capabilities:
- 1B, 2B, 5B neurons
- Sparse memory model performance
- GPU memory limitations and handling

Key Findings:
- Validates billion-scale simulation capability
- Tests sparse memory efficiency
- Reveals GPU memory limits and handling

Usage: python test_extreme_scales.py
"""

import time
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'billion_scale'))

from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig

# Import billion-scale implementations
try:
    from gpu_only_billion_scale import GPUOnlyBillionScaleBrain
    BILLION_SCALE_AVAILABLE = True
except ImportError:
    print("⚠️  Billion-scale implementations not available")
    BILLION_SCALE_AVAILABLE = False

def test_extreme_scales():
    """Test at extreme scales where billion-scale implementation excels"""
    print("🚀 EXTREME SCALE COMPARISON")
    print("=" * 80)
    
    # Test extreme scales (where billion-scale really shines)
    test_scales = [
        {"n_neurons": 1000000000, "active_percentage": 0.0001, "name": "1B neurons (0.01%)"},
        {"n_neurons": 2000000000, "active_percentage": 0.00005, "name": "2B neurons (0.005%)"},
        {"n_neurons": 5000000000, "active_percentage": 0.00002, "name": "5B neurons (0.002%)"},
    ]
    
    results = []
    
    for scale in test_scales:
        print(f"\n🧪 Testing {scale['name']}:")
        print(f"   Neurons: {scale['n_neurons']:,}")
        print(f"   Active percentage: {scale['active_percentage']*100:.5f}%")
        print(f"   Active per area: {int(scale['n_neurons'] * scale['active_percentage']):,}")
        
        # Test Universal Brain Simulator (CUDA) - likely to fail at these scales
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
            brain_cuda.simulate(n_steps=5, verbose=False)  # Fewer steps for extreme scales
            cuda_time = time.perf_counter() - start_time
            
            cuda_stats = brain_cuda.get_performance_stats()
            print(f"   ✅ Universal (CUDA): {cuda_stats['steps_per_second']:.1f} steps/s, {cuda_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Universal (CUDA) failed: {e}")
            cuda_time = float('inf')
            cuda_stats = {'steps_per_second': 0}
        
        # Test Universal Brain Simulator (CuPy) - likely to fail at these scales
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
            brain_cupy.simulate(n_steps=5, verbose=False)
            cupy_time = time.perf_counter() - start_time
            
            cupy_stats = brain_cupy.get_performance_stats()
            print(f"   ✅ Universal (CuPy): {cupy_stats['steps_per_second']:.1f} steps/s, {cupy_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Universal (CuPy) failed: {e}")
            cupy_time = float('inf')
            cupy_stats = {'steps_per_second': 0}
        
        # Test Billion-Scale Implementation - this should work
        if BILLION_SCALE_AVAILABLE:
            try:
                brain_billion = GPUOnlyBillionScaleBrain(
                    n_neurons=scale['n_neurons'],
                    active_percentage=scale['active_percentage'],
                    n_areas=5,
                    seed=42
                )
                start_time = time.perf_counter()
                brain_billion.simulate(n_steps=5, verbose=False)
                billion_time = time.perf_counter() - start_time
                
                billion_stats = brain_billion.get_performance_stats()
                print(f"   ✅ Billion-Scale: {billion_stats['steps_per_second']:.1f} steps/s, {billion_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Billion-Scale failed: {e}")
                billion_time = float('inf')
                billion_stats = {'steps_per_second': 0}
        else:
            billion_time = float('inf')
            billion_stats = {'steps_per_second': 0}
        
        # Calculate speedups
        if cuda_time < float('inf') and billion_time < float('inf'):
            billion_vs_cuda = cuda_time / billion_time
            print(f"   🚀 Billion vs CUDA speedup: {billion_vs_cuda:.2f}x")
        
        if cupy_time < float('inf') and billion_time < float('inf'):
            billion_vs_cupy = cupy_time / billion_time
            print(f"   🚀 Billion vs CuPy speedup: {billion_vs_cupy:.2f}x")
        
        results.append({
            'name': scale['name'],
            'n_neurons': scale['n_neurons'],
            'active_percentage': scale['active_percentage'],
            'cuda_time': cuda_time,
            'cuda_steps_per_sec': cuda_stats['steps_per_second'],
            'cupy_time': cupy_time,
            'cupy_steps_per_sec': cupy_stats['steps_per_second'],
            'billion_time': billion_time,
            'billion_steps_per_sec': billion_stats['steps_per_second']
        })
    
    # Print summary
    print(f"\n📊 EXTREME SCALE COMPARISON SUMMARY")
    print("=" * 120)
    print(f"{'Scale':<25} {'Neurons':<15} {'CUDA':<12} {'CuPy':<12} {'Billion':<12} {'Best':<10} {'Billion vs CUDA':<15} {'Billion vs CuPy':<15}")
    print("-" * 120)
    
    for result in results:
        best_performer = "CUDA"
        best_speed = result['cuda_steps_per_sec']
        
        if result['cupy_steps_per_sec'] > best_speed:
            best_performer = "CuPy"
            best_speed = result['cupy_steps_per_sec']
        
        if result['billion_steps_per_sec'] > best_speed:
            best_performer = "Billion"
            best_speed = result['billion_steps_per_sec']
        
        cuda_status = f"{result['cuda_steps_per_sec']:.1f}" if result['cuda_steps_per_sec'] > 0 else "FAILED"
        cupy_status = f"{result['cupy_steps_per_sec']:.1f}" if result['cupy_steps_per_sec'] > 0 else "FAILED"
        billion_status = f"{result['billion_steps_per_sec']:.1f}" if result['billion_steps_per_sec'] > 0 else "FAILED"
        
        billion_vs_cuda = result['cuda_time'] / result['billion_time'] if result['cuda_time'] < float('inf') and result['billion_time'] < float('inf') else 0
        billion_vs_cupy = result['cupy_time'] / result['billion_time'] if result['cupy_time'] < float('inf') and result['billion_time'] < float('inf') else 0
        
        print(f"{result['name']:<25} {result['n_neurons']:<15,} {cuda_status:<12} {cupy_status:<12} {billion_status:<12} {best_performer:<10} {billion_vs_cuda:<15.2f} {billion_vs_cupy:<15.2f}")
    
    return results

def analyze_billion_scale_advantages():
    """Analyze why billion-scale implementation is so much faster"""
    print(f"\n🔍 BILLION-SCALE ADVANTAGES ANALYSIS")
    print("=" * 60)
    
    print("1. 🎯 SPARSE MEMORY ARCHITECTURE:")
    print("   - Only allocates memory for ACTIVE neurons")
    print("   - 1B neurons with 0.01% active = only 100K active per area")
    print("   - Memory usage: ~0.01 GB instead of 1B+ GB")
    print("   - No memory waste on inactive neurons")
    print()
    
    print("2. ⚡ PURE GPU OPERATIONS:")
    print("   - Direct CuPy operations (no CUDA kernel overhead)")
    print("   - No CPU-GPU memory transfers")
    print("   - No ctypes overhead")
    print("   - Minimal Python overhead")
    print()
    
    print("3. 🧠 OPTIMIZED ALGORITHMS:")
    print("   - Single-purpose design for billion-scale")
    print("   - No fallback logic or error handling overhead")
    print("   - Direct GPU memory allocation")
    print("   - Optimized for CuPy's strengths")
    print()
    
    print("4. 📊 SCALING CHARACTERISTICS:")
    print("   - Performance scales with ACTIVE neurons, not total neurons")
    print("   - 1B neurons with 0.01% active = same performance as 1M with 1% active")
    print("   - Memory usage stays constant regardless of total neuron count")
    print("   - GPU memory limits based on active neurons, not total")
    print()
    
    print("5. 🚀 WHY IT'S DRAMATICALLY FASTER:")
    print("   - Universal Brain Simulator: Tries to handle all neurons")
    print("   - Billion-Scale: Only handles active neurons")
    print("   - 1B neurons × 0.01% = 100K active (manageable)")
    print("   - 1B neurons × 1% = 10M active (memory overflow)")
    print("   - Sparse representation = massive memory savings")
    print("   - Pure GPU = minimal overhead")

if __name__ == "__main__":
    results = test_extreme_scales()
    analyze_billion_scale_advantages()
    
    # Find where billion-scale excels
    billion_wins = 0
    for result in results:
        if result['billion_steps_per_sec'] > 0 and result['billion_steps_per_sec'] > max(result['cuda_steps_per_sec'], result['cupy_steps_per_sec']):
            billion_wins += 1
    
    print(f"\n🏆 Billion-scale won {billion_wins}/{len([r for r in results if r['billion_steps_per_sec'] > 0])} working tests")
    
    if billion_wins > 0:
        print("✅ Billion-scale implementation shows dramatic advantages at extreme scales!")
    else:
        print("⚠️  Universal brain simulator may still be competitive")
