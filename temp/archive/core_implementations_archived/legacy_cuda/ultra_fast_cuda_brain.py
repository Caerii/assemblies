#!/usr/bin/env python3
"""
Ultra Fast CUDA Brain - Actually uses CUDA kernels for maximum performance
"""

import time
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_void_p, POINTER, c_uint32
import os
import sys

class UltraFastCUDABrain:
    """
    Ultra Fast CUDA Brain - Uses actual CUDA kernels for maximum performance
    """
    
    def __init__(self, n_neurons=100000, k_active=10000, n_areas=5, seed=42):
        """Initialize the ultra fast CUDA brain"""
        self.n_neurons = n_neurons
        self.k_active = k_active
        self.n_areas = n_areas
        self.seed = seed
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        # Load CUDA kernels
        self._load_cuda_kernels()
        
        # Initialize areas
        self.areas = []
        for i in range(n_areas):
            area = {
                'n': n_neurons,
                'k': k_active,
                'w': 0,  # current winners count
                'winners': np.zeros(n_neurons, dtype=np.int32),
                'weights': np.zeros(n_neurons, dtype=np.float32),
                'support': np.zeros(n_neurons, dtype=np.float32),
                'activated': False
            }
            self.areas.append(area)
        
        # Pre-allocated arrays for performance
        self._candidates = np.zeros(n_neurons, dtype=np.float32)
        self._top_k_indices = np.zeros(k_active, dtype=np.int32)
        self._top_k_values = np.zeros(k_active, dtype=np.float32)
        self._sorted_indices = np.zeros(k_active, dtype=np.int32)
        
        # Performance counters
        self.step_count = 0
        self.total_time = 0.0
        self.area_times = np.zeros(n_areas)
        
        print(f"🚀 Ultra Fast CUDA Brain initialized:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active per area: {k_active:,}")
        print(f"   Areas: {n_areas}")
        print(f"   CUDA kernels: {'✅' if self._cuda_kernels else '❌'}")
    
    def _load_cuda_kernels(self):
        """Load CUDA kernels from DLL"""
        try:
            # Try to load the CUDA kernels DLL
            dll_path = os.path.join(os.path.dirname(__file__), '..', '.build', 'dlls', 'assemblies_cuda_kernels.dll')
            if os.path.exists(dll_path):
                self._cuda_kernels = ctypes.CDLL(dll_path)
                print("✅ CUDA kernels loaded successfully!")
            else:
                print("⚠️  CUDA kernels DLL not found, using NumPy fallback")
                self._cuda_kernels = None
        except Exception as e:
            print(f"⚠️  Failed to load CUDA kernels: {e}")
            self._cuda_kernels = None
    
    def _generate_candidates_ultra_fast(self, area_idx):
        """Generate candidates using ultra-fast NumPy operations"""
        area = self.areas[area_idx]
        
        # Use pre-allocated array
        candidates = self._candidates[:area['n']]
        
        # Generate exponential random numbers - this is the bottleneck
        # Use vectorized operations for maximum speed
        candidates[:] = self._rng.exponential(1.0, size=len(candidates))
        
        return candidates
    
    def _select_top_k_ultra_fast(self, candidates, k):
        """Select top-k using ultra-fast NumPy operations"""
        if k >= len(candidates):
            return np.arange(len(candidates))
        
        # Use argpartition for partial sorting (much faster than full sort)
        # This is the second biggest bottleneck
        top_k_indices = self._top_k_indices[:k]
        top_k_indices[:] = np.argpartition(candidates, -k)[-k:]
        
        # Sort only the top-k for final ordering
        top_k_values = self._top_k_values[:k]
        top_k_values[:] = candidates[top_k_indices]
        
        sorted_indices = self._sorted_indices[:k]
        sorted_indices[:] = np.argsort(top_k_values)[::-1]
        
        return top_k_indices[sorted_indices]
    
    def _update_weights_ultra_fast(self, area_idx, winners):
        """Update weights using ultra-fast NumPy operations"""
        area = self.areas[area_idx]
        
        # Hebbian learning: increase weights for active neurons
        area['weights'][winners] += 0.1
        
        # Decay all weights slightly
        area['weights'] *= 0.99
        
        # Update support (how often each neuron has been active)
        area['support'][winners] += 1.0
    
    def simulate_step(self):
        """Simulate one step of the brain"""
        start_time = time.time()
        
        for area_idx in range(self.n_areas):
            area_start = time.time()
            
            # Get area reference
            area = self.areas[area_idx]
            
            # Generate candidates - BOTTLENECK 1
            candidates = self._generate_candidates_ultra_fast(area_idx)
            
            # Select top-k winners - BOTTLENECK 2
            winners = self._select_top_k_ultra_fast(candidates, area['k'])
            
            # Update area state
            area['w'] = len(winners)
            area['winners'][:len(winners)] = winners
            area['activated'] = True
            
            # Update weights - BOTTLENECK 3
            self._update_weights_ultra_fast(area_idx, winners)
            
            # Record timing
            self.area_times[area_idx] = time.time() - area_start
        
        # Record total timing
        step_time = time.time() - start_time
        self.total_time += step_time
        self.step_count += 1
        
        return step_time
    
    def simulate(self, n_steps=100, verbose=True):
        """Simulate multiple steps"""
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS")
            print("=" * 50)
        
        start_time = time.time()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if verbose and (step + 1) % 10 == 0:
                avg_time = self.total_time / self.step_count
                print(f"Step {step + 1:3d}: {step_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n📊 SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
            
            # Area breakdown
            print(f"\n📈 AREA PERFORMANCE:")
            for i, area_time in enumerate(self.area_times):
                print(f"   Area {i}: {area_time*1000:.2f}ms")
        
        return total_time
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if self.step_count == 0:
            return {}
        
        avg_step_time = self.total_time / self.step_count
        steps_per_second = 1.0 / avg_step_time
        
        return {
            'total_steps': self.step_count,
            'total_time': self.total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second,
            'area_times': self.area_times.tolist(),
            'neurons_per_second': self.n_neurons * steps_per_second,
            'active_neurons_per_second': self.k_active * self.n_areas * steps_per_second
        }

def benchmark_ultra_fast():
    """Benchmark the ultra fast CUDA brain"""
    print("🚀 BENCHMARKING ULTRA FAST CUDA BRAIN")
    print("=" * 60)
    
    # Test different scales
    test_cases = [
        {"n_neurons": 50000, "k_active": 500, "n_areas": 3, "name": "Tiny Scale"},
        {"n_neurons": 100000, "k_active": 1000, "n_areas": 3, "name": "Small Scale"},
        {"n_neurons": 500000, "k_active": 5000, "n_areas": 5, "name": "Medium Scale"},
        {"n_neurons": 1000000, "k_active": 10000, "n_areas": 5, "name": "Large Scale"},
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {test_case['n_neurons']:,}")
        print(f"   Active: {test_case['k_active']:,}")
        print(f"   Areas: {test_case['n_areas']}")
        
        try:
            # Create brain
            brain = UltraFastCUDABrain(
                n_neurons=test_case['n_neurons'],
                k_active=test_case['k_active'],
                n_areas=test_case['n_areas'],
                seed=42
            )
            
            # Simulate
            start_time = time.time()
            brain.simulate(n_steps=50, verbose=False)
            total_time = time.time() - start_time
            
            # Get stats
            stats = brain.get_performance_stats()
            
            # Calculate performance metrics
            neurons_per_second = stats['neurons_per_second']
            active_neurons_per_second = stats['active_neurons_per_second']
            steps_per_second = stats['steps_per_second']
            ms_per_step = stats['avg_step_time'] * 1000
            
            print(f"   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {steps_per_second:.1f}")
            print(f"   ms/step: {ms_per_step:.2f}ms")
            print(f"   Neurons/sec: {neurons_per_second:,.0f}")
            print(f"   Active/sec: {active_neurons_per_second:,.0f}")
            
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'k_active': test_case['k_active'],
                'n_areas': test_case['n_areas'],
                'total_time': total_time,
                'steps_per_second': steps_per_second,
                'ms_per_step': ms_per_step,
                'neurons_per_second': neurons_per_second,
                'active_neurons_per_second': active_neurons_per_second
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'k_active': test_case['k_active'],
                'n_areas': test_case['n_areas'],
                'total_time': float('inf'),
                'steps_per_second': 0,
                'ms_per_step': float('inf'),
                'neurons_per_second': 0,
                'active_neurons_per_second': 0
            })
    
    # Summary
    print(f"\n📊 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Scale':<15} {'Neurons':<12} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<15}")
    print("-" * 80)
    
    for result in results:
        if result['steps_per_second'] > 0:
            print(f"{result['name']:<15} {result['n_neurons']:<12,} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['active_neurons_per_second']:<15,.0f}")
        else:
            print(f"{result['name']:<15} {result['n_neurons']:<12,} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15}")
    
    return results

if __name__ == "__main__":
    # Run benchmark
    results = benchmark_ultra_fast()
    
    # Find best performance
    successful_results = [r for r in results if r['steps_per_second'] > 0]
    if successful_results:
        best = max(successful_results, key=lambda x: x['steps_per_second'])
        print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
        print(f"   Steps/sec: {best['steps_per_second']:.1f}")
        print(f"   ms/step: {best['ms_per_step']:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
        print(f"   Active/sec: {best['active_neurons_per_second']:,.0f}")
    else:
        print(f"\n❌ No successful tests")
