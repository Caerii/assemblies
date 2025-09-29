#!/usr/bin/env python3
"""
Hybrid GPU Brain - Uses CuPy for memory management but NumPy for computation
=======================================================================

This version uses CuPy for GPU memory allocation but falls back to NumPy
for computation when CURAND is not available. This gives us the best of both worlds.
"""

import cupy as cp
import numpy as np
import time
import ctypes
import os
from typing import Tuple, List, Optional

class HybridGPUBrain:
    def __init__(self, n_neurons: int, n_areas: int = 5, target_memory_gb: float = 12.0):
        """
        Initialize the hybrid GPU brain.
        
        Args:
            n_neurons: Total number of neurons
            n_areas: Number of brain areas
            target_memory_gb: Target GPU memory usage in GB
        """
        self.n_neurons = n_neurons
        self.n_areas = n_areas
        self.target_memory_gb = target_memory_gb
        
        # Calculate optimal active percentage based on memory constraints
        self.active_percent = self._calculate_optimal_active_percent()
        self.n_active_per_area = int(n_neurons * self.active_percent)
        
        print(f"🧠 Hybrid GPU Brain:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Areas: {n_areas}")
        print(f"   Active per area: {self.n_active_per_area:,} ({self.active_percent*100:.4f}%)")
        print(f"   Target memory: {target_memory_gb:.1f} GB")
        
        # Test CuPy availability
        self.use_cupy = self._test_cupy()
        
        # Initialize arrays
        self._initialize_arrays()
        
        # Load CUDA kernels if available
        self.cuda_kernels = self._load_cuda_kernels()
        
        print(f"   Memory per area: {self.memory_per_area_gb:.2f} GB")
        print(f"   Total memory: {self.total_memory_gb:.2f} GB")
        print(f"   CuPy: {'✅' if self.use_cupy else '❌'}")
        print(f"   CUDA kernels: {'✅' if self.cuda_kernels else '❌'}")
        
    def _calculate_optimal_active_percent(self) -> float:
        """Calculate optimal active percentage based on memory constraints."""
        # Estimate memory per neuron (float32 = 4 bytes)
        bytes_per_neuron = 4
        max_neurons_for_target = int(self.target_memory_gb * 1024**3 / bytes_per_neuron)
        
        # Start with 0.1% and scale down if needed
        active_percent = 0.001
        
        # If we need more memory than available, reduce active percentage
        if self.n_neurons > max_neurons_for_target:
            active_percent = max_neurons_for_target / self.n_neurons
            active_percent = max(active_percent, 0.00001)  # Minimum 0.001%
            
        return active_percent
    
    def _test_cupy(self) -> bool:
        """Test if CuPy is working properly."""
        try:
            # Test basic CuPy functionality
            test_array = cp.zeros(100, dtype=cp.float32)
            del test_array
            return True
        except Exception as e:
            print(f"   ⚠️  CuPy not available: {e}")
            return False
    
    def _initialize_arrays(self):
        """Initialize arrays on GPU if possible, otherwise on CPU."""
        try:
            self.areas = []
            self.weights = []
            self.activations = []
            self.candidates = []
            
            for i in range(self.n_areas):
                if self.use_cupy:
                    # Create arrays on GPU
                    area = cp.zeros(self.n_active_per_area, dtype=cp.float32)
                    weights = cp.random.exponential(1.0, size=(self.n_active_per_area, self.n_active_per_area), dtype=cp.float32)
                    activations = cp.zeros(self.n_active_per_area, dtype=cp.float32)
                    candidates = cp.zeros(self.n_active_per_area, dtype=cp.float32)
                else:
                    # Create arrays on CPU
                    area = np.zeros(self.n_active_per_area, dtype=np.float32)
                    weights = np.random.exponential(1.0, size=(self.n_active_per_area, self.n_active_per_area)).astype(np.float32)
                    activations = np.zeros(self.n_active_per_area, dtype=np.float32)
                    candidates = np.zeros(self.n_active_per_area, dtype=np.float32)
                
                self.areas.append(area)
                self.weights.append(weights)
                self.activations.append(activations)
                self.candidates.append(candidates)
            
            # Calculate memory usage
            if self.use_cupy:
                self.memory_per_area_gb = self.areas[0].nbytes / 1024**3
            else:
                self.memory_per_area_gb = self.areas[0].nbytes / 1024**3
            self.total_memory_gb = self.memory_per_area_gb * self.n_areas * 4  # 4 arrays per area
            
            print(f"   ✅ Arrays initialized successfully!")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize arrays: {e}")
            raise
    
    def _load_cuda_kernels(self) -> bool:
        """Load CUDA kernels if available."""
        try:
            cuda_dll_path = os.path.join(os.path.dirname(__file__), '..', '.build', 'dlls', 'assemblies_cuda_kernels.dll')
            if os.path.exists(cuda_dll_path):
                self.cuda_lib = ctypes.CDLL(cuda_dll_path)
                return True
            return False
        except Exception as e:
            print(f"   ⚠️  CUDA kernels not available: {e}")
            return False
    
    def simulate_step(self) -> float:
        """Simulate one step of the neural network."""
        start_time = time.time()
        
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]
            weights = self.weights[area_idx]
            activations = self.activations[area_idx]
            candidates = self.candidates[area_idx]
            
            if self.use_cupy:
                # Use CuPy for GPU computation
                try:
                    # Generate random candidates using GPU
                    candidates[:] = cp.random.exponential(1.0, size=len(candidates))
                    
                    # Compute activations using matrix multiplication on GPU
                    activations[:] = cp.dot(weights, area)
                    
                    # Apply threshold and update area
                    threshold = cp.percentile(activations, 90)
                    area[:] = cp.where(activations > threshold, candidates, 0.0)
                    
                    # Update weights (simplified Hebbian learning)
                    if cp.random.random() < 0.01:  # 1% chance to update weights
                        learning_rate = 0.001
                        weight_update = cp.outer(area, area) * learning_rate
                        weights[:] = cp.clip(weights + weight_update, 0.0, 10.0)
                        
                except Exception as e:
                    # Fallback to NumPy if CuPy fails
                    print(f"   ⚠️  CuPy failed, falling back to NumPy: {e}")
                    self.use_cupy = False
                    self._initialize_arrays()
                    return self.simulate_step()
            else:
                # Use NumPy for CPU computation
                candidates[:] = np.random.exponential(1.0, size=len(candidates))
                activations[:] = np.dot(weights, area)
                threshold = np.percentile(activations, 90)
                area[:] = np.where(activations > threshold, candidates, 0.0)
                
                if np.random.random() < 0.01:
                    learning_rate = 0.001
                    weight_update = np.outer(area, area) * learning_rate
                    weights[:] = np.clip(weights + weight_update, 0.0, 10.0)
        
        return time.time() - start_time
    
    def benchmark(self, n_steps: int = 10) -> dict:
        """Benchmark the neural network performance."""
        print(f"\n🚀 Running benchmark: {n_steps} steps...")
        
        times = []
        for step in range(n_steps):
            step_time = self.simulate_step()
            times.append(step_time)
            
            if step % max(1, n_steps // 10) == 0:
                print(f"   Step {step+1}/{n_steps}: {step_time*1000:.2f}ms")
        
        avg_time = np.mean(times)
        steps_per_sec = 1.0 / avg_time
        neurons_per_sec = self.n_neurons * steps_per_sec
        active_per_sec = self.n_active_per_area * self.n_areas * steps_per_sec
        
        return {
            'avg_time': avg_time,
            'steps_per_sec': steps_per_sec,
            'ms_per_step': avg_time * 1000,
            'neurons_per_sec': neurons_per_sec,
            'active_per_sec': active_per_sec,
            'memory_gb': self.total_memory_gb,
            'using_cupy': self.use_cupy
        }

def test_hybrid_gpu_brain():
    """Test hybrid GPU brain with different scales."""
    print("🌍 TESTING HYBRID GPU BRAIN")
    print("=" * 50)
    
    test_scales = [
        (1_000_000, "Million Scale"),
        (10_000_000, "Ten Million Scale"),
        (100_000_000, "Hundred Million Scale"),
        (1_000_000_000, "BILLION SCALE")
    ]
    
    results = []
    
    for n_neurons, scale_name in test_scales:
        print(f"\n🧪 Testing {scale_name}:")
        print(f"   Neurons: {n_neurons:,}")
        
        try:
            brain = HybridGPUBrain(n_neurons, target_memory_gb=12.0)
            
            # Run benchmark
            benchmark_results = brain.benchmark(n_steps=5)
            
            print(f"   ✅ Success!")
            print(f"   Time: {benchmark_results['avg_time']:.3f}s")
            print(f"   Steps/sec: {benchmark_results['steps_per_sec']:.1f}")
            print(f"   ms/step: {benchmark_results['ms_per_step']:.2f}ms")
            print(f"   Neurons/sec: {benchmark_results['neurons_per_sec']:,.0f}")
            print(f"   Active/sec: {benchmark_results['active_per_sec']:,.0f}")
            print(f"   Using CuPy: {benchmark_results['using_cupy']}")
            
            results.append({
                'scale': scale_name,
                'neurons': n_neurons,
                'steps_per_sec': benchmark_results['steps_per_sec'],
                'ms_per_step': benchmark_results['ms_per_step'],
                'neurons_per_sec': benchmark_results['neurons_per_sec'],
                'active_per_sec': benchmark_results['active_per_sec'],
                'memory_gb': benchmark_results['memory_gb'],
                'using_cupy': benchmark_results['using_cupy']
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'scale': scale_name,
                'neurons': n_neurons,
                'steps_per_sec': 0,
                'ms_per_step': float('inf'),
                'neurons_per_sec': 0,
                'active_per_sec': 0,
                'memory_gb': 0,
                'using_cupy': False
            })
    
    # Print summary
    print(f"\n📊 HYBRID GPU BRAIN BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Scale':<20} {'Neurons':<15} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<12} {'CuPy':<6}")
    print("-" * 90)
    
    for result in results:
        if result['steps_per_sec'] > 0:
            print(f"{result['scale']:<20} {result['neurons']:<15,} {result['steps_per_sec']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_sec']:<15,.0f} {result['active_per_sec']:<12,.0f} {'✅' if result['using_cupy'] else '❌'}")
        else:
            print(f"{result['scale']:<20} {result['neurons']:<15,} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<12} {'❌'}")
    
    # Find best performance
    successful_results = [r for r in results if r['steps_per_sec'] > 0]
    if successful_results:
        best = max(successful_results, key=lambda x: x['steps_per_sec'])
        print(f"\n🏆 BEST PERFORMANCE: {best['scale']}")
        print(f"   Steps/sec: {best['steps_per_sec']:.1f}")
        print(f"   ms/step: {best['ms_per_step']:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_sec']:,.0f}")
        print(f"   Active/sec: {best['active_per_sec']:,.0f}")
        print(f"   Memory: {best['memory_gb']:.2f} GB")
        print(f"   Using CuPy: {best['using_cupy']}")

if __name__ == "__main__":
    test_hybrid_gpu_brain()

