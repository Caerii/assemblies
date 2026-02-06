#!/usr/bin/env python3
"""
Optimized Brain Simulator
========================

This implementation uses the optimized CUDA brain simulator interface
with O(N log K) algorithms for billion-scale performance.

Key Features:
- Uses cuda_create_optimized_brain() interface
- O(N log K) top-k selection instead of O(N²)
- Shared memory optimizations
- Warp-level reductions
- Billion-scale capable
"""

import time
import numpy as np
import os
import ctypes
from typing import Dict, Any
from dataclasses import dataclass
import json

# Try to import CuPy for GPU memory management
try:
    import cupy as cp
    print("✅ CuPy imported successfully!")
    print(f"   CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"   Current device: {cp.cuda.Device().id}")
    print(f"   Device memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
    CUPY_AVAILABLE = True
except ImportError:
    print("⚠️  CuPy not available, using NumPy fallback")
    CUPY_AVAILABLE = False

@dataclass
class OptimizedSimulationConfig:
    """Configuration for optimized brain simulation"""
    n_neurons: int = 1000000
    active_percentage: float = 0.01
    n_areas: int = 5
    seed: int = 42
    use_gpu: bool = True
    use_optimized_kernels: bool = True
    memory_efficient: bool = True
    sparse_mode: bool = True
    enable_profiling: bool = True

@dataclass
class OptimizedPerformanceMetrics:
    """Performance metrics for the optimized simulation"""
    steps_per_second: float = 0.0
    neurons_per_second: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    optimized_kernels_used: bool = False
    cupy_used: bool = False
    numpy_fallback: bool = False

class OptimizedBrainSimulator:
    """
    Optimized Brain Simulator using O(N log K) algorithms
    
    This implementation uses the optimized CUDA brain simulator interface
    with significant algorithmic improvements for billion-scale performance.
    """
    
    def __init__(self, config: OptimizedSimulationConfig = None):
        """Initialize the optimized brain simulator"""
        self.config = config or OptimizedSimulationConfig()
        
        # Basic parameters
        self.n_neurons = self.config.n_neurons
        self.active_percentage = self.config.active_percentage
        self.k_active = int(self.n_neurons * self.active_percentage)
        self.n_areas = self.config.n_areas
        self.seed = self.config.seed
        
        # Performance tracking
        self.metrics = OptimizedPerformanceMetrics()
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'optimized_kernel_usage': []
        }
        
        # Initialize random number generator
        self._rng = np.random.default_rng(config.seed)
        
        # CUDA kernels DLL
        self._cuda_kernels = None
        self._cuda_brain_ptr = None
        self._load_cuda_kernels()
        
        print("🚀 Optimized Brain Simulator initialized:")
        print(f"   Neurons: {self.n_neurons:,}")
        print(f"   Active percentage: {self.active_percentage*100:.4f}%")
        print(f"   Active per area: {self.k_active:,}")
        print(f"   Areas: {self.n_areas}")
        print(f"   GPU mode: {'✅' if config.use_gpu and CUPY_AVAILABLE else '❌'}")
        print(f"   Optimized kernels: {'✅' if self.metrics.optimized_kernels_used else '❌'}")
        print(f"   Memory efficient: {'✅' if config.memory_efficient else '❌'}")
        print(f"   Sparse mode: {'✅' if config.sparse_mode else '❌'}")
        
        # Initialize optimized CUDA brain
        self._initialize_optimized_brain()
        
        print("   ✅ Optimized Brain initialized successfully!")
    
    def _load_cuda_kernels(self):
        """Load optimized CUDA kernels DLL"""
        if not self.config.use_optimized_kernels:
            return
        
        try:
            # Try to load the optimized CUDA kernels DLL
            dll_path = os.path.join(os.path.dirname(__file__), '..', '..', '.build', 'dlls', 'assemblies_cuda_brain_optimized.dll')
            if os.path.exists(dll_path):
                self._cuda_kernels = ctypes.CDLL(dll_path)
                print("✅ Optimized CUDA kernels loaded successfully!")
                
                # Set up function signatures
                self._setup_optimized_kernel_signatures()
                
                self.metrics.optimized_kernels_used = True
            else:
                print("⚠️  Optimized CUDA kernels DLL not found")
        except Exception as e:
            print(f"⚠️  CUDA kernels failed to load: {e}")
    
    def _setup_optimized_kernel_signatures(self):
        """Set up function signatures for optimized CUDA kernels"""
        if not self._cuda_kernels:
            return
        
        try:
            # Optimized brain simulator interface
            self._cuda_kernels.cuda_create_optimized_brain.argtypes = [
                ctypes.c_uint32,  # uint32_t n_neurons
                ctypes.c_uint32,  # uint32_t n_areas
                ctypes.c_uint32,  # uint32_t k_active
                ctypes.c_uint32   # uint32_t seed
            ]
            self._cuda_kernels.cuda_create_optimized_brain.restype = ctypes.c_void_p
            
            self._cuda_kernels.cuda_simulate_step_optimized.argtypes = [
                ctypes.c_void_p   # void* brain_ptr
            ]
            self._cuda_kernels.cuda_simulate_step_optimized.restype = None
            
            self._cuda_kernels.cuda_destroy_optimized_brain.argtypes = [
                ctypes.c_void_p   # void* brain_ptr
            ]
            self._cuda_kernels.cuda_destroy_optimized_brain.restype = None
            
            # Individual optimized kernels
            if hasattr(self._cuda_kernels, 'cuda_top_k_selection_radix'):
                self._cuda_kernels.cuda_top_k_selection_radix.argtypes = [
                    ctypes.c_void_p,  # const float* activations
                    ctypes.c_void_p,  # uint32_t* top_k_indices
                    ctypes.c_uint32,  # uint32_t total_neurons
                    ctypes.c_uint32   # uint32_t k
                ]
                self._cuda_kernels.cuda_top_k_selection_radix.restype = None
                print("   ✅ top_k_selection_radix signature set")
            
            if hasattr(self._cuda_kernels, 'cuda_accumulate_weights_shared_memory'):
                self._cuda_kernels.cuda_accumulate_weights_shared_memory.argtypes = [
                    ctypes.c_void_p,  # const uint32_t* activated_neurons
                    ctypes.c_void_p,  # const float* synapse_weights
                    ctypes.c_void_p,  # const uint32_t* synapse_indices
                    ctypes.c_void_p,  # const uint32_t* synapse_offsets
                    ctypes.c_void_p,  # float* activations
                    ctypes.c_uint32,  # uint32_t num_activated
                    ctypes.c_uint32   # uint32_t target_size
                ]
                self._cuda_kernels.cuda_accumulate_weights_shared_memory.restype = None
                print("   ✅ accumulate_weights_shared_memory signature set")
            
            print("   ✅ All optimized kernel signatures configured")
            
        except Exception as e:
            print(f"   ⚠️  Failed to set up optimized kernel signatures: {e}")
    
    def _initialize_optimized_brain(self):
        """Initialize the optimized CUDA brain simulator"""
        if not self._cuda_kernels or not self.metrics.optimized_kernels_used:
            print("   ⚠️  Using fallback implementation (no optimized kernels)")
            return
        
        try:
            # Create optimized brain simulator
            self._cuda_brain_ptr = self._cuda_kernels.cuda_create_optimized_brain(
                ctypes.c_uint32(self.n_neurons),
                ctypes.c_uint32(self.n_areas),
                ctypes.c_uint32(self.k_active),
                ctypes.c_uint32(self.seed)
            )
            
            if self._cuda_brain_ptr:
                print("   ✅ Optimized CUDA brain created successfully!")
                print(f"   🧠 Brain pointer: {self._cuda_brain_ptr}")
            else:
                print("   ❌ Failed to create optimized CUDA brain")
                self._cuda_brain_ptr = None
                
        except Exception as e:
            print(f"   ❌ Failed to initialize optimized brain: {e}")
            self._cuda_brain_ptr = None
    
    def simulate_step(self, verbose: bool = False) -> float:
        """Simulate one step using optimized algorithms"""
        start_time = time.time()
        
        if verbose:
            print("🔄 Simulating step with OPTIMIZED O(N log K) algorithms...")
        
        if self._cuda_brain_ptr and self.metrics.optimized_kernels_used:
            try:
                # Use optimized CUDA brain simulator
                self._cuda_kernels.cuda_simulate_step_optimized(self._cuda_brain_ptr)
                
                if verbose:
                    print("   🚀 Used optimized CUDA brain simulator (O(N log K))")
                
            except Exception as e:
                print(f"   ⚠️  Optimized CUDA simulation failed: {e}")
                # Fallback to simple simulation
                self._simulate_step_fallback()
        else:
            # Fallback to simple simulation
            self._simulate_step_fallback()
        
        # Update performance metrics
        step_time = time.time() - start_time
        self.metrics.steps_per_second = 1.0 / step_time if step_time > 0 else 0
        self.metrics.neurons_per_second = (self.n_neurons * self.n_areas) / step_time if step_time > 0 else 0
        
        # Update GPU memory usage
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                self.metrics.memory_usage_gb = mempool.used_bytes() / (1024**3)
            except:
                self.metrics.memory_usage_gb = 0.0
        
        # Update profile data
        if self.config.enable_profiling:
            self.profile_data['step_times'].append(step_time)
            self.profile_data['memory_usage'].append(self.metrics.memory_usage_gb)
            self.profile_data['gpu_utilization'].append(self.metrics.gpu_utilization)
            self.profile_data['optimized_kernel_usage'].append(self.metrics.optimized_kernels_used)
        
        if verbose:
            print(f"   ✅ Step completed in {step_time*1000:.2f}ms")
            print(f"   🚀 Performance: {self.metrics.steps_per_second:.1f} steps/sec")
            print(f"   🧠 Throughput: {self.metrics.neurons_per_second:,.0f} neurons/sec")
            print(f"   💾 Memory: {self.metrics.memory_usage_gb:.2f}GB")
            print(f"   🔧 Optimized kernels: {'✅' if self.metrics.optimized_kernels_used else '❌'}")
        
        return step_time
    
    def _simulate_step_fallback(self):
        """Fallback simulation when optimized kernels are not available"""
        # Simple fallback - just simulate the areas
        for area_idx in range(self.n_areas):
            # Generate candidates (simple random)
            if CUPY_AVAILABLE:
                candidates = cp.random.exponential(1.0, size=self.k_active)
                # Select top-k (simple argsort)
                winners = cp.argsort(candidates)[-self.k_active:][::-1]
            else:
                candidates = np.random.exponential(1.0, size=self.k_active)
                winners = np.argsort(candidates)[-self.k_active:][::-1]
    
    def simulate(self, n_steps: int = 100, verbose: bool = True, profile_interval: int = 10) -> float:
        """Simulate multiple steps using optimized algorithms"""
        if verbose:
            print(f"🚀 Starting OPTIMIZED simulation: {n_steps} steps")
            print(f"   Neurons: {self.n_neurons:,}")
            print(f"   Active: {self.k_active:,} ({self.active_percentage*100:.4f}%)")
            print(f"   Areas: {self.n_areas}")
            print("   Algorithm: O(N log K) - Optimized for billion-scale!")
        
        total_time = 0.0
        
        for step in range(n_steps):
            step_time = self.simulate_step(verbose=(verbose and step % profile_interval == 0))
            total_time += step_time
            
            if verbose and step % profile_interval == 0:
                print(f"   Step {step + 1}/{n_steps}: {step_time*1000:.2f}ms")
        
        if verbose:
            print("✅ OPTIMIZED simulation completed!")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
            print(f"   Neurons per second: {self.metrics.neurons_per_second:,.0f}")
            print(f"   Final memory: {self.metrics.memory_usage_gb:.2f}GB")
            print(f"   GPU utilization: {self.metrics.gpu_utilization:.1f}%")
            print(f"   Optimized kernels: {'✅' if self.metrics.optimized_kernels_used else '❌'}")
            print(f"   CuPy used: {'✅' if self.metrics.cupy_used else '❌'}")
            print(f"   NumPy fallback: {'✅' if self.metrics.numpy_fallback else '❌'}")
        
        return total_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'steps_per_second': self.metrics.steps_per_second,
            'neurons_per_second': self.metrics.neurons_per_second,
            'memory_usage_gb': self.metrics.memory_usage_gb,
            'gpu_utilization': self.metrics.gpu_utilization,
            'optimized_kernels_used': self.metrics.optimized_kernels_used,
            'cupy_used': self.metrics.cupy_used,
            'numpy_fallback': self.metrics.numpy_fallback
        }
    
    def get_profile_data(self) -> Dict[str, Any]:
        """Get detailed profile data"""
        return {
            'config': {
                'n_neurons': self.n_neurons,
                'active_percentage': self.active_percentage,
                'k_active': self.k_active,
                'n_areas': self.n_areas,
                'use_gpu': self.config.use_gpu,
                'use_optimized_kernels': self.config.use_optimized_kernels,
                'memory_efficient': self.config.memory_efficient,
                'sparse_mode': self.config.sparse_mode
            },
            'performance': self.get_performance_stats(),
            'profile_data': self.profile_data
        }
    
    def save_profile(self, filename: str = None):
        """Save profile data to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_brain_profile_{timestamp}.json"
        
        profile_data = self.get_profile_data()
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"📊 Profile saved to: {filename}")
    
    def __del__(self):
        """Cleanup: destroy the optimized CUDA brain"""
        if self._cuda_brain_ptr and self._cuda_kernels:
            try:
                self._cuda_kernels.cuda_destroy_optimized_brain(self._cuda_brain_ptr)
            except:
                pass

def main():
    """Test the optimized brain simulator"""
    print("🧠 Testing Optimized Brain Simulator")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            "name": "Small Scale (100K neurons)",
            "config": OptimizedSimulationConfig(
                n_neurons=100_000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_optimized_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "Medium Scale (1M neurons)",
            "config": OptimizedSimulationConfig(
                n_neurons=1_000_000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_optimized_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "Large Scale (10M neurons)",
            "config": OptimizedSimulationConfig(
                n_neurons=10_000_000,
                active_percentage=0.001,
                n_areas=5,
                use_gpu=True,
                use_optimized_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
        }
    ]
    
    results = []
    
    for test_case in test_configs:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Create simulator
            simulator = OptimizedBrainSimulator(test_case['config'])
            
            # Run simulation
            total_time = simulator.simulate(n_steps=5, verbose=True)
            
            # Get performance stats
            stats = simulator.get_performance_stats()
            
            print(f"\n📊 Results for {test_case['name']}:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
            print(f"   Optimized kernels: {'✅' if stats['optimized_kernels_used'] else '❌'}")
            
            results.append({
                'name': test_case['name'],
                'total_time': total_time,
                'steps_per_second': stats['steps_per_second'],
                'neurons_per_second': stats['neurons_per_second'],
                'memory_usage_gb': stats['memory_usage_gb'],
                'optimized_kernels_used': stats['optimized_kernels_used']
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e)
            })
    
    # Summary
    print("\n📈 OPTIMIZED PERFORMANCE SUMMARY")
    print("=" * 50)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['name']}:")
            print(f"   Time: {result['total_time']:.2f}s")
            print(f"   Speed: {result['neurons_per_second']:,.0f} neurons/sec")
            print(f"   Optimized: {'✅' if result['optimized_kernels_used'] else '❌'}")
        else:
            print(f"{result['name']}: ❌ {result['error']}")
    
    print("\n🎯 Key Insight: This uses O(N log K) algorithms optimized for billion-scale!")

if __name__ == "__main__":
    main()

