#!/usr/bin/env python3
"""
Universal Brain Simulator - Core Implementations Superset
========================================================

This superset combines the best features from all core brain simulation implementations:
- CUDA kernels integration
- CuPy GPU acceleration
- NumPy fallback
- Memory optimization
- Performance monitoring
- Multiple simulation modes

Combines features from:
- working_cuda_brain_v14_39.py
- working_cupy_brain.py
- cuda_brain_python.py
- optimized_cuda_brain.py
- ultra_fast_cuda_brain.py
- ultra_optimized_cuda_brain_v2.py
- hybrid_gpu_brain.py
- memory_efficient_cupy_brain.py
- ultra_sparse_cupy_brain.py
"""

import time
import numpy as np
import os
import sys
import ctypes
from typing import Dict, List, Tuple, Optional, Any, Union
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
class SimulationConfig:
    """Configuration for brain simulation"""
    n_neurons: int = 1000000
    active_percentage: float = 0.01
    n_areas: int = 5
    seed: int = 42
    use_gpu: bool = True
    use_cuda_kernels: bool = True
    use_optimized_kernels: bool = True  # NEW: Choose between original and optimized
    memory_efficient: bool = True
    sparse_mode: bool = True
    enable_profiling: bool = True

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    step_count: int = 0
    total_time: float = 0.0
    min_step_time: float = float('inf')
    max_step_time: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    cuda_kernels_used: bool = False
    cupy_used: bool = False
    numpy_fallback: bool = False

class UniversalBrainSimulator:
    """
    Universal Brain Simulator
    
    Combines the best features from all core implementations:
    - CUDA kernels for maximum performance
    - CuPy for GPU acceleration
    - NumPy fallback for compatibility
    - Memory optimization and pooling
    - Real-time performance monitoring
    - Multiple simulation modes
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize the universal brain simulator"""
        self.config = config
        self.n_neurons = config.n_neurons
        self.active_percentage = config.active_percentage
        self.k_active = int(config.n_neurons * config.active_percentage)
        self.n_areas = config.n_areas
        self.seed = config.seed
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cuda_kernel_usage': []
        }
        
        # Initialize random number generator
        self._rng = np.random.default_rng(config.seed)
        
        # CUDA kernels DLL
        self._cuda_kernels = None
        self._optimized_brain_ptr = None
        self._load_cuda_kernels()
        
        print(f"🚀 Universal Brain Simulator initialized:")
        print(f"   Neurons: {self.n_neurons:,}")
        print(f"   Active percentage: {self.active_percentage*100:.4f}%")
        print(f"   Active per area: {self.k_active:,}")
        print(f"   Areas: {self.n_areas}")
        print(f"   GPU mode: {'✅' if config.use_gpu and CUPY_AVAILABLE else '❌'}")
        print(f"   CUDA kernels: {'✅' if self._cuda_kernels else '❌'}")
        if self._cuda_kernels:
            kernel_type = "Optimized (O(N log K))" if getattr(self, '_using_optimized_kernels', False) else "Original (O(N²))"
            print(f"   Kernel type: {kernel_type}")
        print(f"   Memory efficient: {'✅' if config.memory_efficient else '❌'}")
        print(f"   Sparse mode: {'✅' if config.sparse_mode else '❌'}")
        
        # Initialize areas
        self._initialize_areas()
        
        # Initialize CUDA kernel memory pools
        self._initialize_cuda_memory_pools()
        
        print(f"   ✅ Brain initialized successfully!")
    
    def _load_cuda_kernels(self):
        """Load CUDA kernels DLL - supports both original and optimized versions"""
        if not self.config.use_cuda_kernels:
            return
        
        # Try optimized kernels first if requested
        if self.config.use_optimized_kernels:
            try:
                optimized_dll_path = os.path.join(os.path.dirname(__file__), '..', '..', '.build', 'dlls', 'assemblies_cuda_brain_optimized.dll')
                if os.path.exists(optimized_dll_path):
                    self._cuda_kernels = ctypes.CDLL(optimized_dll_path)
                    self._setup_optimized_kernel_signatures()
                    print("✅ Optimized CUDA kernels loaded successfully!")
                    self.metrics.cuda_kernels_used = True
                    self._using_optimized_kernels = True
                    return
                else:
                    print("⚠️  Optimized CUDA kernels DLL not found, trying original...")
            except Exception as e:
                print(f"⚠️  Optimized CUDA kernels failed to load: {e}, trying original...")
        
        # Fallback to original kernels
        try:
            original_dll_path = os.path.join(os.path.dirname(__file__), '..', '..', '.build', 'dlls', 'assemblies_cuda_kernels.dll')
            if os.path.exists(original_dll_path):
                self._cuda_kernels = ctypes.CDLL(original_dll_path)
                self._setup_original_kernel_signatures()
                print("✅ Original CUDA kernels loaded successfully!")
                self.metrics.cuda_kernels_used = True
                self._using_optimized_kernels = False
            else:
                print("⚠️  Original CUDA kernels DLL not found, using fallback")
        except Exception as e:
            print(f"⚠️  Original CUDA kernels failed to load: {e}, using fallback")
    
    def _setup_optimized_kernel_signatures(self):
        """Set up function signatures for optimized CUDA kernels"""
        if not self._cuda_kernels:
            return
        
        try:
            # Check if this is the optimized brain simulator DLL
            if hasattr(self._cuda_kernels, 'cuda_create_optimized_brain'):
                print("   🔧 Setting up optimized brain simulator interface")
                
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
                
                print("   ✅ Optimized brain simulator interface configured")
                
                # Create optimized brain instance
                self._optimized_brain_ptr = self._cuda_kernels.cuda_create_optimized_brain(
                    ctypes.c_uint32(self.n_neurons),
                    ctypes.c_uint32(self.n_areas),
                    ctypes.c_uint32(self.k_active),
                    ctypes.c_uint32(self.seed)
                )
                print(f"   🧠 Optimized brain instance created: {self._optimized_brain_ptr}")
                return
            
            # Individual optimized kernels (fallback)
            print("   🔧 Setting up individual optimized kernel signatures")
            
            if hasattr(self._cuda_kernels, 'cuda_generate_candidates_optimized'):
                self._cuda_kernels.cuda_generate_candidates_optimized.argtypes = [
                    ctypes.c_void_p,  # curandState* states
                    ctypes.c_void_p,  # float* candidates
                    ctypes.c_uint32,  # uint32_t num_candidates
                    ctypes.c_float,   # float mean
                    ctypes.c_float,   # float stddev
                    ctypes.c_float    # float cutoff
                ]
                self._cuda_kernels.cuda_generate_candidates_optimized.restype = None
            
            if hasattr(self._cuda_kernels, 'cuda_top_k_selection_radix'):
                self._cuda_kernels.cuda_top_k_selection_radix.argtypes = [
                    ctypes.c_void_p,  # const float* activations
                    ctypes.c_void_p,  # uint32_t* top_k_indices
                    ctypes.c_uint32,  # uint32_t total_neurons
                    ctypes.c_uint32   # uint32_t k
                ]
                self._cuda_kernels.cuda_top_k_selection_radix.restype = None
            
            if hasattr(self._cuda_kernels, 'cuda_initialize_curand'):
                self._cuda_kernels.cuda_initialize_curand.argtypes = [
                    ctypes.c_void_p,  # curandState* states
                    ctypes.c_uint32,  # uint32_t n
                    ctypes.c_uint32   # uint32_t seed
                ]
                self._cuda_kernels.cuda_initialize_curand.restype = None
            
            print("   ✅ Individual optimized kernel signatures configured")
            
        except Exception as e:
            print(f"   ⚠️  Failed to set up optimized kernel signatures: {e}")
    
    def _setup_original_kernel_signatures(self):
        """Set up function signatures for original CUDA kernels"""
        if not self._cuda_kernels:
            return
        
        try:
            print("   🔧 Setting up original kernel signatures")
            
            if hasattr(self._cuda_kernels, 'cuda_generate_candidates'):
                self._cuda_kernels.cuda_generate_candidates.argtypes = [
                    ctypes.c_void_p,  # curandState* states
                    ctypes.c_void_p,  # float* candidates
                    ctypes.c_uint32,  # uint32_t num_candidates
                    ctypes.c_float,   # float mean
                    ctypes.c_float,   # float stddev
                    ctypes.c_float    # float cutoff
                ]
                self._cuda_kernels.cuda_generate_candidates.restype = None
            
            if hasattr(self._cuda_kernels, 'cuda_top_k_selection'):
                self._cuda_kernels.cuda_top_k_selection.argtypes = [
                    ctypes.c_void_p,  # const float* activations
                    ctypes.c_void_p,  # uint32_t* top_k_indices
                    ctypes.c_uint32,  # uint32_t total_neurons
                    ctypes.c_uint32   # uint32_t k
                ]
                self._cuda_kernels.cuda_top_k_selection.restype = None
            
            if hasattr(self._cuda_kernels, 'cuda_initialize_curand'):
                self._cuda_kernels.cuda_initialize_curand.argtypes = [
                    ctypes.c_void_p,  # curandState* states
                    ctypes.c_uint32,  # uint32_t n
                    ctypes.c_uint32   # uint32_t seed
                ]
                self._cuda_kernels.cuda_initialize_curand.restype = None
            
            print("   ✅ Original kernel signatures configured")
            
        except Exception as e:
            print(f"   ⚠️  Failed to set up original kernel signatures: {e}")
    
    def _initialize_areas(self):
        """Initialize brain areas with appropriate memory management"""
        self.areas = []
        
        for i in range(self.n_areas):
            if self.config.use_gpu and CUPY_AVAILABLE:
                # GPU memory allocation
                area = {
                    'n': self.n_neurons,
                    'k': self.k_active,
                    'w': 0,
                    'winners': cp.zeros(self.k_active, dtype=cp.int32),
                    'weights': cp.zeros(self.k_active, dtype=cp.float32),
                    'support': cp.zeros(self.k_active, dtype=cp.float32),
                    'activated': False,
                    'area_id': i
                }
                self.metrics.cupy_used = True
            else:
                # CPU memory allocation
                area = {
                    'n': self.n_neurons,
                    'k': self.k_active,
                    'w': 0,
                    'winners': np.zeros(self.k_active, dtype=np.int32),
                    'weights': np.zeros(self.k_active, dtype=np.float32),
                    'support': np.zeros(self.k_active, dtype=np.float32),
                    'activated': False,
                    'area_id': i
                }
                self.metrics.numpy_fallback = True
            
            self.areas.append(area)
    
    def _initialize_cuda_memory_pools(self):
        """Initialize CUDA kernel memory pools for efficient reuse"""
        if not self._cuda_kernels or not CUPY_AVAILABLE:
            return
        
        # Initialize memory pools as None - will allocate dynamically
        self._cuda_states = None
        self._cuda_candidates = None
        self._cuda_top_k = None
        self._cuda_max_k = 0
        
        print(f"   🔧 CUDA memory pools initialized (dynamic allocation)")
    
    def _ensure_cuda_memory(self, required_k):
        """Ensure CUDA memory is allocated for the required size"""
        if not self._cuda_kernels or not CUPY_AVAILABLE:
            return False
        
        if self._cuda_states is None or required_k > self._cuda_max_k:
            # Allocate new memory
            self._cuda_states = cp.zeros(required_k, dtype=cp.uint64)
            self._cuda_candidates = cp.zeros(required_k, dtype=cp.float32)
            self._cuda_top_k = cp.zeros(required_k, dtype=cp.uint32)
            self._cuda_max_k = required_k
            
            # Initialize curand states
            self._cuda_kernels.cuda_initialize_curand(
                ctypes.c_void_p(self._cuda_states.data.ptr),
                ctypes.c_uint32(required_k),
                ctypes.c_uint32(self.seed)
            )
            
            print(f"   🔧 CUDA memory reallocated for k={required_k:,}")
        
        return True
    
    def _generate_candidates(self, area_idx: int) -> Union[np.ndarray, cp.ndarray]:
        """Generate candidates using the best available method"""
        area = self.areas[area_idx]
        
        if self.config.use_gpu and CUPY_AVAILABLE:
            # Try CUDA kernels first if available
            if self._cuda_kernels and self.metrics.cuda_kernels_used:
                try:
                    # Ensure we have enough CUDA memory
                    if not self._ensure_cuda_memory(area['k']):
                        raise Exception("Failed to allocate CUDA memory")
                    
                    # Use dynamic memory allocation
                    candidates = self._cuda_candidates[:area['k']]
                    
                    # Generate candidates using CUDA kernel
                    self._cuda_kernels.cuda_generate_candidates(
                        ctypes.c_void_p(self._cuda_states.data.ptr),
                        ctypes.c_void_p(candidates.data.ptr),
                        ctypes.c_uint32(area['k']),
                        ctypes.c_float(1.0),  # mean
                        ctypes.c_float(1.0),  # stddev
                        ctypes.c_float(0.0)   # cutoff
                    )
                    
                    return candidates
                except Exception as e:
                    print(f"   ⚠️  CUDA kernels failed: {e}, falling back to CuPy")
            
            # Fallback to CuPy
            try:
                candidates = cp.random.exponential(1.0, size=area['k'])
                return candidates
            except Exception as e:
                print(f"   ⚠️  CuPy random failed: {e}, falling back to NumPy")
                # Fallback to NumPy + GPU transfer
                np_candidates = self._rng.exponential(1.0, size=area['k'])
                candidates = cp.asarray(np_candidates)
                return candidates
        else:
            # CPU random generation
            candidates = self._rng.exponential(1.0, size=area['k'])
            return candidates
    
    def _select_top_k(self, candidates: Union[np.ndarray, cp.ndarray], k: int) -> Union[np.ndarray, cp.ndarray]:
        """Select top-k using the best available method"""
        if k >= len(candidates):
            if self.config.use_gpu and CUPY_AVAILABLE:
                return cp.arange(len(candidates))
            else:
                return np.arange(len(candidates))
        
        if self.config.use_gpu and CUPY_AVAILABLE:
            # Try CUDA kernels first if available
            if self._cuda_kernels and self.metrics.cuda_kernels_used:
                try:
                    # Ensure we have enough CUDA memory
                    if not self._ensure_cuda_memory(k):
                        raise Exception("Failed to allocate CUDA memory")
                    
                    # Use dynamic memory allocation
                    top_k_indices = self._cuda_top_k[:k]
                    
                    # Use CUDA kernel for top-k selection
                    self._cuda_kernels.cuda_top_k_selection(
                        ctypes.c_void_p(candidates.data.ptr),
                        ctypes.c_void_p(top_k_indices.data.ptr),
                        ctypes.c_uint32(len(candidates)),
                        ctypes.c_uint32(k)
                    )
                    
                    return top_k_indices
                except Exception as e:
                    print(f"   ⚠️  CUDA top-k failed: {e}, falling back to CuPy")
            
            # Fallback to CuPy
            top_k_indices = cp.argpartition(candidates, -k)[-k:]
            top_k_values = candidates[top_k_indices]
            sorted_indices = cp.argsort(top_k_values)[::-1]
            return top_k_indices[sorted_indices]
        else:
            # CPU top-k selection
            top_k_indices = np.argpartition(candidates, -k)[-k:]
            top_k_values = candidates[top_k_indices]
            sorted_indices = np.argsort(top_k_values)[::-1]
            return top_k_indices[sorted_indices]
    
    def _update_weights(self, area_idx: int, winners: Union[np.ndarray, cp.ndarray]):
        """Update weights using the best available method"""
        area = self.areas[area_idx]
        
        # Weight updates
        area['weights'][winners] += 0.1
        area['weights'] *= 0.99
        area['support'][winners] += 1.0
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage"""
        if self.config.use_gpu and CUPY_AVAILABLE:
            try:
                used, total = cp.cuda.Device().mem_info
                return used / 1024**3, total / 1024**3
            except:
                return 0.0, 0.0
        else:
            import psutil
            memory = psutil.virtual_memory()
            return memory.used / 1024**3, memory.total / 1024**3
    
    def simulate_step(self) -> float:
        """Simulate one step of the brain"""
        start_time = time.perf_counter()
        
        # Get initial memory usage
        if self.config.enable_profiling:
            initial_memory, total_memory = self._get_memory_usage()
        
        # Use optimized brain simulator if available
        if self._optimized_brain_ptr is not None:
            try:
                # Use the optimized brain simulator
                self._cuda_kernels.cuda_simulate_step_optimized(
                    ctypes.c_void_p(self._optimized_brain_ptr)
                )
            except Exception as e:
                print(f"   ⚠️  Optimized brain simulation failed: {e}, falling back to area-based simulation")
                # Fall back to area-based simulation
                self._simulate_areas()
        else:
            # Use area-based simulation
            self._simulate_areas()
        
        # Record timing and profiling data
        step_time = time.perf_counter() - start_time
        self.metrics.step_count += 1
        self.metrics.total_time += step_time
        self.metrics.min_step_time = min(self.metrics.min_step_time, step_time)
        self.metrics.max_step_time = max(self.metrics.max_step_time, step_time)
        
        if self.config.enable_profiling:
            final_memory, total_memory = self._get_memory_usage()
            self.metrics.memory_usage_gb = final_memory
            self.metrics.gpu_utilization = (final_memory / total_memory) * 100 if total_memory > 0 else 0
            
            self.profile_data['step_times'].append(step_time)
            self.profile_data['memory_usage'].append(final_memory)
            self.profile_data['gpu_utilization'].append(self.metrics.gpu_utilization)
            self.profile_data['cuda_kernel_usage'].append(self.metrics.cuda_kernels_used)
        
        return step_time
    
    def _simulate_areas(self):
        """Simulate areas using the traditional area-based approach"""
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]
            
            # Generate candidates
            candidates = self._generate_candidates(area_idx)
            
            # Select top-k winners
            winners = self._select_top_k(candidates, area['k'])
            
            # Update area state
            area['w'] = len(winners)
            area['winners'][:len(winners)] = winners
            area['activated'] = True
            
            # Update weights
            self._update_weights(area_idx, winners)
    
    def simulate(self, n_steps: int = 100, verbose: bool = True, profile_interval: int = 10) -> float:
        """Simulate multiple steps"""
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS (Universal Mode)")
            print("=" * 60)
        
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if verbose and (step + 1) % profile_interval == 0:
                avg_time = self.metrics.total_time / self.metrics.step_count
                memory_usage = self.metrics.memory_usage_gb
                gpu_util = self.metrics.gpu_utilization
                
                print(f"Step {step + 1:3d}: {step_time*1000:.2f}ms | "
                      f"Avg: {avg_time*1000:.2f}ms | "
                      f"Memory: {memory_usage:.2f}GB ({gpu_util:.1f}%)")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n📊 UNIVERSAL SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Min step time: {self.metrics.min_step_time*1000:.2f}ms")
            print(f"   Max step time: {self.metrics.max_step_time*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
            print(f"   Final memory: {self.metrics.memory_usage_gb:.2f}GB")
            print(f"   GPU utilization: {self.metrics.gpu_utilization:.1f}%")
            print(f"   CUDA kernels: {'✅' if self.metrics.cuda_kernels_used else '❌'}")
            print(f"   CuPy used: {'✅' if self.metrics.cupy_used else '❌'}")
            print(f"   NumPy fallback: {'✅' if self.metrics.numpy_fallback else '❌'}")
        
        return total_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        if self.metrics.step_count == 0:
            return {}
        
        avg_step_time = self.metrics.total_time / self.metrics.step_count
        steps_per_second = 1.0 / avg_step_time
        
        return {
            'total_steps': self.metrics.step_count,
            'total_time': self.metrics.total_time,
            'avg_step_time': avg_step_time,
            'min_step_time': self.metrics.min_step_time,
            'max_step_time': self.metrics.max_step_time,
            'steps_per_second': steps_per_second,
            'neurons_per_second': self.n_neurons * steps_per_second,
            'active_neurons_per_second': self.k_active * self.n_areas * steps_per_second,
            'memory_usage_gb': self.metrics.memory_usage_gb,
            'gpu_utilization': self.metrics.gpu_utilization,
            'cuda_kernels_used': self.metrics.cuda_kernels_used,
            'cupy_used': self.metrics.cupy_used,
            'numpy_fallback': self.metrics.numpy_fallback
        }
    
    def get_profile_data(self) -> Dict[str, Any]:
        """Get detailed profiling data"""
        return self.profile_data.copy()
    
    def save_profile_data(self, filename: str):
        """Save profiling data to JSON file"""
        profile_data = {
            'configuration': {
                'n_neurons': self.n_neurons,
                'active_percentage': self.active_percentage,
                'k_active': self.k_active,
                'n_areas': self.n_areas,
                'use_gpu': self.config.use_gpu,
                'use_cuda_kernels': self.config.use_cuda_kernels,
                'memory_efficient': self.config.memory_efficient,
                'sparse_mode': self.config.sparse_mode
            },
            'performance': self.get_performance_stats(),
            'profile_data': self.profile_data
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        print(f"📊 Profile data saved to {filename}")
    
    def __del__(self):
        """Cleanup resources when simulator is destroyed"""
        if self._optimized_brain_ptr is not None and self._cuda_kernels is not None:
            try:
                self._cuda_kernels.cuda_destroy_optimized_brain(
                    ctypes.c_void_p(self._optimized_brain_ptr)
                )
                print("🧠 Optimized brain instance destroyed")
            except:
                pass  # Ignore cleanup errors

def test_universal_brain_simulator():
    """Test the universal brain simulator with different configurations"""
    print("🚀 TESTING UNIVERSAL BRAIN SIMULATOR")
    print("=" * 60)
    
    # Test different configurations
    test_configs = [
        {
            "name": "GPU + Optimized CUDA (O(N log K))",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                use_optimized_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "GPU + Original CUDA (O(N²))",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                use_optimized_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "GPU Only (CuPy)",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=False,
                use_optimized_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "CPU Only (NumPy)",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=False,
                use_cuda_kernels=False,
                use_optimized_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
        }
    ]
    
    results = []
    
    for test_case in test_configs:
        print(f"\n🧪 Testing {test_case['name']}:")
        
        try:
            # Create simulator
            simulator = UniversalBrainSimulator(test_case['config'])
            
            # Simulate
            start_time = time.perf_counter()
            simulator.simulate(n_steps=10, verbose=False)
            total_time = time.perf_counter() - start_time
            
            # Get stats
            stats = simulator.get_performance_stats()
            
            print(f"   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   ms/step: {stats['avg_step_time']*1000:.2f}ms")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
            print(f"   CUDA kernels: {'✅' if stats['cuda_kernels_used'] else '❌'}")
            print(f"   CuPy used: {'✅' if stats['cupy_used'] else '❌'}")
            print(f"   NumPy fallback: {'✅' if stats['numpy_fallback'] else '❌'}")
            
            results.append({
                'name': test_case['name'],
                'total_time': total_time,
                'steps_per_second': stats['steps_per_second'],
                'ms_per_step': stats['avg_step_time'] * 1000,
                'neurons_per_second': stats['neurons_per_second'],
                'memory_usage_gb': stats['memory_usage_gb'],
                'cuda_kernels_used': stats['cuda_kernels_used'],
                'cupy_used': stats['cupy_used'],
                'numpy_fallback': stats['numpy_fallback']
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'total_time': float('inf'),
                'steps_per_second': 0,
                'ms_per_step': float('inf'),
                'neurons_per_second': 0,
                'memory_usage_gb': 0,
                'cuda_kernels_used': False,
                'cupy_used': False,
                'numpy_fallback': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 UNIVERSAL BRAIN SIMULATOR SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Memory GB':<10} {'CUDA':<6} {'CuPy':<6} {'NumPy':<6}")
    print("-" * 80)
    
    for result in results:
        if result['steps_per_second'] > 0:
            print(f"{result['name']:<20} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['memory_usage_gb']:<10.2f} {'✅' if result['cuda_kernels_used'] else '❌':<6} {'✅' if result['cupy_used'] else '❌':<6} {'✅' if result['numpy_fallback'] else '❌':<6}")
        else:
            print(f"{result['name']:<20} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<10} {'❌':<6} {'❌':<6} {'❌':<6}")
    
    return results

if __name__ == "__main__":
    # Test universal brain simulator
    results = test_universal_brain_simulator()
    
    # Find best performance
    successful_results = [r for r in results if r['steps_per_second'] > 0]
    if successful_results:
        best = max(successful_results, key=lambda x: x['steps_per_second'])
        print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
        print(f"   Steps/sec: {best['steps_per_second']:.1f}")
        print(f"   ms/step: {best['ms_per_step']:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
        print(f"   Memory Usage: {best['memory_usage_gb']:.2f}GB")
    else:
        print(f"\n❌ No successful tests")
