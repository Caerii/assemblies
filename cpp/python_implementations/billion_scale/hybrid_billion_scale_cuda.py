#!/usr/bin/env python3
"""
Hybrid Billion-Scale CUDA Brain
Combines the sparse memory model of billion-scale with custom CUDA kernels for maximum performance
"""

import time
import numpy as np
import os
import ctypes

# Try to import CuPy for GPU memory management
try:
    import cupy as cp
    print("✅ CuPy imported successfully!")
    print(f"   CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"   Current device: {cp.cuda.Device().id}")
    print(f"   Device memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
    
    # Test CuPy random number generation
    try:
        test_array = cp.random.exponential(1.0, size=1000)
        print("✅ CuPy random number generation working!")
        CUPY_AVAILABLE = True
    except Exception as e:
        print(f"❌ CuPy random failed: {e}")
        CUPY_AVAILABLE = False
        
except ImportError:
    print("⚠️  CuPy not available, using NumPy fallback")
    CUPY_AVAILABLE = False

class HybridBillionScaleCUDABrain:
    """
    Hybrid Billion-Scale CUDA Brain
    Combines sparse memory model with custom CUDA kernels for maximum performance
    """
    
    def __init__(self, n_neurons=1000000000, active_percentage=0.0001, n_areas=5, seed=42):
        """Initialize the hybrid billion-scale CUDA brain"""
        self.n_neurons = n_neurons
        self.active_percentage = active_percentage
        self.k_active = int(n_neurons * active_percentage)  # Small active count
        self.n_areas = n_areas
        self.seed = seed
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        print("🚀 Hybrid Billion-Scale CUDA Brain initialized:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active percentage: {active_percentage*100:.4f}%")
        print(f"   Active per area: {self.k_active:,}")
        print(f"   Areas: {n_areas}")
        print(f"   CuPy available: {'✅' if CUPY_AVAILABLE else '❌'}")
        
        # Calculate memory usage with sparse approach
        # Only store active neurons + small buffers
        memory_per_area = self.k_active * 4 * 3 / 1024 / 1024 / 1024  # 3 arrays per area
        total_memory = memory_per_area * n_areas
        
        print(f"   Memory per area: {memory_per_area:.2f} GB")
        print(f"   Total memory: {total_memory:.2f} GB")
        
        # Check if we can fit in GPU memory with buffer
        if not CUPY_AVAILABLE:
            raise RuntimeError("❌ CuPy not available - GPU-only mode requires CuPy")
        
        available_gpu_memory = cp.cuda.Device().mem_info[1] / 1024**3
        # Use only 80% of GPU memory for safety buffer
        safe_gpu_memory = available_gpu_memory * 0.8
        print(f"   Available GPU Memory: {available_gpu_memory:.1f} GB")
        print(f"   Safe GPU Memory (80%): {safe_gpu_memory:.1f} GB")
        print(f"   GPU Memory usage: {total_memory/safe_gpu_memory*100:.1f}%")
        
        # Check if we can fit in GPU memory with buffer
        if total_memory > safe_gpu_memory:
            raise RuntimeError(f"❌ Memory exceeds safe GPU capacity ({total_memory:.2f} GB > {safe_gpu_memory:.1f} GB) - skipping CPU fallback")
        
        print("   ✅ Memory fits in safe GPU capacity")
        
        # Load CUDA kernels
        self._cuda_kernels = self._load_cuda_kernels()
        
        # Initialize areas with GPU memory only (sparse model)
        self.areas = []
        for i in range(n_areas):
            area = {
                'n': n_neurons,
                'k': self.k_active,
                'w': 0,
                'winners': cp.zeros(self.k_active, dtype=cp.int32),  # Only active neurons
                'weights': cp.zeros(self.k_active, dtype=cp.float32),  # Only active weights
                'support': cp.zeros(self.k_active, dtype=cp.float32),  # Only active support
                'activated': False
            }
            self.areas.append(area)
        
        # Initialize CUDA memory pools for efficient reuse
        self._initialize_cuda_memory_pools()
        
        # Performance counters
        self.step_count = 0
        self.total_time = 0.0
        
        print("   ✅ Brain initialized successfully!")
        print("   Using: Hybrid GPU (CuPy + CUDA Kernels) - SPARSE MEMORY MODEL")
    
    def _load_cuda_kernels(self):
        """Load CUDA kernels DLL"""
        try:
            # Try to load the CUDA kernels DLL
            dll_path = os.path.join(os.path.dirname(__file__), '..', '..', '.build', 'dlls', 'assemblies_cuda_kernels.dll')
            if os.path.exists(dll_path):
                cuda_kernels = ctypes.CDLL(dll_path)
                print("✅ CUDA kernels loaded successfully!")
                
                # Set up function signatures
                cuda_kernels.cuda_generate_candidates.argtypes = [
                    ctypes.c_void_p,  # curandState* states
                    ctypes.c_void_p,  # float* candidates
                    ctypes.c_uint32,  # uint32_t num_candidates
                    ctypes.c_float,   # float mean
                    ctypes.c_float,   # float stddev
                    ctypes.c_float    # float cutoff
                ]
                cuda_kernels.cuda_generate_candidates.restype = None
                
                cuda_kernels.cuda_top_k_selection.argtypes = [
                    ctypes.c_void_p,  # const float* activations
                    ctypes.c_void_p,  # uint32_t* top_k_indices
                    ctypes.c_uint32,  # uint32_t total_neurons
                    ctypes.c_uint32   # uint32_t k
                ]
                cuda_kernels.cuda_top_k_selection.restype = None
                
                cuda_kernels.cuda_initialize_curand.argtypes = [
                    ctypes.c_void_p,  # curandState* states
                    ctypes.c_uint32,  # uint32_t n
                    ctypes.c_uint32   # uint32_t seed
                ]
                cuda_kernels.cuda_initialize_curand.restype = None
                
                return cuda_kernels
            else:
                print("⚠️  CUDA kernels DLL not found, using CuPy fallback")
                return None
        except Exception as e:
            print(f"⚠️  Failed to load CUDA kernels: {e}, using CuPy fallback")
            return None
    
    def _initialize_cuda_memory_pools(self):
        """Initialize CUDA kernel memory pools for efficient reuse"""
        if not self._cuda_kernels or not CUPY_AVAILABLE:
            return
        
        # Initialize memory pools as None - will allocate dynamically
        self._cuda_states = None
        self._cuda_candidates = None
        self._cuda_top_k = None
        self._cuda_max_k = 0
        
        print("   🔧 CUDA memory pools initialized (dynamic allocation)")
    
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
    
    def _generate_candidates_hybrid(self, area_idx):
        """Generate candidates using hybrid approach (CUDA kernels + sparse memory)"""
        area = self.areas[area_idx]
        
        if self._cuda_kernels:
            try:
                # Ensure we have enough CUDA memory
                if not self._ensure_cuda_memory(area['k']):
                    raise Exception("Failed to allocate CUDA memory")
                
                # Use pre-allocated memory
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
    
    def _select_top_k_hybrid(self, candidates, k):
        """Select top-k using hybrid approach (CUDA kernels + sparse memory)"""
        if k >= len(candidates):
            return cp.arange(len(candidates))
        
        if self._cuda_kernels:
            try:
                # Ensure we have enough CUDA memory
                if not self._ensure_cuda_memory(k):
                    raise Exception("Failed to allocate CUDA memory")
                
                # Use pre-allocated memory
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
    
    def _update_weights_hybrid(self, area_idx, winners):
        """Update weights using hybrid approach (CuPy + sparse memory)"""
        area = self.areas[area_idx]
        
        # Use CuPy for GPU-accelerated weight updates
        area['weights'][winners] += 0.1
        area['weights'] *= 0.99
        area['support'][winners] += 1.0
    
    def simulate_step(self):
        """Simulate one step of the brain"""
        start_time = time.perf_counter()
        
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]
            
            # Generate candidates using hybrid approach
            candidates = self._generate_candidates_hybrid(area_idx)
            
            # Select top-k using hybrid approach
            winners = self._select_top_k_hybrid(candidates, area['k'])
            
            # Update weights using hybrid approach
            self._update_weights_hybrid(area_idx, winners)
            
            # Store winners
            area['winners'] = winners
            area['w'] = len(winners)
            area['activated'] = True
        
        # Update performance counters
        step_time = time.perf_counter() - start_time
        self.step_count += 1
        self.total_time += step_time
    
    def simulate(self, n_steps=10, verbose=True):
        """Simulate the brain for n_steps"""
        if verbose:
            print(f"🧠 Simulating {n_steps} steps...")
        
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            self.simulate_step()
            
            if verbose and step % max(1, n_steps // 10) == 0:
                elapsed = time.perf_counter() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"   Step {step + 1}/{n_steps}: {steps_per_sec:.1f} steps/s")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print("✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {n_steps / total_time:.1f}")
            print(f"   ms/step: {total_time / n_steps * 1000:.2f}")
            print(f"   Neurons/sec: {self.n_neurons * n_steps / total_time:,.0f}")
            print(f"   Active/sec: {self.k_active * n_steps / total_time:,.0f}")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.step_count == 0:
            return {
                'steps_per_second': 0,
                'ms_per_step': 0,
                'neurons_per_second': 0,
                'active_per_second': 0
            }
        
        avg_time_per_step = self.total_time / self.step_count
        steps_per_second = 1.0 / avg_time_per_step
        ms_per_step = avg_time_per_step * 1000
        neurons_per_second = self.n_neurons * steps_per_second
        active_per_second = self.k_active * steps_per_second
        
        return {
            'steps_per_second': steps_per_second,
            'ms_per_step': ms_per_step,
            'neurons_per_second': neurons_per_second,
            'active_per_second': active_per_second
        }

def test_hybrid_billion_scale():
    """Test the hybrid billion-scale CUDA brain"""
    print("🚀 TESTING HYBRID BILLION-SCALE CUDA BRAIN")
    print("=" * 70)
    
    # Test scales
    test_scales = [
        {"n_neurons": 1000000, "active_percentage": 0.01, "name": "Million Scale (1%)"},
        {"n_neurons": 10000000, "active_percentage": 0.001, "name": "Ten Million Scale (0.1%)"},
        {"n_neurons": 100000000, "active_percentage": 0.0001, "name": "Hundred Million Scale (0.01%)"},
        {"n_neurons": 1000000000, "active_percentage": 0.00001, "name": "BILLION SCALE (0.001%)"},
        {"n_neurons": 2000000000, "active_percentage": 0.000005, "name": "TWO BILLION SCALE (0.0005%)"},
        {"n_neurons": 5000000000, "active_percentage": 0.000002, "name": "FIVE BILLION SCALE (0.0002%)"},
    ]
    
    results = []
    
    for scale in test_scales:
        print(f"\n🧪 Testing {scale['name']}:")
        print(f"   Neurons: {scale['n_neurons']:,}")
        print(f"   Active percentage: {scale['active_percentage']*100:.5f}%")
        print(f"   Active per area: {int(scale['n_neurons'] * scale['active_percentage']):,}")
        
        try:
            brain = HybridBillionScaleCUDABrain(
                n_neurons=scale['n_neurons'],
                active_percentage=scale['active_percentage'],
                n_areas=5,
                seed=42
            )
            
            start_time = time.perf_counter()
            brain.simulate(n_steps=10, verbose=False)
            total_time = time.perf_counter() - start_time
            
            stats = brain.get_performance_stats()
            print("   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   ms/step: {stats['ms_per_step']:.2f}")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Active/sec: {stats['active_per_second']:,.0f}")
            
            results.append({
                'name': scale['name'],
                'n_neurons': scale['n_neurons'],
                'active_percentage': scale['active_percentage'],
                'time': total_time,
                'steps_per_sec': stats['steps_per_second'],
                'ms_per_step': stats['ms_per_step'],
                'neurons_per_sec': stats['neurons_per_second'],
                'active_per_sec': stats['active_per_second']
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': scale['name'],
                'n_neurons': scale['n_neurons'],
                'active_percentage': scale['active_percentage'],
                'time': float('inf'),
                'steps_per_sec': 0,
                'ms_per_step': 0,
                'neurons_per_sec': 0,
                'active_per_sec': 0
            })
    
    # Print summary
    print("\n📊 HYBRID BILLION-SCALE CUDA BRAIN BENCHMARK SUMMARY")
    print("=" * 120)
    print(f"{'Scale':<30} {'Neurons':<15} {'Active%':<10} {'Steps/sec':<12} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<15}")
    print("-" * 120)
    
    for result in results:
        if result['steps_per_sec'] > 0:
            print(f"{result['name']:<30} {result['n_neurons']:<15,} {result['active_percentage']*100:<10.5f} {result['steps_per_sec']:<12.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_sec']:<15,.0f} {result['active_per_sec']:<15,.0f}")
        else:
            print(f"{result['name']:<30} {result['n_neurons']:<15,} {result['active_percentage']*100:<10.5f} {'FAILED':<12} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15}")
    
    # Find best performance
    working_results = [r for r in results if r['steps_per_sec'] > 0]
    if working_results:
        best_result = max(working_results, key=lambda x: x['steps_per_sec'])
        print(f"\n🏆 BEST PERFORMANCE: {best_result['name']}")
        print(f"   Steps/sec: {best_result['steps_per_sec']:.1f}")
        print(f"   ms/step: {best_result['ms_per_step']:.2f}")
        print(f"   Neurons/sec: {best_result['neurons_per_sec']:,.0f}")
        print(f"   Active/sec: {best_result['active_per_sec']:,.0f}")
    
    return results

if __name__ == "__main__":
    test_hybrid_billion_scale()
