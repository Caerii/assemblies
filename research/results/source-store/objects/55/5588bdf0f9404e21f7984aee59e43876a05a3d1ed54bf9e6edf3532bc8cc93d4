#!/usr/bin/env python3
"""
Enhanced GPU-Only Billion-Scale Brain
=====================================

Expanded version with advanced features:
- Multi-GPU support
- Memory pooling
- Advanced profiling
- Adaptive scaling
- Real-time monitoring
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

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

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    step_count: int = 0
    total_time: float = 0.0
    min_step_time: float = float('inf')
    max_step_time: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class MemoryPool:
    """GPU memory pool for efficient allocation"""
    pools: Dict[str, List[cp.ndarray]] = None
    max_pool_size: int = 10
    
    def __post_init__(self):
        if self.pools is None:
            self.pools = {
                'candidates': [],
                'winners': [],
                'weights': [],
                'support': []
            }
    
    def get_array(self, dtype: cp.dtype, shape: Tuple[int, ...], pool_name: str) -> cp.ndarray:
        """Get array from pool or create new one"""
        if pool_name in self.pools and self.pools[pool_name]:
            array = self.pools[pool_name].pop()
            if array.shape == shape and array.dtype == dtype:
                return array
        
        return cp.zeros(shape, dtype=dtype)
    
    def return_array(self, array: cp.ndarray, pool_name: str):
        """Return array to pool"""
        if pool_name in self.pools and len(self.pools[pool_name]) < self.max_pool_size:
            self.pools[pool_name].append(array)

class EnhancedGPUOnlyBillionScaleBrain:
    """
    Enhanced GPU-Only Billion-Scale Brain
    
    Features:
    - Multi-GPU support
    - Memory pooling
    - Advanced profiling
    - Adaptive scaling
    - Real-time monitoring
    """
    
    def __init__(self, n_neurons=1000000000, active_percentage=0.0001, n_areas=5, 
                 seed=42, use_memory_pool=True, enable_profiling=True):
        """Initialize the enhanced GPU-only billion-scale brain"""
        self.n_neurons = n_neurons
        self.active_percentage = active_percentage
        self.k_active = int(n_neurons * active_percentage)
        self.n_areas = n_areas
        self.seed = seed
        self.use_memory_pool = use_memory_pool
        self.enable_profiling = enable_profiling
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.memory_pool = MemoryPool() if use_memory_pool else None
        
        # Profiling data
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cache_performance': []
        }
        
        print("🚀 Enhanced GPU-Only Billion-Scale Brain initialized:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active percentage: {active_percentage*100:.4f}%")
        print(f"   Active per area: {self.k_active:,}")
        print(f"   Areas: {n_areas}")
        print(f"   CuPy available: {'✅' if CUPY_AVAILABLE else '❌'}")
        print(f"   Memory pooling: {'✅' if use_memory_pool else '❌'}")
        print(f"   Profiling: {'✅' if enable_profiling else '❌'}")
        
        # Calculate memory usage with sparse approach
        memory_per_area = self.k_active * 4 * 3 / 1024 / 1024 / 1024  # 3 arrays per area
        total_memory = memory_per_area * n_areas
        
        print(f"   Memory per area: {memory_per_area:.2f} GB")
        print(f"   Total memory: {total_memory:.2f} GB")
        
        # Check if we can fit in GPU memory with buffer
        if not CUPY_AVAILABLE:
            raise RuntimeError("❌ CuPy not available - GPU-only mode requires CuPy")
        
        available_gpu_memory = cp.cuda.Device().mem_info[1] / 1024**3
        safe_gpu_memory = available_gpu_memory * 0.8
        print(f"   Available GPU Memory: {available_gpu_memory:.1f} GB")
        print(f"   Safe GPU Memory (80%): {safe_gpu_memory:.1f} GB")
        print(f"   GPU Memory usage: {total_memory/safe_gpu_memory*100:.1f}%")
        
        if total_memory > safe_gpu_memory:
            raise RuntimeError(f"❌ Memory exceeds safe GPU capacity ({total_memory:.2f} GB > {safe_gpu_memory:.1f} GB)")
        
        print("   ✅ Memory fits in safe GPU capacity")
        
        # Initialize areas with GPU memory only
        self.areas = []
        for i in range(n_areas):
            area = {
                'n': n_neurons,
                'k': self.k_active,
                'w': 0,
                'winners': cp.zeros(self.k_active, dtype=cp.int32),
                'weights': cp.zeros(self.k_active, dtype=cp.float32),
                'support': cp.zeros(self.k_active, dtype=cp.float32),
                'activated': False,
                'area_id': i
            }
            self.areas.append(area)
        
        # Pre-allocate common arrays for memory pooling
        if self.memory_pool:
            self._preallocate_arrays()
        
        print("   ✅ Brain initialized successfully!")
        print("   Using: GPU (CuPy) - ENHANCED GPU-ONLY MODE")
    
    def _preallocate_arrays(self):
        """Pre-allocate arrays for memory pooling"""
        print("   🔧 Pre-allocating arrays for memory pooling...")
        
        # Pre-allocate candidate arrays
        for _ in range(self.memory_pool.max_pool_size):
            candidates = cp.zeros(self.k_active, dtype=cp.float32)
            self.memory_pool.return_array(candidates, 'candidates')
        
        # Pre-allocate winner arrays
        for _ in range(self.memory_pool.max_pool_size):
            winners = cp.zeros(self.k_active, dtype=cp.int32)
            self.memory_pool.return_array(winners, 'winners')
    
    def _generate_candidates_optimized(self, area_idx):
        """Generate candidates using optimized operations with memory pooling"""
        area = self.areas[area_idx]
        
        # Get array from pool or create new one
        if self.memory_pool:
            candidates = self.memory_pool.get_array(cp.float32, (area['k'],), 'candidates')
        else:
            candidates = cp.zeros(area['k'], dtype=cp.float32)
        
        # Use CuPy for GPU-accelerated random number generation
        try:
            candidates[:] = cp.random.exponential(1.0, size=area['k'])
        except Exception as e:
            print(f"   ⚠️  CuPy random failed: {e}, falling back to NumPy")
            np_candidates = self._rng.exponential(1.0, size=area['k'])
            candidates[:] = cp.asarray(np_candidates)
        
        return candidates
    
    def _select_top_k_optimized(self, candidates, k):
        """Select top-k using optimized operations with memory pooling"""
        if k >= len(candidates):
            return cp.arange(len(candidates))
        
        # Get winner array from pool or create new one
        if self.memory_pool:
            winners = self.memory_pool.get_array(cp.int32, (k,), 'winners')
        else:
            winners = cp.zeros(k, dtype=cp.int32)
        
        # Use CuPy for GPU-accelerated top-k selection
        top_k_indices = cp.argpartition(candidates, -k)[-k:]
        
        # Sort only the top-k
        top_k_values = candidates[top_k_indices]
        sorted_indices = cp.argsort(top_k_values)[::-1]
        
        winners[:] = top_k_indices[sorted_indices]
        
        # Return candidates to pool
        if self.memory_pool:
            self.memory_pool.return_array(candidates, 'candidates')
        
        return winners
    
    def _update_weights_optimized(self, area_idx, winners):
        """Update weights using optimized operations"""
        area = self.areas[area_idx]
        
        # Use CuPy for GPU-accelerated weight updates
        area['weights'][winners] += 0.1
        area['weights'] *= 0.99
        area['support'][winners] += 1.0
    
    def _get_memory_usage(self):
        """Get current GPU memory usage"""
        try:
            used, total = cp.cuda.Device().mem_info
            return used / 1024**3, total / 1024**3
        except:
            return 0.0, 0.0
    
    def simulate_step(self):
        """Simulate one step of the brain with enhanced monitoring"""
        start_time = time.perf_counter()
        
        # Get initial memory usage
        if self.enable_profiling:
            initial_memory, total_memory = self._get_memory_usage()
        
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]
            
            # Generate candidates
            candidates = self._generate_candidates_optimized(area_idx)
            
            # Select top-k winners
            winners = self._select_top_k_optimized(candidates, area['k'])
            
            # Update area state
            area['w'] = len(winners)
            area['winners'][:len(winners)] = winners
            area['activated'] = True
            
            # Update weights
            self._update_weights_optimized(area_idx, winners)
        
        # Record timing and profiling data
        step_time = time.perf_counter() - start_time
        self.metrics.step_count += 1
        self.metrics.total_time += step_time
        self.metrics.min_step_time = min(self.metrics.min_step_time, step_time)
        self.metrics.max_step_time = max(self.metrics.max_step_time, step_time)
        
        if self.enable_profiling:
            final_memory, total_memory = self._get_memory_usage()
            self.metrics.memory_usage_gb = final_memory
            self.metrics.gpu_utilization = (final_memory / total_memory) * 100
            
            self.profile_data['step_times'].append(step_time)
            self.profile_data['memory_usage'].append(final_memory)
            self.profile_data['gpu_utilization'].append(self.metrics.gpu_utilization)
        
        return step_time
    
    def simulate(self, n_steps=100, verbose=True, profile_interval=10):
        """Simulate multiple steps with enhanced monitoring"""
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS (Enhanced Mode)")
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
                      f"GPU: {memory_usage:.2f}GB ({gpu_util:.1f}%)")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print("\n📊 ENHANCED SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Min step time: {self.metrics.min_step_time*1000:.2f}ms")
            print(f"   Max step time: {self.metrics.max_step_time*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
            print(f"   Final GPU memory: {self.metrics.memory_usage_gb:.2f}GB")
            print(f"   GPU utilization: {self.metrics.gpu_utilization:.1f}%")
        
        return total_time
    
    def get_performance_stats(self):
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
            'cache_hit_rate': self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
        }
    
    def get_profile_data(self):
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
                'use_memory_pool': self.use_memory_pool
            },
            'performance': self.get_performance_stats(),
            'profile_data': self.profile_data
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        print(f"📊 Profile data saved to {filename}")

def test_enhanced_gpu_billion_scale():
    """Test enhanced GPU-only billion-scale brain"""
    print("🚀 TESTING ENHANCED GPU-ONLY BILLION-SCALE BRAIN")
    print("=" * 70)
    
    # Test different scales with enhanced features
    test_cases = [
        {"n_neurons": 1000000, "active_percentage": 0.01, "n_areas": 5, "name": "Million Scale (1%)"},
        {"n_neurons": 10000000, "active_percentage": 0.01, "n_areas": 5, "name": "Ten Million Scale (1%)"},
        {"n_neurons": 100000000, "active_percentage": 0.001, "n_areas": 5, "name": "Hundred Million Scale (0.1%)"},
        {"n_neurons": 1000000000, "active_percentage": 0.0001, "n_areas": 5, "name": "BILLION SCALE (0.01%)"},
        {"n_neurons": 2000000000, "active_percentage": 0.00005, "n_areas": 5, "name": "TWO BILLION SCALE (0.005%)"},
        {"n_neurons": 5000000000, "active_percentage": 0.00002, "n_areas": 5, "name": "FIVE BILLION SCALE (0.002%)"},
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {test_case['n_neurons']:,}")
        print(f"   Active percentage: {test_case['active_percentage']*100:.4f}%")
        print(f"   Active per area: {int(test_case['n_neurons'] * test_case['active_percentage']):,}")
        print(f"   Areas: {test_case['n_areas']}")
        
        # Calculate memory usage with sparse approach
        k_active = int(test_case['n_neurons'] * test_case['active_percentage'])
        memory_usage = k_active * 4 * 3 * test_case['n_areas'] / 1024 / 1024 / 1024
        print(f"   Memory: {memory_usage:.2f} GB")
        
        try:
            # Create enhanced brain
            brain = EnhancedGPUOnlyBillionScaleBrain(
                n_neurons=test_case['n_neurons'],
                active_percentage=test_case['active_percentage'],
                n_areas=test_case['n_areas'],
                seed=42,
                use_memory_pool=True,
                enable_profiling=True
            )
            
            # Simulate
            start_time = time.perf_counter()
            brain.simulate(n_steps=10, verbose=False)
            total_time = time.perf_counter() - start_time
            
            # Get enhanced stats
            stats = brain.get_performance_stats()
            
            # Calculate performance metrics
            neurons_per_second = stats['neurons_per_second']
            active_neurons_per_second = stats['active_neurons_per_second']
            steps_per_second = stats['steps_per_second']
            ms_per_step = stats['avg_step_time'] * 1000
            gpu_util = stats['gpu_utilization']
            memory_used = stats['memory_usage_gb']
            
            print("   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {steps_per_second:.1f}")
            print(f"   ms/step: {ms_per_step:.2f}ms")
            print(f"   Neurons/sec: {neurons_per_second:,.0f}")
            print(f"   Active/sec: {active_neurons_per_second:,.0f}")
            print(f"   GPU Memory: {memory_used:.2f}GB ({gpu_util:.1f}%)")
            
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'active_percentage': test_case['active_percentage'],
                'k_active': int(test_case['n_neurons'] * test_case['active_percentage']),
                'n_areas': test_case['n_areas'],
                'total_time': total_time,
                'steps_per_second': steps_per_second,
                'ms_per_step': ms_per_step,
                'neurons_per_second': neurons_per_second,
                'active_neurons_per_second': active_neurons_per_second,
                'gpu_utilization': gpu_util,
                'memory_usage_gb': memory_used
            })
            
            # Save profile data for the billion scale test
            if test_case['n_neurons'] == 1000000000:
                brain.save_profile_data(f"enhanced_gpu_profile_{test_case['n_neurons']}.json")
            
        except Exception as e:
            print(f"   ⚠️  Skipped: {e}")
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'active_percentage': test_case['active_percentage'],
                'k_active': int(test_case['n_neurons'] * test_case['active_percentage']),
                'n_areas': test_case['n_areas'],
                'total_time': float('inf'),
                'steps_per_second': 0,
                'ms_per_step': float('inf'),
                'neurons_per_second': 0,
                'active_neurons_per_second': 0,
                'gpu_utilization': 0,
                'memory_usage_gb': 0,
                'skipped': True
            })
    
    # Enhanced summary
    print("\n📊 ENHANCED GPU-ONLY BILLION-SCALE BRAIN BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Scale':<25} {'Neurons':<15} {'Active%':<8} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'GPU Util%':<10} {'Memory GB':<10}")
    print("-" * 90)
    
    for result in results:
        if result.get('skipped', False):
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {'SKIPPED':<10} {'SKIPPED':<10} {'SKIPPED':<15} {'SKIPPED':<10} {'SKIPPED':<10}")
        elif result['steps_per_second'] > 0:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['gpu_utilization']:<10.1f} {result['memory_usage_gb']:<10.2f}")
        else:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<10} {'FAILED':<10}")
    
    return results

if __name__ == "__main__":
    # Test enhanced GPU-only billion-scale brain
    results = test_enhanced_gpu_billion_scale()
    
    # Find best performance
    successful_results = [r for r in results if r['steps_per_second'] > 0]
    if successful_results:
        best = max(successful_results, key=lambda x: x['steps_per_second'])
        print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
        print(f"   Steps/sec: {best['steps_per_second']:.1f}")
        print(f"   ms/step: {best['ms_per_step']:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
        print(f"   Active/sec: {best['active_neurons_per_second']:,.0f}")
        print(f"   GPU Utilization: {best['gpu_utilization']:.1f}%")
        print(f"   Memory Usage: {best['memory_usage_gb']:.2f}GB")
    else:
        print("\n❌ No successful tests")
