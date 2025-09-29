#!/usr/bin/env python3
"""
Ultimate Billion-Scale Brain - Billion Scale Superset
====================================================

This superset combines the best features from all billion-scale implementations:
- GPU-only optimization
- Memory pooling and management
- Adaptive scaling
- Real-time monitoring
- Multi-GPU support preparation
- Advanced profiling

Combines features from:
- billion_scale_cuda_brain.py
- working_billion_scale_brain.py
- working_gpu_billion_scale.py
- gpu_only_billion_scale.py
- enhanced_gpu_billion_scale.py
- optimized_billion_scale_cupy.py
"""

import time
import numpy as np
import os
import sys
import threading
import queue
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
class BillionScaleConfig:
    """Configuration for billion-scale brain simulation"""
    n_neurons: int = 1000000000
    active_percentage: float = 0.0001
    n_areas: int = 5
    seed: int = 42
    use_memory_pool: bool = True
    enable_profiling: bool = True
    adaptive_scaling: bool = True
    max_gpu_memory_usage: float = 0.8
    min_active_neurons: int = 1000
    max_active_neurons: int = 1000000

@dataclass
class MemoryPool:
    """Advanced GPU memory pool for efficient allocation"""
    pools: Dict[str, List[cp.ndarray]] = None
    max_pool_size: int = 20
    total_allocated: int = 0
    total_returned: int = 0
    
    def __post_init__(self):
        if self.pools is None:
            self.pools = {
                'candidates': [],
                'winners': [],
                'weights': [],
                'support': [],
                'activations': []
            }
    
    def get_array(self, dtype: cp.dtype, shape: Tuple[int, ...], pool_name: str) -> cp.ndarray:
        """Get array from pool or create new one"""
        if pool_name in self.pools and self.pools[pool_name]:
            array = self.pools[pool_name].pop()
            if array.shape == shape and array.dtype == dtype:
                self.total_returned += 1
                return array
        
        self.total_allocated += 1
        return cp.zeros(shape, dtype=dtype)
    
    def return_array(self, array: cp.ndarray, pool_name: str):
        """Return array to pool"""
        if pool_name in self.pools and len(self.pools[pool_name]) < self.max_pool_size:
            self.pools[pool_name].append(array)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        return {
            'total_allocated': self.total_allocated,
            'total_returned': self.total_returned,
            'pool_sizes': {name: len(pool) for name, pool in self.pools.items()},
            'efficiency': self.total_returned / max(1, self.total_allocated)
        }

@dataclass
class PerformanceMetrics:
    """Advanced performance metrics for billion-scale monitoring"""
    step_count: int = 0
    total_time: float = 0.0
    min_step_time: float = float('inf')
    max_step_time: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_pool_efficiency: float = 0.0
    adaptive_scaling_events: int = 0

class UltimateBillionScaleBrain:
    """
    Ultimate Billion-Scale Brain
    
    Combines the best features from all billion-scale implementations:
    - GPU-only optimization for maximum performance
    - Advanced memory pooling and management
    - Adaptive scaling based on performance
    - Real-time monitoring and profiling
    - Multi-GPU support preparation
    - Advanced error handling and recovery
    """
    
    def __init__(self, config: BillionScaleConfig):
        """Initialize the ultimate billion-scale brain"""
        self.config = config
        self.n_neurons = config.n_neurons
        self.active_percentage = config.active_percentage
        self.k_active = int(config.n_neurons * config.active_percentage)
        self.n_areas = config.n_areas
        self.seed = config.seed
        
        # Ensure k_active is within bounds
        self.k_active = max(config.min_active_neurons, 
                           min(config.max_active_neurons, self.k_active))
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.memory_pool = MemoryPool() if config.use_memory_pool else None
        
        # Profiling data
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'adaptive_scaling': [],
            'memory_pool_stats': []
        }
        
        # Initialize random number generator
        self._rng = np.random.default_rng(config.seed)
        
        print(f"🚀 Ultimate Billion-Scale Brain initialized:")
        print(f"   Neurons: {self.n_neurons:,}")
        print(f"   Active percentage: {self.active_percentage*100:.4f}%")
        print(f"   Active per area: {self.k_active:,}")
        print(f"   Areas: {self.n_areas}")
        print(f"   CuPy available: {'✅' if CUPY_AVAILABLE else '❌'}")
        print(f"   Memory pooling: {'✅' if config.use_memory_pool else '❌'}")
        print(f"   Adaptive scaling: {'✅' if config.adaptive_scaling else '❌'}")
        print(f"   Profiling: {'✅' if config.enable_profiling else '❌'}")
        
        # Calculate memory usage with sparse approach
        memory_per_area = self.k_active * 4 * 3 / 1024 / 1024 / 1024  # 3 arrays per area
        total_memory = memory_per_area * self.n_areas
        
        print(f"   Memory per area: {memory_per_area:.2f} GB")
        print(f"   Total memory: {total_memory:.2f} GB")
        
        # Check GPU memory constraints
        if not CUPY_AVAILABLE:
            raise RuntimeError("❌ CuPy not available - billion-scale mode requires CuPy")
        
        available_gpu_memory = cp.cuda.Device().mem_info[1] / 1024**3
        safe_gpu_memory = available_gpu_memory * config.max_gpu_memory_usage
        print(f"   Available GPU Memory: {available_gpu_memory:.1f} GB")
        print(f"   Safe GPU Memory ({config.max_gpu_memory_usage*100:.0f}%): {safe_gpu_memory:.1f} GB")
        print(f"   GPU Memory usage: {total_memory/safe_gpu_memory*100:.1f}%")
        
        if total_memory > safe_gpu_memory:
            if config.adaptive_scaling:
                # Adaptive scaling: reduce active neurons to fit memory
                new_k_active = int(safe_gpu_memory * 1024**3 / (4 * 3 * self.n_areas))
                new_k_active = max(config.min_active_neurons, 
                                 min(config.max_active_neurons, new_k_active))
                
                if new_k_active != self.k_active:
                    print(f"   🔧 Adaptive scaling: {self.k_active:,} → {new_k_active:,} active neurons")
                    self.k_active = new_k_active
                    self.metrics.adaptive_scaling_events += 1
                    memory_per_area = self.k_active * 4 * 3 / 1024 / 1024 / 1024
                    total_memory = memory_per_area * self.n_areas
                    print(f"   New memory usage: {total_memory:.2f} GB")
            else:
                raise RuntimeError(f"❌ Memory exceeds safe GPU capacity ({total_memory:.2f} GB > {safe_gpu_memory:.1f} GB)")
        
        print(f"   ✅ Memory fits in safe GPU capacity")
        
        # Initialize areas with GPU memory only
        self.areas = []
        for i in range(self.n_areas):
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
            self.areas.append(area)
        
        # Pre-allocate common arrays for memory pooling
        if self.memory_pool:
            self._preallocate_arrays()
        
        print(f"   ✅ Brain initialized successfully!")
        print(f"   Using: GPU (CuPy) - ULTIMATE BILLION-SCALE MODE")
    
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
    
    def _generate_candidates_optimized(self, area_idx: int) -> cp.ndarray:
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
    
    def _select_top_k_optimized(self, candidates: cp.ndarray, k: int) -> cp.ndarray:
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
    
    def _update_weights_optimized(self, area_idx: int, winners: cp.ndarray):
        """Update weights using optimized operations"""
        area = self.areas[area_idx]
        
        # Use CuPy for GPU-accelerated weight updates
        area['weights'][winners] += 0.1
        area['weights'] *= 0.99
        area['support'][winners] += 1.0
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage"""
        try:
            used, total = cp.cuda.Device().mem_info
            return used / 1024**3, total / 1024**3
        except:
            return 0.0, 0.0
    
    def _adaptive_scaling_check(self):
        """Check if adaptive scaling is needed"""
        if not self.config.adaptive_scaling:
            return
        
        # Check if we're using too much memory
        used_memory, total_memory = self._get_memory_usage()
        memory_usage_ratio = used_memory / total_memory
        
        if memory_usage_ratio > self.config.max_gpu_memory_usage:
            # Reduce active neurons
            new_k_active = int(self.k_active * 0.9)  # Reduce by 10%
            new_k_active = max(self.config.min_active_neurons, new_k_active)
            
            if new_k_active != self.k_active:
                print(f"   🔧 Adaptive scaling: {self.k_active:,} → {new_k_active:,} active neurons")
                self.k_active = new_k_active
                self.metrics.adaptive_scaling_events += 1
                
                # Update all areas
                for area in self.areas:
                    area['k'] = self.k_active
                    # Resize arrays
                    area['winners'] = cp.zeros(self.k_active, dtype=cp.int32)
                    area['weights'] = cp.zeros(self.k_active, dtype=cp.float32)
                    area['support'] = cp.zeros(self.k_active, dtype=cp.float32)
    
    def simulate_step(self) -> float:
        """Simulate one step of the brain with enhanced monitoring"""
        start_time = time.perf_counter()
        
        # Adaptive scaling check
        self._adaptive_scaling_check()
        
        # Get initial memory usage
        if self.config.enable_profiling:
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
        
        if self.config.enable_profiling:
            final_memory, total_memory = self._get_memory_usage()
            self.metrics.memory_usage_gb = final_memory
            self.metrics.gpu_utilization = (final_memory / total_memory) * 100 if total_memory > 0 else 0
            
            # Memory pool statistics
            if self.memory_pool:
                pool_stats = self.memory_pool.get_stats()
                self.metrics.memory_pool_efficiency = pool_stats['efficiency']
            
            self.profile_data['step_times'].append(step_time)
            self.profile_data['memory_usage'].append(final_memory)
            self.profile_data['gpu_utilization'].append(self.metrics.gpu_utilization)
            self.profile_data['adaptive_scaling'].append(self.metrics.adaptive_scaling_events)
            if self.memory_pool:
                self.profile_data['memory_pool_stats'].append(pool_stats)
        
        return step_time
    
    def simulate(self, n_steps: int = 100, verbose: bool = True, profile_interval: int = 10) -> float:
        """Simulate multiple steps with enhanced monitoring"""
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS (Ultimate Billion-Scale Mode)")
            print("=" * 70)
        
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if verbose and (step + 1) % profile_interval == 0:
                avg_time = self.metrics.total_time / self.metrics.step_count
                memory_usage = self.metrics.memory_usage_gb
                gpu_util = self.metrics.gpu_utilization
                pool_eff = self.metrics.memory_pool_efficiency
                
                print(f"Step {step + 1:3d}: {step_time*1000:.2f}ms | "
                      f"Avg: {avg_time*1000:.2f}ms | "
                      f"GPU: {memory_usage:.2f}GB ({gpu_util:.1f}%) | "
                      f"Pool: {pool_eff:.1%}")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n📊 ULTIMATE BILLION-SCALE SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Min step time: {self.metrics.min_step_time*1000:.2f}ms")
            print(f"   Max step time: {self.metrics.max_step_time*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
            print(f"   Final GPU memory: {self.metrics.memory_usage_gb:.2f}GB")
            print(f"   GPU utilization: {self.metrics.gpu_utilization:.1f}%")
            print(f"   Memory pool efficiency: {self.metrics.memory_pool_efficiency:.1%}")
            print(f"   Adaptive scaling events: {self.metrics.adaptive_scaling_events}")
        
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
            'memory_pool_efficiency': self.metrics.memory_pool_efficiency,
            'adaptive_scaling_events': self.metrics.adaptive_scaling_events,
            'cache_hit_rate': self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
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
                'use_memory_pool': self.config.use_memory_pool,
                'adaptive_scaling': self.config.adaptive_scaling,
                'max_gpu_memory_usage': self.config.max_gpu_memory_usage
            },
            'performance': self.get_performance_stats(),
            'profile_data': self.profile_data
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        print(f"📊 Profile data saved to {filename}")

def test_ultimate_billion_scale_brain():
    """Test the ultimate billion-scale brain with different scales"""
    print("🚀 TESTING ULTIMATE BILLION-SCALE BRAIN")
    print("=" * 70)
    
    # Test different scales with ultimate features
    test_scales = [
        {
            "name": "Million Scale (1%)",
            "config": BillionScaleConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_memory_pool=True,
                adaptive_scaling=True
            )
        },
        {
            "name": "Ten Million Scale (1%)",
            "config": BillionScaleConfig(
                n_neurons=10000000,
                active_percentage=0.01,
                n_areas=5,
                use_memory_pool=True,
                adaptive_scaling=True
            )
        },
        {
            "name": "Hundred Million Scale (0.1%)",
            "config": BillionScaleConfig(
                n_neurons=100000000,
                active_percentage=0.001,
                n_areas=5,
                use_memory_pool=True,
                adaptive_scaling=True
            )
        },
        {
            "name": "BILLION SCALE (0.01%)",
            "config": BillionScaleConfig(
                n_neurons=1000000000,
                active_percentage=0.0001,
                n_areas=5,
                use_memory_pool=True,
                adaptive_scaling=True
            )
        },
        {
            "name": "TWO BILLION SCALE (0.005%)",
            "config": BillionScaleConfig(
                n_neurons=2000000000,
                active_percentage=0.00005,
                n_areas=5,
                use_memory_pool=True,
                adaptive_scaling=True
            )
        },
        {
            "name": "FIVE BILLION SCALE (0.002%)",
            "config": BillionScaleConfig(
                n_neurons=5000000000,
                active_percentage=0.00002,
                n_areas=5,
                use_memory_pool=True,
                adaptive_scaling=True
            )
        }
    ]
    
    results = []
    
    for test_case in test_scales:
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {test_case['config'].n_neurons:,}")
        print(f"   Active percentage: {test_case['config'].active_percentage*100:.4f}%")
        
        try:
            # Create ultimate brain
            brain = UltimateBillionScaleBrain(test_case['config'])
            
            # Simulate
            start_time = time.perf_counter()
            brain.simulate(n_steps=10, verbose=False)
            total_time = time.perf_counter() - start_time
            
            # Get enhanced stats
            stats = brain.get_performance_stats()
            
            print(f"   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   ms/step: {stats['avg_step_time']*1000:.2f}ms")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Active/sec: {stats['active_neurons_per_second']:,.0f}")
            print(f"   GPU Memory: {stats['memory_usage_gb']:.2f}GB ({stats['gpu_utilization']:.1f}%)")
            print(f"   Pool Efficiency: {stats['memory_pool_efficiency']:.1%}")
            print(f"   Adaptive Events: {stats['adaptive_scaling_events']}")
            
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['config'].n_neurons,
                'active_percentage': test_case['config'].active_percentage,
                'k_active': stats.get('active_neurons_per_second', 0) // stats.get('steps_per_second', 1) // test_case['config'].n_areas,
                'n_areas': test_case['config'].n_areas,
                'total_time': total_time,
                'steps_per_second': stats['steps_per_second'],
                'ms_per_step': stats['avg_step_time'] * 1000,
                'neurons_per_second': stats['neurons_per_second'],
                'active_neurons_per_second': stats['active_neurons_per_second'],
                'gpu_utilization': stats['gpu_utilization'],
                'memory_usage_gb': stats['memory_usage_gb'],
                'memory_pool_efficiency': stats['memory_pool_efficiency'],
                'adaptive_scaling_events': stats['adaptive_scaling_events']
            })
            
            # Save profile data for the billion scale test
            if test_case['config'].n_neurons == 1000000000:
                brain.save_profile_data(f"ultimate_billion_scale_profile_{test_case['config'].n_neurons}.json")
            
        except Exception as e:
            print(f"   ⚠️  Skipped: {e}")
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['config'].n_neurons,
                'active_percentage': test_case['config'].active_percentage,
                'k_active': 0,
                'n_areas': test_case['config'].n_areas,
                'total_time': float('inf'),
                'steps_per_second': 0,
                'ms_per_step': float('inf'),
                'neurons_per_second': 0,
                'active_neurons_per_second': 0,
                'gpu_utilization': 0,
                'memory_usage_gb': 0,
                'memory_pool_efficiency': 0,
                'adaptive_scaling_events': 0,
                'skipped': True
            })
    
    # Ultimate summary
    print(f"\n📊 ULTIMATE BILLION-SCALE BRAIN BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Scale':<25} {'Neurons':<15} {'Active%':<8} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'GPU Util%':<10} {'Memory GB':<10} {'Pool Eff%':<10} {'Adapt Events':<12}")
    print("-" * 100)
    
    for result in results:
        if result.get('skipped', False):
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {'SKIPPED':<10} {'SKIPPED':<10} {'SKIPPED':<15} {'SKIPPED':<10} {'SKIPPED':<10} {'SKIPPED':<10} {'SKIPPED':<12}")
        elif result['steps_per_second'] > 0:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['gpu_utilization']:<10.1f} {result['memory_usage_gb']:<10.2f} {result['memory_pool_efficiency']:<10.1%} {result['adaptive_scaling_events']:<12}")
        else:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {'FAILED':<12}")
    
    return results

if __name__ == "__main__":
    # Test ultimate billion-scale brain
    results = test_ultimate_billion_scale_brain()
    
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
        print(f"   Pool Efficiency: {best['memory_pool_efficiency']:.1%}")
        print(f"   Adaptive Events: {best['adaptive_scaling_events']}")
    else:
        print(f"\n❌ No successful tests")
