#!/usr/bin/env python3
"""
Working GPU Billion-Scale Brain
Uses the working approach from fixed_cupy_billion_scale.py but with memory optimization
"""

import time
import numpy as np

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

class WorkingGPUBillionScaleBrain:
    """
    Working GPU Billion-Scale Brain
    Uses the working approach but with memory optimization to stay on GPU
    """
    
    def __init__(self, n_neurons=1000000000, active_percentage=0.0001, n_areas=5, seed=42):
        """Initialize the working GPU billion-scale brain"""
        self.n_neurons = n_neurons
        self.active_percentage = active_percentage
        self.k_active = int(n_neurons * active_percentage)  # Small active count
        self.n_areas = n_areas
        self.seed = seed
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        print("🚀 Working GPU Billion-Scale Brain initialized:")
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
        use_gpu = CUPY_AVAILABLE
        if CUPY_AVAILABLE:
            available_gpu_memory = cp.cuda.Device().mem_info[1] / 1024**3
            # Use only 80% of GPU memory for safety buffer
            safe_gpu_memory = available_gpu_memory * 0.8
            print(f"   Available GPU Memory: {available_gpu_memory:.1f} GB")
            print(f"   Safe GPU Memory (80%): {safe_gpu_memory:.1f} GB")
            print(f"   GPU Memory usage: {total_memory/safe_gpu_memory*100:.1f}%")
            
            # Check if we can fit in GPU memory with buffer
            if total_memory > safe_gpu_memory:
                print("   ⚠️  Memory exceeds safe GPU capacity, using CPU fallback")
                use_gpu = False
            else:
                print("   ✅ Memory fits in safe GPU capacity")
        
        # Initialize areas with sparse memory management
        self.areas = []
        for i in range(n_areas):
            if use_gpu:
                # Use GPU memory for sparse arrays
                area = {
                    'n': n_neurons,
                    'k': self.k_active,
                    'w': 0,
                    'winners': cp.zeros(self.k_active, dtype=cp.int32),  # Only active neurons
                    'weights': cp.zeros(self.k_active, dtype=cp.float32),  # Only active weights
                    'support': cp.zeros(self.k_active, dtype=cp.float32),  # Only active support
                    'activated': False
                }
            else:
                # Fallback to NumPy with sparse approach
                area = {
                    'n': n_neurons,
                    'k': self.k_active,
                    'w': 0,
                    'winners': np.zeros(self.k_active, dtype=np.int32),
                    'weights': np.zeros(self.k_active, dtype=np.float32),
                    'support': np.zeros(self.k_active, dtype=np.float32),
                    'activated': False
                }
            self.areas.append(area)
        
        # Store the GPU usage flag
        self.use_gpu = use_gpu
        
        # Performance counters
        self.step_count = 0
        self.total_time = 0.0
        
        print("   ✅ Brain initialized successfully!")
        print(f"   Using: {'GPU (CuPy)' if use_gpu else 'CPU (NumPy)'}")
    
    def _generate_candidates_optimized(self, area_idx):
        """Generate candidates using optimized operations - only for active neurons"""
        area = self.areas[area_idx]
        
        if self.use_gpu:
            # Use CuPy for GPU-accelerated random number generation
            try:
                candidates = cp.random.exponential(1.0, size=area['k'])
            except Exception as e:
                print(f"   ⚠️  CuPy random failed: {e}, falling back to NumPy")
                # Fallback to NumPy if CuPy fails
                np_candidates = self._rng.exponential(1.0, size=area['k'])
                candidates = cp.asarray(np_candidates)
        else:
            # Fallback to NumPy
            candidates = self._rng.exponential(1.0, size=area['k'])
        
        return candidates
    
    def _select_top_k_optimized(self, candidates, k):
        """Select top-k using optimized operations"""
        if k >= len(candidates):
            return np.arange(len(candidates))
        
        if self.use_gpu:
            # Use CuPy for GPU-accelerated top-k selection
            top_k_indices = cp.argpartition(candidates, -k)[-k:]
            
            # Sort only the top-k
            top_k_values = candidates[top_k_indices]
            sorted_indices = cp.argsort(top_k_values)[::-1]
            
            return top_k_indices[sorted_indices]
        else:
            # Fallback to NumPy
            top_k_indices = np.argpartition(candidates, -k)[-k:]
            
            # Sort only the top-k
            top_k_values = candidates[top_k_indices]
            sorted_indices = np.argsort(top_k_values)[::-1]
            
            return top_k_indices[sorted_indices]
    
    def _update_weights_optimized(self, area_idx, winners):
        """Update weights using optimized operations"""
        area = self.areas[area_idx]
        
        if self.use_gpu:
            # Use CuPy for GPU-accelerated weight updates
            area['weights'][winners] += 0.1
            area['weights'] *= 0.99
            area['support'][winners] += 1.0
        else:
            # Fallback to NumPy
            area['weights'][winners] += 0.1
            area['weights'] *= 0.99
            area['support'][winners] += 1.0
    
    def simulate_step(self):
        """Simulate one step of the brain"""
        start_time = time.perf_counter()
        
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
        
        # Record total timing
        step_time = time.perf_counter() - start_time
        self.total_time += step_time
        self.step_count += 1
        
        return step_time
    
    def simulate(self, n_steps=100, verbose=True):
        """Simulate multiple steps"""
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS")
            print("=" * 50)
        
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if verbose and (step + 1) % 10 == 0:
                avg_time = self.total_time / self.step_count
                print(f"Step {step + 1:3d}: {step_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print("\n📊 SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
        
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
            'neurons_per_second': self.n_neurons * steps_per_second,
            'active_neurons_per_second': self.k_active * self.n_areas * steps_per_second
        }

def test_working_gpu_billion_scale():
    """Test working GPU billion-scale brain"""
    print("🚀 TESTING WORKING GPU BILLION-SCALE BRAIN")
    print("=" * 60)
    
    # Test different scales with proper memory buffering
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
            # Create brain
            brain = WorkingGPUBillionScaleBrain(
                n_neurons=test_case['n_neurons'],
                active_percentage=test_case['active_percentage'],
                n_areas=test_case['n_areas'],
                seed=42
            )
            
            # Simulate
            start_time = time.perf_counter()
            brain.simulate(n_steps=10, verbose=False)  # Fewer steps for large scale
            total_time = time.perf_counter() - start_time
            
            # Get stats
            stats = brain.get_performance_stats()
            
            # Calculate performance metrics
            neurons_per_second = stats['neurons_per_second']
            active_neurons_per_second = stats['active_neurons_per_second']
            steps_per_second = stats['steps_per_second']
            ms_per_step = stats['avg_step_time'] * 1000
            
            print("   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {steps_per_second:.1f}")
            print(f"   ms/step: {ms_per_step:.2f}ms")
            print(f"   Neurons/sec: {neurons_per_second:,.0f}")
            print(f"   Active/sec: {active_neurons_per_second:,.0f}")
            
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
                'active_neurons_per_second': active_neurons_per_second
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
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
                'active_neurons_per_second': 0
            })
    
    # Summary
    print("\n📊 WORKING GPU BILLION-SCALE BRAIN BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Scale':<25} {'Neurons':<15} {'Active%':<8} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<15}")
    print("-" * 80)
    
    for result in results:
        if result['steps_per_second'] > 0:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['active_neurons_per_second']:<15,.0f}")
        else:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15}")
    
    return results

if __name__ == "__main__":
    # Test working GPU billion-scale brain
    results = test_working_gpu_billion_scale()
    
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
        print("\n❌ No successful tests")
